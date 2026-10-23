#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect an original-precision full-song baseline from the existing local service.

No engine, model loading, service management, downloads, playback or quantization.
Completion means native termination and valid objective audio, NOT a listening review.
"""

import argparse
import hashlib
import http.client
import importlib.metadata
import importlib.util
import json
import os
import platform
import plistlib
import re
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import psutil

GIB = 1024**3
REQUEST_TIMEOUT = 1800
MIN_AVAILABLE = 4 * GIB
MAX_TREE_RSS = 14 * GIB
MIN_DISK_FREE = 2 * GIB
MODEL_NAME = "yue2-3b"
SEED = 12300
MAX_AUDIO_FRAMES = 9000
SERVICE_LABEL = "com.local.yue2-3b-metal"
DEFAULT_PLIST = Path.home() / "Library/Application Support/yue2-3b" / f"{SERVICE_LABEL}.plist"
SAMPLE_RELATIVE = Path("examples/tonight-awake.json")


class BaselineFailure(RuntimeError):
    pass


class ResourceFailure(BaselineFailure):
    pass


def utc_now():
    return datetime.now(UTC).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def owned_file(path):
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file() or path.stat().st_uid != os.getuid():
        raise BaselineFailure(f"需要当前用户拥有的本地普通文件：{path}")
    return path


def load_deployment_module(name):
    spec = importlib.util.spec_from_file_location(
        f"_yue2_baseline_{name}", Path(__file__).with_name(f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_identity(path):
    info = Path(path).stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def fingerprint(path):
    path = owned_file(path)
    before = file_identity(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024**2):
            digest.update(chunk)
    if before != file_identity(path):
        raise BaselineFailure(f"指纹读取期间文件改变：{path}")
    return {
        "path": str(path),
        "bytes": before[2],
        "stat_identity": list(before),
        "sha256": digest.hexdigest(),
    }


def checkpoint_files(directory):
    index = directory / "model.safetensors.index.json"
    names = (
        sorted(set(json.loads(index.read_text())["weight_map"].values()))
        if index.is_file()
        else ["model.safetensors"]
    )
    return [directory / name for name in names]


def checkpoint_fingerprint(directory):
    """Hash only the selected checkpoint once per invocation, never once per request."""
    if directory.stat().st_uid != os.getuid():
        raise BaselineFailure(f"模型缓存不属于当前用户：{directory}")
    files = checkpoint_files(directory)
    entries = []
    dtypes = {}
    for path in files:
        entry = fingerprint(path)
        with path.open("rb") as source:
            length = struct.unpack("<Q", source.read(8))[0]
            header = json.loads(source.read(length))
        for name, tensor in header.items():
            if name != "__metadata__":
                dtype = tensor["dtype"]
                dtypes[dtype] = dtypes.get(dtype, 0) + 1
        entries.append(entry)
    metadata_names = [
        "config.json",
        "model.safetensors.index.json",
        "qwen.tiktoken",
        "yue2_generation_config.json",
        "generation_config.json",
        "weights_manifest.json",
        ".mv",
        ".mdl",
    ]
    return {
        "directory": str(directory),
        "weights": entries,
        "tensor_dtype_counts": dtypes,
        # .mdl is opaque cache metadata: fingerprint bytes, NEVER unpickle it.
        "metadata": [
            fingerprint(directory / name) for name in metadata_names if (directory / name).is_file()
        ],
    }


def validate_original_mode(plist, profile, actual_environment):
    expected = {
        "YUE2_METAL_QUANTIZATION": "off",
        "VLLM_OMNI_METAL_ANE": "off",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    configured = plist.get("EnvironmentVariables", {})
    for key, value in expected.items():
        if configured.get(key) != value or actual_environment.get(key) != value:
            raise BaselineFailure(f"原精度基线要求 plist 和实际服务环境 {key}={value}")
    stages = profile.get("stages", [])
    if profile.get("pipeline") != "yue2" or profile.get("dtype") != "bfloat16" or len(stages) != 1:
        raise BaselineFailure("必须保留 YuE2 原生单阶段 BF16 profile")
    stage = stages[0]
    if stage.get("dtype") not in (None, "bfloat16"):
        raise BaselineFailure("stage 不允许覆盖原 BF16 精度")
    if stage.get("max_model_len") != 24576 or stage.get("max_num_seqs") != 1:
        raise BaselineFailure("必须保留 context24576、单请求 profile")
    if (
        stage.get("gpu_memory_utilization") != 0.30
        or stage.get("kv_cache_memory_bytes") is not None
    ):
        raise BaselineFailure("必须保留原 gpu_memory_utilization=0.30，不改变 KV 预算")
    if any(
        config.get(key) is not None
        for config in (profile, stage)
        for key in ("quantization", "quantization_config")
    ):
        raise BaselineFailure("原精度基线不允许量化配置")


def option(arguments, name):
    if arguments.count(name) != 1:
        raise BaselineFailure(f"服务参数缺少或重复：{name}")
    index = arguments.index(name)
    if index + 1 == len(arguments):
        raise BaselineFailure(f"服务参数缺少值：{name}")
    return arguments[index + 1]


def service_environment(plist_path, base_url, output):
    """Read the already running, user-owned launchd service; never start/restart it."""
    import yaml

    plist_path = owned_file(plist_path)
    plist = plistlib.loads(plist_path.read_bytes())
    if plist.get("Label") != SERVICE_LABEL or plist.get("KeepAlive") is not False:
        raise BaselineFailure("不是预期的现有 YuE2 launchd 服务")
    arguments = plist.get("ProgramArguments", [])
    if len(arguments) < 4 or arguments[1] != "serve" or "--omni" not in arguments:
        raise BaselineFailure("不是原生 Omni serve 任务")
    url = urlsplit(base_url)
    if (
        url.scheme != "http"
        or url.hostname not in ("127.0.0.1", "localhost")
        or url.username
        or url.password
        or url.path not in ("", "/")
        or url.query
        or url.fragment
    ):
        raise BaselineFailure("仅允许直接连接现有本机 HTTP 服务，不经过其他代理")
    if option(arguments, "--host") != "127.0.0.1" or int(option(arguments, "--port")) != (
        url.port or 80
    ):
        raise BaselineFailure("URL 与已加载本机服务端口不一致")
    if option(arguments, "--served-model-name") != MODEL_NAME:
        raise BaselineFailure("现有服务不是 yue2-3b")
    state = subprocess.run(
        ["launchctl", "print", f"gui/{os.getuid()}/{SERVICE_LABEL}"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    ).stdout
    (output / "launchctl.txt").write_text(state)
    pid_match = re.search(r"^\s*pid = (\d+)\s*$", state, re.MULTILINE)
    if not pid_match or not re.search(r"^\s*state = running\s*$", state, re.MULTILINE):
        raise BaselineFailure("现有服务未运行；本工具不会启动它")
    pid = int(pid_match[1])
    process = psutil.Process(pid)
    if process.uids().real != os.getuid():
        raise BaselineFailure("现有服务进程不属于当前用户")
    actual_arguments = process.cmdline()
    if actual_arguments[-len(arguments) :] != arguments:
        # Console scripts run as [python, script, ...]; no other command-line differences allowed.
        raise BaselineFailure("运行进程参数与 plist 不一致；拒绝把新配置当作实际环境")
    actual_environment = process.environ()
    selected_environment = {
        key: actual_environment.get(key) for key in plist.get("EnvironmentVariables", {})
    }
    profile_path = owned_file(option(arguments, "--deploy-config"))
    if profile_path.stat().st_mtime > process.create_time():
        raise BaselineFailure("profile 在服务启动后改变，不能证明当前加载配置")
    profile = yaml.safe_load(profile_path.read_text())
    validate_original_mode(plist, profile, actual_environment)
    listener = subprocess.run(
        ["lsof", "-nP", "-a", "-p", str(pid), f"-iTCP:{url.port or 80}", "-sTCP:LISTEN", "-Fp"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    ).stdout
    if f"p{pid}" not in listener.splitlines():
        raise BaselineFailure("目标端口的 listener 不属于所选服务")
    validate_model = load_deployment_module("prepare").validate_model
    model = validate_model(arguments[2], "yue2")
    vae = validate_model(plist["EnvironmentVariables"]["YUE2_VAE"], "yue2_vae")
    if Path(actual_environment.get("YUE2_VAE", "")).resolve() != vae:
        raise BaselineFailure("实际 YUE2_VAE 与 plist 不一致")
    model_config = json.loads((model / "config.json").read_text())
    vae_config = json.loads((vae / "config.json").read_text())
    if (
        model_config.get("dtype") != "bfloat16"
        or model_config.get("max_position_embeddings") != 24576
    ):
        raise BaselineFailure("主 checkpoint 不是原 BF16/context24576")
    if vae_config.get("dtype") != "float32":
        raise BaselineFailure("VAE checkpoint 必须保留 FP32")
    overlay = Path(arguments[0]).parent.parent
    if Path(sys.prefix).resolve() != overlay.resolve():
        raise BaselineFailure(f"请使用现有服务的私有解释器：{overlay / 'bin/python'}")
    omni = importlib.metadata.distribution("vllm-omni")
    if "yue2mps2" not in omni.version:
        raise BaselineFailure("当前解释器没有所选 YuE2 Metal 私有 overlay")
    source_paths = [
        omni.locate_file(name)
        for name in (
            "vllm_omni/model_executor/models/yue2/yue2.py",
            "vllm_omni/tokenizers/yue2_prompt.py",
            "vllm_omni/tokenizers/yue2_tokenizer.py",
            "vllm_omni/entrypoints/openai/tts_adapters/yue2.py",
        )
    ]
    if any(not Path(path).resolve().is_relative_to(overlay.resolve()) for path in source_paths):
        raise BaselineFailure("所选 Omni helpers 不属于服务私有 overlay")
    source = Path(source_paths[0]).read_text()
    if not re.search(r"^ODE_STEPS\s*=\s*32\s*$", source, re.MULTILINE) or not re.search(
        r"^ODE_METHOD\s*=\s*[\"\']midpoint[\"\']\s*$", source, re.MULTILINE
    ):
        raise BaselineFailure("所选原生模型不是32步midpoint ODE；不修改采样")
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "vllm",
            "vllm-metal",
            "vllm-omni",
            "vllm-omni-metal",
            "torch",
            "numpy",
            "soundfile",
            "psutil",
        )
    }
    environment = {
        "observed_at": utc_now(),
        "base_url": base_url,
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "versions": versions,
        "service_pid": pid,
        "service_create_time": process.create_time(),
        "actual_arguments": actual_arguments,
        "actual_environment": selected_environment,
        "plist": plist,
        "profile_path": str(profile_path),
        "profile": profile,
        "model_path": str(model),
        "vae_path": str(vae),
        "precision_evidence": "process environment + loaded profile + checkpoint headers/config; no in-process dtype RPC",
        "configured_precision": {"ar": "bfloat16", "nar": "bfloat16", "vae": "float32"},
        "native_synthesis": {"ode_steps": 32, "ode_method": "midpoint", "audio_fps": 25},
        "service_gpu_memory_bytes": None,
        "service_gpu_memory_note": "No service allocator RPC; client MPS memory is not service GPU memory",
        "limits": {
            "request_seconds": REQUEST_TIMEOUT,
            "min_available_bytes": MIN_AVAILABLE,
            "max_service_tree_rss_bytes": MAX_TREE_RSS,
            "min_disk_free_bytes": MIN_DISK_FREE,
        },
        "new_engine_loaded": False,
        "listening_review": "not_performed",
        "automatic_playback": False,
    }
    write_json(output / "environment.json", environment)
    initial_resources = resource_snapshot(pid, process.create_time(), output)
    write_json(output / "preflight-resources.json", initial_resources)
    enforce_resources(initial_resources)
    fingerprints = {
        "model": checkpoint_fingerprint(model),
        "vae": checkpoint_fingerprint(vae),
        "runtime": [fingerprint(path) for path in source_paths],
        "deployment": [fingerprint(plist_path), fingerprint(profile_path), fingerprint(__file__)],
    }
    write_json(output / "fingerprints.json", fingerprints)
    return environment


def resource_snapshot(pid, create_time, output):
    root = psutil.Process(pid)
    if (
        root.create_time() != create_time
        or not root.is_running()
        or root.status() == psutil.STATUS_ZOMBIE
    ):
        raise ResourceFailure("所选服务退出或 PID 身份改变；不重启服务")
    processes = [root, *root.children(recursive=True)]
    rows = []
    for process in processes:
        try:
            rows.append({"pid": process.pid, "rss_bytes": process.memory_info().rss})
        except psutil.NoSuchProcess:
            if process.pid == pid:
                raise ResourceFailure("服务在资源采样期间退出") from None
    return {
        "observed_at": utc_now(),
        "available_bytes": psutil.virtual_memory().available,
        "service_tree_rss_bytes": sum(row["rss_bytes"] for row in rows),
        "processes": rows,
        "disk_free_bytes": shutil.disk_usage(output).free,
    }


def enforce_resources(snapshot):
    if snapshot["available_bytes"] < MIN_AVAILABLE:
        raise ResourceFailure("available RAM 低于4GiB；仅取消本次HTTP请求")
    if snapshot["service_tree_rss_bytes"] > MAX_TREE_RSS:
        raise ResourceFailure("服务进程树 RSS 超过14GiB；仅取消本次HTTP请求")
    if snapshot["disk_free_bytes"] < MIN_DISK_FREE:
        raise ResourceFailure("磁盘可用空间低于2GiB；仅取消本次HTTP请求")


class CancellableHTTPConnection(http.client.HTTPConnection):
    """Keep the socket even when HTTPResponse owns it (Connection: close)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._socket_lock = threading.Lock()
        self._transport_socket = None
        self._cancelled = False

    def connect(self):
        # Connect is also bounded while its not-yet-connected socket is private
        # to socket.create_connection; subsequent reads retain the production limit.
        request_timeout = self.timeout
        self.timeout = min(5, request_timeout)
        try:
            super().connect()
        finally:
            self.timeout = request_timeout
        self.sock.settimeout(request_timeout)
        with self._socket_lock:
            self._transport_socket = self.sock
            cancelled = self._cancelled
        if cancelled:
            self.cancel()
            raise BaselineFailure("HTTP请求已取消")

    def cancel(self):
        with self._socket_lock:
            self._cancelled = True
            transport = self._transport_socket
        if transport is not None:
            try:
                transport.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        # shutdown, rather than close alone, interrupts a concurrent blocking read.
        self.close()


class RequestGuard:
    """Only the monitoring thread runs in background; HTTP stays on the caller thread."""

    def __init__(self, connection, probe, evidence, *, timeout=REQUEST_TIMEOUT, interval=0.5):
        self.connection = connection
        self.probe = probe
        self.evidence = evidence
        self.timeout = timeout
        self.interval = interval
        self.started = time.monotonic()
        self.done = threading.Event()
        self.error = None
        self.thread = None
        self.samples = 0
        self.minimum_available = None
        self.minimum_disk_free = None
        self.maximum_rss = 0

    def check(self):
        elapsed = time.monotonic() - self.started
        snapshot = {**self.probe(), "elapsed_seconds": elapsed}
        with self.evidence.open("a") as file:
            file.write(json.dumps(snapshot, allow_nan=False) + "\n")
        self.samples += 1
        available, disk = snapshot["available_bytes"], snapshot["disk_free_bytes"]
        self.minimum_available = (
            available if self.minimum_available is None else min(self.minimum_available, available)
        )
        self.minimum_disk_free = (
            disk if self.minimum_disk_free is None else min(self.minimum_disk_free, disk)
        )
        self.maximum_rss = max(self.maximum_rss, snapshot["service_tree_rss_bytes"])
        enforce_resources(snapshot)
        if elapsed >= self.timeout:
            raise ResourceFailure(f"本次请求超过 production HTTP {self.timeout}s 上限")

    def start(self):
        self.check()
        self.thread = threading.Thread(target=self._watch, name="yue2-request-resource-guard")
        self.thread.start()

    def _watch(self):
        while not self.done.wait(self.interval):
            try:
                self.check()
            except BaseException as error:
                self.error = error
                self.connection.cancel()
                return

    def stop(self):
        self.done.set()
        if self.thread is not None:
            self.thread.join()

    def summary(self):
        return {
            "samples": self.samples,
            "minimum_available_bytes": self.minimum_available,
            "minimum_disk_free_bytes": self.minimum_disk_free,
            "maximum_service_tree_rss_bytes": self.maximum_rss,
            "error": str(self.error) if self.error else None,
            "monitor_joined": self.thread is None or not self.thread.is_alive(),
        }


def http_request(
    base_url,
    endpoint,
    payload,
    directory,
    probe,
    *,
    body_name="response.json",
    timeout=REQUEST_TIMEOUT,
):
    directory.mkdir(parents=True, exist_ok=False)
    url = urlsplit(base_url)
    connection = CancellableHTTPConnection(url.hostname, url.port or 80, timeout=timeout)
    guard = RequestGuard(connection, probe, directory / "resources.jsonl", timeout=timeout)
    request_id = "yue2-baseline-" + uuid.uuid4().hex
    method = "GET" if payload is None else "POST"
    write_json(
        directory / "request.json",
        {
            "method": method,
            "endpoint": endpoint,
            "base_url": base_url,
            "request_id": request_id,
            "body": payload,
            "timeout_seconds": timeout,
        },
    )
    result = {
        "started_at": utc_now(),
        "request_id": request_id,
        "status": "failed",
        "cancellation_requested": False,
        "server_cancellation_acknowledged": False,
    }
    started = time.monotonic()
    partial = directory / "response-body.partial"
    response = None
    try:
        guard.start()
        body = (
            None
            if payload is None
            else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        )
        connection.request(
            method,
            endpoint,
            body=body,
            headers={
                "Content-Type": "application/json",
                "X-Request-Id": request_id,
                "Connection": "close",
            },
        )
        response = connection.getresponse()
        pairs = response.getheaders()
        headers = {}
        for key, value in pairs:
            key = key.lower()
            headers[key] = value if key not in headers else headers[key] + ", " + value
        write_json(
            directory / "response-headers.json",
            {"status": response.status, "reason": response.reason, "headers": pairs},
        )
        result.update(http_status=response.status, headers=headers)
        digest = hashlib.sha256()
        count = 0
        with partial.open("wb") as file:
            while chunk := response.read(1024**2):
                if guard.error is not None:
                    raise guard.error
                file.write(chunk)
                digest.update(chunk)
                count += len(chunk)
        if guard.error is not None:
            raise guard.error
        length = headers.get("content-length")
        if length is not None and (not length.isdecimal() or int(length) != count):
            raise BaselineFailure("HTTP Content-Length 与实际响应body长度不一致")
        target = directory / (body_name if response.status == 200 else "response-error.bin")
        partial.replace(target)
        result.update(body_path=str(target), bytes=count, sha256=digest.hexdigest())
        if response.status != 200:
            raise BaselineFailure(f"{endpoint}: HTTP {response.status}；保留原始错误body")
        guard.stop()
        guard.check()
        result["status"] = "received"
    except BaseException as error:
        connection.cancel()
        result.update(
            error_type=type(error).__name__,
            error=str(error),
            cancellation_requested=True,
            cancellation_method="shutdown(SHUT_RDWR) of this request's own HTTP socket; no service signals/restart",
        )
        if guard.error is not None and error is not guard.error:
            result.update(error_type=type(guard.error).__name__, error=str(guard.error))
            raise guard.error from error
        raise
    finally:
        guard.stop()
        if response is not None:
            response.close()
        connection.close()
        result.update(elapsed_seconds=time.monotonic() - started, resources=guard.summary())
        write_json(directory / "transport.json", result)
    return result


def analyze_audio(path, *, save_npy=False):
    """Analyze real decoded PCM/float samples in bounded blocks, not full-song copies."""
    import numpy as np
    import soundfile as sf

    finite, peak, squares, count, near_full_scale, outside = True, 0.0, 0.0, 0, 0, 0
    first_active = last_active = None
    offset = 0
    saved = None
    target = Path(path).with_suffix(".npy")
    with sf.SoundFile(path) as audio:
        subtype = audio.subtype
        dtype = "float64" if subtype == "DOUBLE" else "float32"
        info = {
            "format": audio.format,
            "subtype": subtype,
            "sample_rate": audio.samplerate,
            "channels": audio.channels,
            "frames": len(audio),
            "duration_seconds": len(audio) / audio.samplerate,
        }
        # SoundFile normalizes PCM16 to [-1, 32767/32768]; float WAV is not rescaled.
        threshold = 1.0 - 1.0 / 32768 if subtype == "PCM_16" else 1.0
        if save_npy:
            saved = np.lib.format.open_memmap(
                target, mode="w+", dtype=dtype, shape=(len(audio), audio.channels)
            )
        try:
            for samples in audio.blocks(blocksize=65536, dtype=dtype, always_2d=True):
                if saved is not None:
                    saved[offset : offset + len(samples)] = samples
                block_finite = bool(np.isfinite(samples).all())
                finite = finite and block_finite
                count += samples.size
                if block_finite:
                    absolute = np.abs(samples)
                    peak = max(peak, float(absolute.max()))
                    squares += float(np.einsum("ij,ij->", samples, samples, dtype=np.float64))
                    near_full_scale += int(np.count_nonzero(absolute >= threshold))
                    outside += int(np.count_nonzero(absolute > 1.0))
                    active = np.flatnonzero(np.max(absolute, axis=1) > 1e-4)
                    if active.size:
                        if first_active is None:
                            first_active = offset + int(active[0])
                        last_active = offset + int(active[-1])
                offset += len(samples)
        finally:
            if saved is not None:
                saved.flush()
                del saved
    result = {
        **info,
        "decoded_dtype": dtype,
        "finite": finite,
        "listening_review": "not_performed",
        "lyrics_accuracy_review": "not_performed",
    }
    if not finite or count == 0:
        result.update(
            non_silent=False,
            peak_abs=None,
            rms=None,
            near_full_scale_fraction=None,
            out_of_range_fraction=None,
            leading_silence_seconds=None,
            trailing_silence_seconds=None,
        )
    else:
        result.update(
            peak_abs=peak,
            rms=float(np.sqrt(squares / count)),
            near_full_scale_fraction=near_full_scale / count,
            out_of_range_fraction=outside / count,
            silence_threshold_abs=1e-4,
            non_silent=first_active is not None,
            leading_silence_seconds=first_active / info["sample_rate"]
            if first_active is not None
            else info["duration_seconds"],
            trailing_silence_seconds=(info["frames"] - 1 - last_active) / info["sample_rate"]
            if last_active is not None
            else info["duration_seconds"],
        )
    result["valid_objective_audio"] = bool(
        info["format"] in ("WAV", "WAVEX")
        and subtype in ("PCM_16", "FLOAT", "DOUBLE")
        and info["sample_rate"] == 48000
        and info["channels"] == 2
        and 5 <= info["duration_seconds"] <= MAX_AUDIO_FRAMES / 25 + 1
        and finite
        and result["non_silent"]
        and result.get("peak_abs", 0) <= 1.0
    )
    if save_npy:
        result["npy"] = {
            **fingerprint(target),
            "meaning": "decoded HTTP WAV samples, NOT pre-PCM model float output",
        }
    return result


def assess_song(transport, audio):
    headers = transport.get("headers", {})
    flag = headers.get("x-audio-truncated", "").strip().lower()
    mime = headers.get("content-type", "").split(";", 1)[0].strip().lower()
    reasons = []
    if flag not in ("true", "false"):
        reasons.append("missing_or_invalid_X-Audio-Truncated")
    elif flag == "true":
        reasons.append("native_frame_budget_truncated")
    if mime not in ("audio/wav", "audio/x-wav", "audio/wave"):
        reasons.append("unexpected_audio_content_type")
    if transport.get("http_status") != 200 or transport.get("bytes", 0) <= 44:
        reasons.append("invalid_http_audio_body")
    if not audio.get("valid_objective_audio"):
        reasons.append("objective_audio_validation_failed")
    return {
        "status": "complete" if not reasons else "failed",
        "complete": not reasons,
        "truncated": flag == "true" if flag in ("true", "false") else None,
        "naturally_ended": flag == "false",
        "finish_reason": "native_music_end_reported_by_X-Audio-Truncated:false"
        if flag == "false"
        else ("native_frame_budget" if flag == "true" else "unverified"),
        "semantic_token_ids": None,
        "termination_evidence": "HTTP adapter meta.truncated header; speech API does not return semantic tokens",
        "failure_reasons": reasons,
        "audio": audio,
        "listening_review": "not_performed",
        "lyrics_accuracy_review": "not_performed",
    }


def song_request(sample, cot, abc=None):
    extra = {"cot": cot}
    if cot == "full":
        if not isinstance(abc, str) or not abc.strip():
            raise BaselineFailure("full 必须使用本次真实模型生成且自然结束的完整 ABC")
        extra["abc"] = abc
    elif cot != "off" or abc is not None:
        raise BaselineFailure("仅支持 off/full；off 不携带 ABC")
    return {
        "model": MODEL_NAME,
        "input": sample["lyrics"],
        "instructions": sample["style"],
        "seed": SEED,
        "max_new_tokens": MAX_AUDIO_FRAMES,
        "extra_params": extra,
        "response_format": "wav",
        "stream": False,
    }


def generate_song(base_url, sample, cot, directory, probe, *, abc=None, save_npy=False):
    try:
        transport = http_request(
            base_url,
            "/v1/audio/speech",
            song_request(sample, cot, abc),
            directory,
            probe,
            body_name="audio.wav",
        )
        audio = analyze_audio(Path(transport["body_path"]), save_npy=save_npy)
        result = {**assess_song(transport, audio), "cot": cot, "transport": transport}
        write_json(directory / "result.json", result)
        if not result["complete"]:
            raise BaselineFailure(
                f"{cot} 不是完整自然结束歌曲：{result['failure_reasons']}；保留原始WAV"
            )
        return result
    except BaseException as error:
        if directory.exists() and not (directory / "result.json").exists():
            write_json(
                directory / "result.json",
                {
                    "status": "failed",
                    "complete": False,
                    "cot": cot,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "listening_review": "not_performed",
                },
            )
        raise


def generate_plan(base_url, model, sample, directory, probe):
    planning = load_deployment_module("yue2_planning")

    try:
        payload = planning.build_abc_request(model, sample, model_name=MODEL_NAME)
        transport = http_request(base_url, "/v1/completions", payload, directory, probe)
        if (
            transport["headers"].get("content-type", "").split(";", 1)[0].strip().lower()
            != "application/json"
        ):
            raise BaselineFailure("ABC 规划响应不是 application/json")
        response = json.loads(Path(transport["body_path"]).read_text())
        plan = planning.extract_abc(response, model)
        if (
            plan.get("naturally_ended") is not True
            or not isinstance(plan.get("abc"), str)
            or not plan["abc"].strip()
        ):
            raise BaselineFailure("ABC 规划未真实自然结束或为空；禁止继续full")
        (directory / "score.abc").write_text(plan["abc"])
        write_json(
            directory / "token-ids.json",
            {key: plan[key] for key in ("abc_token_ids", "generated_token_ids")},
        )
        write_json(
            directory / "result.json", {"status": "complete", **plan, "transport": transport}
        )
        return plan
    except BaseException as error:
        directory.mkdir(parents=True, exist_ok=True)
        write_json(
            directory / "result.json",
            {
                "status": "failed",
                "naturally_ended": False,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def run_baseline(mode, base_url, model, sample, output, probe, *, save_npy=False):
    """Sequential dependency chain: a failure is propagated, never replaced by a shorter request."""
    results = {}
    try:
        if mode in ("off", "all"):
            results["off"] = generate_song(
                base_url, sample, "off", output / "off", probe, save_npy=save_npy
            )
        if mode in ("full", "all"):
            results["planning"] = generate_plan(base_url, model, sample, output / "planning", probe)
            results["full"] = generate_song(
                base_url,
                sample,
                "full",
                output / "full",
                probe,
                abc=results["planning"]["abc"],
                save_npy=save_npy,
            )
        if mode not in ("off", "full", "all"):
            raise BaselineFailure(f"不支持的模式：{mode}")
    except BaseException as error:
        write_json(
            output / "failed.json",
            {
                "status": "failed",
                "mode": mode,
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_stages": list(results),
                "listening_review": "not_performed",
                "ended_at": utc_now(),
            },
        )
        raise
    write_json(
        output / "completed.json",
        {
            "status": "complete",
            "mode": mode,
            "stages": list(results),
            "listening_review": "not_performed",
            "lyrics_accuracy_review": "not_performed",
            "ended_at": utc_now(),
        },
    )
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("off", "full", "all"), default="all")
    parser.add_argument("--url", default="http://127.0.0.1:8096")
    parser.add_argument("--plist", type=Path, default=DEFAULT_PLIST)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New evidence directory; existing directories are not overwritten",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Also save decoded WAV samples, not original model float output",
    )
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        environment = service_environment(args.plist, args.url, output)
        model = Path(environment["model_path"])
        sample_path = owned_file(model / SAMPLE_RELATIVE)
        sample = json.loads(sample_path.read_text())
        if (
            sample.get("title") != "今晚不眠"
            or sample.get("seed") != SEED
            or not isinstance(sample.get("lyrics"), str)
            or not sample["lyrics"].strip()
            or not isinstance(sample.get("style"), str)
            or not sample["style"].strip()
        ):
            raise BaselineFailure(
                "需要原完整官方《今晚不眠》sample/seed12300；本工具不改写歌词或风格"
            )
        write_json(
            output / "input.json",
            {
                "sample_path": str(sample_path),
                "sample_fingerprint": fingerprint(sample_path),
                "sample": sample,
                "mode": args.mode,
                "seed": SEED,
                "max_audio_frames": MAX_AUDIO_FRAMES,
                "native_sampling": True,
                "reference_boundary": "Official companion MP3/examples.json describe an earlier checkpoint; not a current-3B same-weight golden output or target duration. This run must naturally end on its own.",
            },
        )

        def probe():
            return resource_snapshot(
                environment["service_pid"], environment["service_create_time"], output
            )

        enforce_resources(probe())
        models = http_request(args.url, "/v1/models", None, output / "models", probe)
        response = json.loads(Path(models["body_path"]).read_text())
        matches = [entry for entry in response.get("data", []) if entry.get("id") == MODEL_NAME]
        if len(matches) != 1 or matches[0].get("max_model_len") != 24576:
            raise BaselineFailure("HTTP 模型身份/context24576未确认")
        if Path(matches[0].get("root", "")).resolve() != model:
            raise BaselineFailure("HTTP 所选模型路径与实际服务checkpoint不一致")
        run_baseline(args.mode, args.url, model, sample, output, probe, save_npy=args.save_npy)
    except BaseException as error:
        if not (output / "failed.json").exists():
            write_json(
                output / "failed.json",
                {
                    "status": "failed",
                    "mode": args.mode,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "ended_at": utc_now(),
                    "listening_review": "not_performed",
                },
            )
        print(f"基线失败：{error}；证据：{output}", file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    print(f"完整自然结束基线已保存：{output}；客观检查不等于人工听感/歌词验收，未自动播放。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
