# SPDX-License-Identifier: Apache-2.0
"""Install the generic lifecycle proxy as a local launchctl service; never reload implicitly."""

import argparse
import hashlib
import http.client
import json
import os
import plistlib
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

LABEL = "com.local.model-lifecycle"
PORT = 18100
SOURCE = Path(__file__).resolve().parent
RUNTIME = Path.home() / "Library/Application Support/model-lifecycle"
DOMAIN = f"gui/{os.getuid()}"
TARGETS = {
    "h2o": "h2o-lightning-4b",
    "higgs": "higgs-audio-v3-tts-4b",
    "yue2": "yue2-3b",
    "moss": "moss-transcribe-diarize",
    "paddleocr": "paddleocr-vl",
    "qwenimage": "qwen-image-2.1-turbo",
    "seedvr2": "seedvr2",
}
SERVICE = f"{DOMAIN}/{LABEL}"


def run(*args, check=True):
    return subprocess.run(
        args, check=check, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60
    )


def status():
    connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=3)
    try:
        connection.request("GET", "/_lifecycle/status")
        response = connection.getresponse()
        value = json.loads(response.read(1024 * 1024))
        if response.status != 200 or value.get("object") != "model_lifecycle":
            raise ValueError("18100 端口不是预期生命周期管理器")
        return value
    finally:
        connection.close()


def registered():
    return run("launchctl", "print", SERVICE, check=False).returncode == 0


def fingerprint():
    digest = hashlib.sha256()
    for name in (
        "model_lifecycle.py",
        "lifecycle_launch.py",
        "lifecycle_control.py",
        "model-lifecycle.json",
    ):
        digest.update(name.encode())
        digest.update((SOURCE / name).read_bytes())
    digest.update(sys.executable.encode())
    return digest.hexdigest()


def configuration():
    # -I entry points do not import from cwd; only this explicit trusted source directory.
    sys.path.insert(0, str(SOURCE))
    from model_lifecycle import load_config

    return load_config(SOURCE / "model-lifecycle.json")


def mode(model):
    services = configuration()["services"]
    if model == "all":
        return (
            "resident"
            if any(item["mode"] == "resident" for item in services.values())
            else "dynamic"
        )
    model = TARGETS.get(model, model)
    if model not in services:
        raise ValueError(f"未配置模型：{model}")
    return services[model]["mode"]


def check_dynamic(model):
    model = TARGETS.get(model, model)
    item = configuration()["services"][model]
    if item["mode"] != "dynamic":
        raise ValueError(f"{model} 不是 dynamic 模式")
    # This is a deployment-specific preflight, not part of the generic proxy contract.
    command = item["start_command"]
    if len(command) < 5 or command[1:3] != ["lifecycle_launch.py", "start"]:
        raise ValueError("此部署的 dynamic 启动命令必须使用 lifecycle_launch.py")
    path = Path(command[3]).expanduser()
    if not path.is_file():
        raise ValueError(f"{model} 尚未准备：先按模型 README 完成一次显式部署；缺少 {path}")
    run(sys.executable, "-I", str(SOURCE / "lifecycle_launch.py"), "check", str(path), command[4])


def start():
    services = configuration()["services"]
    identity = fingerprint()
    marker = RUNTIME / "fingerprint"
    if registered():
        if not marker.is_file() or marker.read_text().strip() != identity:
            raise ValueError(
                "生命周期配置或代码已变化；先确认无在途请求，再 ./stop.sh lifecycle，"
                "然后 ./start.sh gateway；不会隐式重启"
            )
        status()
        print("生命周期管理器已运行；保留原模式与在途请求。")
        return
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", PORT)) == 0:
            raise ValueError("18100 端口已占用；不停止其他进程")
    RUNTIME.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(RUNTIME, 0o700)
    for name in ("model_lifecycle.py", "lifecycle_launch.py", "model-lifecycle.json"):
        shutil.copyfile(SOURCE / name, RUNTIME / name)
        os.chmod(RUNTIME / name, 0o600)
    logs = Path.home() / "Library/Logs"
    logs.mkdir(parents=True, exist_ok=True)
    config = {
        "Label": LABEL,
        "ProgramArguments": [
            sys.executable,
            "-I",
            str(RUNTIME / "model_lifecycle.py"),
            "--config",
            str(RUNTIME / "model-lifecycle.json"),
            "--port",
            str(PORT),
        ],
        "WorkingDirectory": str(RUNTIME),
        "RunAtLoad": True,
        "KeepAlive": False,
        "ExitTimeOut": len(services) * 60 + 150,
        "AbandonProcessGroup": False,
        "EnvironmentVariables": {
            "PATH": f"{Path(sys.executable).parent}:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "PYTHONUNBUFFERED": "1",
        },
        "StandardOutPath": str(logs / "model-lifecycle.log"),
        "StandardErrorPath": str(logs / "model-lifecycle.error.log"),
    }
    plist = RUNTIME / f"{LABEL}.plist"
    with tempfile.NamedTemporaryFile(dir=RUNTIME, delete=False) as file:
        temporary = Path(file.name)
        plistlib.dump(config, file)
    try:
        os.replace(temporary, plist)
    finally:
        temporary.unlink(missing_ok=True)
    run("launchctl", "bootstrap", DOMAIN, str(plist))
    try:
        deadline = time.monotonic() + 20
        while True:
            try:
                status()
                break
            except (OSError, ValueError, http.client.HTTPException):
                if time.monotonic() >= deadline:
                    raise ValueError(
                        "生命周期管理器未就绪；检查 ~/Library/Logs/model-lifecycle.error.log"
                    ) from None
                time.sleep(0.25)
        marker.write_text(identity + "\n")
    except BaseException:
        run("launchctl", "bootout", SERVICE, check=False)
        raise
    print(f"生命周期管理器已启动：127.0.0.1:{PORT}；未加载任何模型。")


def stop():
    if not registered():
        print("生命周期管理器未运行。")
        return
    value = status()
    connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=3)
    try:
        # Atomic admission barrier: a status-then-signal check can miss a new request.
        connection.request("POST", "/_lifecycle/drain", body=b"")
        response = connection.getresponse()
        response.read(1024 * 1024)
        if response.status != 200:
            raise ValueError("仍有在途/加载中请求，拒绝停止；等待请求结束后重试")
    finally:
        connection.close()
    shutdown_timeout = len(value["services"]) * 60 + 150
    subprocess.run(
        ["launchctl", "bootout", SERVICE],
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=shutdown_timeout,
    )
    deadline = time.monotonic() + shutdown_timeout
    while registered():
        if time.monotonic() >= deadline:
            raise ValueError("生命周期管理器尚未退出，请检查日志")
        time.sleep(0.25)
    print("生命周期管理器已停止；未接管的模型不受影响。")


def unload(model):
    model = TARGETS.get(model, model)
    if mode(model) != "dynamic" or not registered():
        return False
    connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=65)
    try:
        connection.request(
            "POST", "/_lifecycle/unload", body=b"", headers={"X-Model-Lifecycle-Service": model}
        )
        response = connection.getresponse()
        value = json.loads(response.read(1024 * 1024))
        if response.status != 200 or value.get("object") != "model_lifecycle":
            raise ValueError("动态模型仍有在途请求或停止失败；不绕过管理器直接停止")
        return value["handled"]
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("start", "stop", "status", "validate", "mode", "check-dynamic", "unload")
    )
    parser.add_argument("model", nargs="?")
    args = parser.parse_args()
    try:
        if args.action == "validate":
            configuration()
            print("生命周期配置有效。")
        elif args.action == "mode":
            if not args.model:
                raise ValueError("mode 需要模型 ID")
            print(mode(args.model))
        elif args.action == "check-dynamic":
            if not args.model:
                raise ValueError("check-dynamic 需要模型 ID")
            check_dynamic(args.model)
        elif args.action == "unload":
            if not args.model:
                raise ValueError("unload 需要模型 ID")
            if not unload(args.model):
                return 3  # Unmanaged service: preserve the original explicit stop script.
        elif args.action == "status":
            print(json.dumps(status(), ensure_ascii=False, indent=2))
        else:
            if sys.platform != "darwin":
                raise ValueError("此部署控制器需要 macOS launchctl；通用代理可直接运行")
            {"start": start, "stop": stop}[args.action]()
    except (OSError, ValueError, subprocess.SubprocessError, http.client.HTTPException) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
