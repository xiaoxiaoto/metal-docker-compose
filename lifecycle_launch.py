#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""启动/停止已经准备好的用户 launchctl plist；不准备或下载模型。"""

import argparse
import errno
import os
import plistlib
import re
import shutil
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

LAUNCHCTL = "/bin/launchctl"
COMMAND_TIMEOUT = 15.0
STOP_TIMEOUT = 45.0


class LaunchError(Exception):
    """可安全展示的操作错误（不包含 plist 内容或子进程输出）。"""


def validate_plist(value):
    path = Path(os.path.abspath(Path(value).expanduser()))
    try:
        # O_NOFOLLOW protects the final component even if it changes after lstat.
        if path.is_symlink():
            raise LaunchError("plist 不得是符号链接")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise LaunchError("plist 必须是普通文件")
            if info.st_uid != os.getuid() or info.st_mode & 0o022:
                raise LaunchError("plist 必须由当前用户所有，且禁止 group/world 写入")
            if info.st_size > 1024 * 1024:
                raise LaunchError("plist 大于 1 MiB，拒绝加载")
            config = plistlib.load(source)
    except (OSError, ValueError, TypeError, plistlib.InvalidFileException) as exc:
        raise LaunchError("无法安全读取有效 plist") from exc
    if not isinstance(config, dict):
        raise LaunchError("plist 顶层必须是字典")
    label = config.get("Label")
    if (
        not isinstance(label, str)
        or not re.fullmatch(r"com\.local\.[A-Za-z0-9_.-]+", label)
        or path.name != label + ".plist"
    ):
        raise LaunchError("Label 必须为 com.local.*，且与 plist 文件名一致")
    arguments = config.get("ProgramArguments")
    if (
        not isinstance(arguments, list)
        or not arguments
        or any(not isinstance(item, str) or "\0" in item for item in arguments)
    ):
        raise LaunchError("ProgramArguments 必须是非空字符串 argv 数组")
    executable = Path(arguments[0])
    if (
        not executable.is_absolute()
        or not executable.is_file()
        or not os.access(executable, os.X_OK)
    ):
        raise LaunchError("ProgramArguments[0] 必须为存在且可执行的绝对路径")
    # Program overrides argv[0] in launchd; disallow a different executable.
    if "Program" in config and config["Program"] != arguments[0]:
        raise LaunchError("Program 与 ProgramArguments[0] 不一致，拒绝操作")
    if "UserName" in config or "GroupName" in config:
        raise LaunchError("用户 GUI 任务不得指定 UserName/GroupName")
    return path.resolve(strict=True), label


def launchctl(*arguments, timeout=COMMAND_TIMEOUT):
    try:
        return subprocess.run(
            [LAUNCHCTL, *arguments],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise LaunchError("launchctl 操作超时；未输出可能含敏感信息的任务内容") from exc
    except OSError as exc:
        raise LaunchError("无法执行 /bin/launchctl") from exc


def registration(service, timeout=COMMAND_TIMEOUT):
    result = launchctl("print", service, timeout=timeout)
    if result.returncode == 0:
        return result.stdout.decode("utf-8", errors="replace")
    # launchctl uses ESRCH (113 on Darwin) for an absent service.
    if result.returncode == 113 or b"Could not find service" in result.stderr:
        return None
    raise LaunchError("无法查询当前用户 GUI launchctl 任务；请确认登录会话可用")


def require_identity(output, path):
    # Only the root-level path field identifies the registered source plist;
    # do not match nested environment variables or arguments.
    paths = re.findall(r"^\tpath = (.+)$", output, flags=re.MULTILINE)
    if len(paths) != 1:
        raise LaunchError("无法确认已注册任务的 plist 来源，拒绝 bootout")
    try:
        actual = Path(paths[0]).resolve(strict=True)
    except OSError as exc:
        raise LaunchError("已注册任务的 plist 来源不可验证，拒绝 bootout") from exc
    if actual != path:
        raise LaunchError("已注册任务来自不同 plist，拒绝 bootout")


def assert_port_available(port):
    # Binding catches loopback and wildcard listeners without touching their PIDs.
    for family, address in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                if family == socket.AF_INET6:
                    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                sock.bind((address, port))
        except OSError as exc:
            if family == socket.AF_INET6 and exc.errno in (
                errno.EAFNOSUPPORT,
                errno.EADDRNOTAVAIL,
                errno.EPROTONOSUPPORT,
            ):
                continue
            raise LaunchError(f"本机端口 {port} 已占用或不可绑定；不会停止其他进程") from exc


def port_open(port):
    for family, address in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.2)
                result = sock.connect_ex((address, port))
                if result not in (
                    errno.ECONNREFUSED,
                    errno.EAFNOSUPPORT,
                    errno.EADDRNOTAVAIL,
                    errno.EPROTONOSUPPORT,
                ):
                    # Successful connects and indeterminate/time-out results
                    # both prevent claiming that a listener has disappeared.
                    return True
        except OSError as exc:
            if exc.errno not in (errno.EAFNOSUPPORT, errno.EADDRNOTAVAIL, errno.EPROTONOSUPPORT):
                raise LaunchError("无法确认本机端口是否关闭") from exc
    return False


def stop(service, path, port):
    deadline = time.monotonic() + STOP_TIMEOUT
    output = registration(service)
    if output is not None:
        require_identity(output, path)
        result = launchctl(
            "bootout", service, timeout=min(COMMAND_TIMEOUT, max(0.1, deadline - time.monotonic()))
        )
        if result.returncode != 0:
            raise LaunchError("launchctl bootout 失败；未停止其他任务，请检查用户任务状态")
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        output = registration(service, timeout=min(COMMAND_TIMEOUT, max(0.1, remaining)))
        if output is not None:
            require_identity(output, path)
        if output is None and not port_open(port):
            return
        time.sleep(min(0.2, max(0, deadline - time.monotonic())))
    raise LaunchError("停止等待超过 45 秒：任务仍注册或端口仍开放；未 kill 任何 PID")


def require_resources(path, minimum_gib):
    """Conservative free-memory estimate; never reclaim or alter system limits."""
    try:
        memory = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "hw.memsize"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=COMMAND_TIMEOUT,
            check=False,
        )
        pressure = subprocess.run(
            ["/usr/bin/memory_pressure", "-Q"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=COMMAND_TIMEOUT,
            check=False,
            env={**os.environ, "LC_ALL": "C", "LANG": "C"},
        )
        if memory.returncode != 0 or pressure.returncode != 0:
            raise LaunchError("无法读取系统可用内存，拒绝冷启动")
        total = int(memory.stdout.strip())
        match = re.search(
            rb"System-wide memory free percentage:\s*(\d+)%",
            pressure.stdout,
        )
        if total <= 0 or match is None or not 0 <= int(match[1]) <= 100:
            raise LaunchError("系统可用内存输出不可识别，拒绝冷启动")
        available = total * int(match[1]) // 100
        if available < minimum_gib * 1024**3:
            raise LaunchError(
                f"冷启动需要至少 {minimum_gib:g} GiB 保守估计可用内存；不会停止其他模型"
            )
        if shutil.disk_usage(path.parent).free < 2 * 1024**3:
            raise LaunchError("plist 所在磁盘可用空间不足 2 GiB，拒绝冷启动")
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        raise LaunchError("系统资源检查失败或超时，拒绝冷启动") from exc


def main():
    parser = argparse.ArgumentParser(
        description="仅管理已准备的当前用户 com.local.* plist，不下载或准备模型。",
        epilog="start 只确认 launchctl 注册；模型健康/身份由生命周期管理器等待。stop 最多等待 45 秒。",
    )
    parser.add_argument("action", choices=("start", "stop", "check"))
    parser.add_argument("plist_path", help="已准备的 plist 路径，支持 ~，文件名须为 <Label>.plist")
    parser.add_argument("port", type=int, help="模型本机 TCP 端口（1–65535）")
    parser.add_argument(
        "--min-free-gib", type=float, default=16, help="冷启动最低保守空闲内存，默认 16 GiB"
    )
    args = parser.parse_args()
    try:
        if sys.platform != "darwin":
            raise LaunchError("此适配器仅支持 macOS launchctl GUI 用户任务")
        if not 1 <= args.port <= 65535:
            raise LaunchError("端口必须在 1–65535 范围内")
        if not 0 < args.min_free_gib <= 1024:
            raise LaunchError("min-free-gib 必须是 0–1024 范围内的有限正数")
        path, label = validate_plist(args.plist_path)
        service = f"gui/{os.getuid()}/{label}"
        if args.action == "stop":
            stop(service, path, args.port)
        elif args.action == "start":
            if registration(service) is not None:
                raise LaunchError("任务已经注册；拒绝接管或重启已有任务")
            assert_port_available(args.port)
            require_resources(path, args.min_free_gib)
            try:
                result = launchctl("bootstrap", f"gui/{os.getuid()}", str(path))
            except LaunchError:
                # A timed-out bootstrap may already have registered this plist.
                # Never boot out a task whose registered source differs.
                output = registration(service)
                if output is not None:
                    require_identity(output, path)
                    stop(service, path, args.port)
                raise
            if result.returncode != 0:
                raise LaunchError("launchctl bootstrap 失败；未接管已有任务")
        print(f"{args.action} 完成：{label}")
        return 0
    except (LaunchError, OSError) as exc:
        # Never print subprocess output, argv, environment, or plist data.
        message = str(exc) if isinstance(exc, LaunchError) else "本机文件或系统操作失败"
        print(f"lifecycle_launch: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
