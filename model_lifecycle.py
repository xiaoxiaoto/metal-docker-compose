# SPDX-License-Identifier: Apache-2.0
"""Loopback-only HTTP proxy with explicit, per-service lifecycle ownership.

Dynamic states are cold -> starting -> ready -> stopping -> cold. Errors retain
ownership until a successful stop; only a successful start command grants it.
The per-service condition serializes starts/stops, and a global loading lock
serializes cold starts across services. A lease includes loading, forwarding and
upstream draining after a client disconnect. Its release starts the idle TTL.
Resident services never execute either command. Externally ready dynamic services
are usable but never owned or stopped. Configuration is immutable until restart.

This is a local trusted gateway, not an authentication boundary. HTTP upgrades,
CONNECT and request trailers are unsupported. Responses use close delimiting;
SSE and binary responses are streamed without application-level interpretation.
At a request's hard deadline the upstream connection is explicitly closed. A
request without confirmed upstream completion pins its service against unloading
until manager restart; backend cancellation cannot be inferred from socket close.
Commands must return nonzero without leaving a newly started service behind;
a failed command cannot safely confer ownership (it may have refused an external
service). No process discovery, arbitrary PID killing or hot reload is performed.
"""

import argparse
import http.client
import json
import logging
import math
import os
import signal
import socket
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

LOG = logging.getLogger("model_lifecycle")
MAX_BODY = 100 * 1024 * 1024
MAX_READY_BODY = 1024 * 1024
STOP_TIMEOUT = 60.0
START_COMMAND_TIMEOUT = 120.0
SELECTOR = "X-Model-Lifecycle-Service"
HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "proxy-connection",
}


class LifecycleError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class _Connection(http.client.HTTPConnection):
    """Retain the socket even when getresponse detaches a close-delimited reply."""

    transport = None
    deadline = math.inf

    def connect(self):
        super().connect()
        self.transport = self.sock
        if time.monotonic() >= self.deadline:
            self.close()
            raise TimeoutError("connection deadline exceeded")


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _upstream(value):
    if not isinstance(value, str):
        raise ValueError("upstream must be an HTTP loopback URL")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("invalid upstream URL") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username is not None
        or parsed.password is not None
        or port is None
        or not 1 <= port <= 65535
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or any(ord(char) <= 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError("upstream must be http://127.0.0.1:<port> without a path")
    return f"http://127.0.0.1:{port}"


def _path(value):
    if not isinstance(value, str) or not value.startswith("/") or value.startswith("//"):
        raise ValueError("route paths must be absolute origin-form paths")
    parsed = urlsplit(value)
    if (
        parsed.query
        or parsed.fragment
        or parsed.path != value
        or any(ord(char) <= 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError("route paths cannot contain queries, fragments or controls")
    return value


def _services(config):
    if not isinstance(config, dict) or not config:
        raise ValueError("services must be a nonempty object")
    result = {}
    allowed = {
        "mode",
        "idle_timeout",
        "startup_timeout",
        "request_timeout",
        "upstream",
        "start_command",
        "stop_command",
        "ready_path",
        "ready_model",
        "path_upstreams",
    }
    for name, supplied in config.items():
        if (
            not isinstance(name, str)
            or not name
            or any(ord(c) <= 32 or ord(c) == 127 for c in name)
        ):
            raise ValueError("service keys must be nonempty strings without whitespace")
        if not isinstance(supplied, dict) or set(supplied) - allowed:
            raise ValueError(f"invalid configuration keys for service {name}")
        item = dict(supplied)
        if item.get("mode") not in ("resident", "dynamic"):
            raise ValueError(f"service {name}: mode must be resident or dynamic")
        item["upstream"] = _upstream(item.get("upstream"))
        for key, default in (
            ("idle_timeout", 300),
            ("startup_timeout", 600),
            ("request_timeout", 1800),
        ):
            item[key] = _positive(item.get(key, default), f"{name}.{key}")
        for key in ("start_command", "stop_command"):
            command = item.get(key, [])
            if (
                not isinstance(command, list)
                or any(not isinstance(arg, str) or not arg or "\x00" in arg for arg in command)
                or (item["mode"] == "dynamic" and not command)
            ):
                raise ValueError(f"service {name}: {key} must be a nonempty string argv array")
            item[key] = list(command)
        model = item.get("ready_model")
        if model is not None and (not isinstance(model, str) or not model):
            raise ValueError(f"service {name}: ready_model must be a nonempty string")
        ready_path = _path(item.get("ready_path", "/health"))
        if model is not None:
            if "ready_path" in item and ready_path != "/v1/models":
                raise ValueError(f"service {name}: ready_model requires ready_path=/v1/models")
            ready_path = "/v1/models"
        item["ready_path"] = ready_path
        paths = item.get("path_upstreams", {})
        if not isinstance(paths, dict):
            raise ValueError(f"service {name}: path_upstreams must be an object")
        item["path_upstreams"] = {_path(path): _upstream(url) for path, url in paths.items()}
        result[name] = item
    return result


def _routes(routes, services):
    if not isinstance(routes, dict):
        raise ValueError("routes must be an object")
    result = {}
    for path, name in routes.items():
        if not isinstance(name, str) or name not in services:
            raise ValueError("route targets must name a configured service")
        result[_path(path)] = name
    return result


def load_config(path):
    """Read/validate JSON and return {'services': ..., 'routes': ...}."""

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    with open(path, encoding="utf-8") as stream:
        config = json.load(stream, object_pairs_hook=unique_object)
    if not isinstance(config, dict) or set(config) - {"services", "routes"}:
        raise ValueError("configuration must contain services and optional routes only")
    services = _services(config.get("services"))
    return {"services": services, "routes": _routes(config.get("routes", {}), services)}


class _Service:
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.condition = threading.Condition()
        self.state = "resident" if config["mode"] == "resident" else "cold"
        self.owned = False
        self.inflight = 0
        self.idle_since = None
        self.last_error = None
        self.generation = 0
        self.failed_generation = None
        self.pinned = False


class Manager:
    """Own immutable service configuration and background idle reclamation.

    Construct with Manager(load_config(path)['services'], Path(path).parent,
    routes=load_config(path)['routes']). close() prevents new leases and only
    reclaims owned services with no in-flight request. Stop accepting and join
    the HTTP server's workers before close() to drain active requests on exit.
    """

    def __init__(self, services, cwd=None, routes=None):
        configs = _services(services)
        self.services = {name: _Service(name, config) for name, config in configs.items()}
        self.routes = _routes(routes or {}, configs)
        self.cwd = os.fspath(Path(cwd or os.getcwd()).resolve())
        self._loading = threading.Lock()
        self._closing = threading.Event()
        self._starts = []
        self._starts_lock = threading.Lock()
        self._reaper = threading.Thread(target=self._reap, name="lifecycle-idle", daemon=True)
        self._reaper.start()

    def select(self, path, values):
        if len(values) > 1:
            raise LifecycleError(400, "duplicate lifecycle service selector")
        selected = values[0] if values else None
        route = self.routes.get(path)
        if selected is not None and route is not None and selected != route:
            raise LifecycleError(400, "lifecycle selector conflicts with configured route")
        name = selected if selected is not None else route
        if not name:
            raise LifecycleError(400, "missing lifecycle service selector")
        if name not in self.services:
            raise LifecycleError(404, "unknown lifecycle service")
        return self.services[name]

    def status(self):
        now = time.monotonic()
        services = {}
        for name, service in self.services.items():
            with service.condition:
                config = service.config
                idle = None if service.idle_since is None else max(0.0, now - service.idle_since)
                services[name] = {
                    "mode": config["mode"],
                    "state": service.state,
                    "inflight": service.inflight,
                    "owned": service.owned,
                    "pinned": service.pinned,
                    "idle_seconds": idle,
                    "idle_timeout": config["idle_timeout"],
                    "startup_timeout": config["startup_timeout"],
                    "request_timeout": config["request_timeout"],
                    "last_error": service.last_error,
                }
        return {
            "object": "model_lifecycle",
            "status": "stopping" if self._closing.is_set() else "ok",
            "services": services,
        }

    def _ready(self, service, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        parsed = urlsplit(service.config["upstream"])
        connection = _Connection(parsed.hostname, parsed.port, timeout=min(2.0, remaining))
        connection.deadline = min(deadline, time.monotonic() + 2.0)
        timer = threading.Timer(min(2.0, remaining), self._abort_connection, args=(connection,))
        timer.daemon = True
        timer.start()
        response = None
        try:
            connection.request("GET", service.config["ready_path"], headers={"Connection": "close"})
            response = connection.getresponse()
            if response.status != 200:
                return False
            model = service.config.get("ready_model")
            if model is None:
                return True
            body = response.read(MAX_READY_BODY + 1)
            if len(body) > MAX_READY_BODY:
                return False
            data = json.loads(body)
            return (
                isinstance(data, dict)
                and isinstance(data.get("data"), list)
                and any(isinstance(item, dict) and item.get("id") == model for item in data["data"])
            )
        except (OSError, ValueError, http.client.HTTPException):
            return False
        finally:
            timer.cancel()
            if response is not None:
                response.close()
            connection.close()

    @staticmethod
    def _abort_connection(connection):
        sock = getattr(connection, "transport", None) or connection.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    def _command(self, service, key, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise LifecycleError(504, f"{key} deadline exceeded")
        # Inherit deployment logs; do not capture unbounded command output or log argv.
        try:
            completed = subprocess.run(
                service.config[key],
                cwd=self.cwd,
                stdin=subprocess.DEVNULL,
                timeout=remaining,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise LifecycleError(504, f"{key} timed out") from exc
        except OSError as exc:
            raise LifecycleError(503, f"{key} could not execute") from exc
        if completed.returncode:
            raise LifecycleError(503, f"{key} failed (exit {completed.returncode})")

    def _stop(self, service):
        deadline = time.monotonic() + STOP_TIMEOUT
        try:
            self._command(service, "stop_command", deadline)
            while self._ready(service, deadline):
                if time.monotonic() >= deadline:
                    raise LifecycleError(504, "service remains ready after stop_command")
                time.sleep(min(0.2, max(0, deadline - time.monotonic())))
            # A timed-out readiness probe is not evidence that a service stopped.
            if time.monotonic() >= deadline:
                raise LifecycleError(504, "stop readiness deadline exceeded")
        except LifecycleError as exc:
            with service.condition:
                service.state = "error"
                service.last_error = str(exc)
                service.idle_since = time.monotonic() if service.inflight == 0 else None
                service.condition.notify_all()
            LOG.error("service=%s stop failed: %s; ownership retained", service.name, exc)
            return False
        with service.condition:
            service.owned = False
            service.state = "cold"
            service.idle_since = None
            service.last_error = None
            service.condition.notify_all()
        LOG.info("service=%s stopped", service.name)
        return True

    def _start(self, service, deadline):
        # A ready service must not queue behind another model's cold start.
        if self._ready(service, deadline):
            with service.condition:
                service.state = "ready"
                service.last_error = None
            return
        with service.condition:
            if service.owned:
                raise LifecycleError(503, "owned service is unhealthy; refusing a second start")
        locked = self._loading.acquire(timeout=max(0.0, deadline - time.monotonic()))
        if not locked:
            raise LifecycleError(504, "global cold-start queue deadline exceeded")
        started = False
        try:
            # An external owner may have started the model while we were queued.
            if self._ready(service, deadline):
                with service.condition:
                    service.state = "ready"
                    service.last_error = None
                LOG.info("service=%s externally ready; not owned", service.name)
                return
            if time.monotonic() >= deadline:
                raise LifecycleError(504, "cold-start deadline exceeded before dispatch")
            # Request cancellation must not kill a launcher between bootstrap and
            # its acknowledgement. Finish its independently bounded transaction.
            self._command(service, "start_command", time.monotonic() + START_COMMAND_TIMEOUT)
            with service.condition:
                service.owned = True
            started = True
            LOG.info("service=%s start command succeeded; ownership acquired", service.name)
            while time.monotonic() < deadline and not self._closing.is_set():
                if self._ready(service, deadline):
                    with service.condition:
                        service.state = "ready"
                        service.last_error = None
                    return
                time.sleep(min(0.2, max(0, deadline - time.monotonic())))
            raise LifecycleError(504, "service readiness deadline exceeded")
        except LifecycleError:
            if started:
                with service.condition:
                    service.state = "stopping"
                self._stop(service)
            raise
        finally:
            self._loading.release()

    def _run_start(self, service, deadline):
        try:
            self._start(service, deadline)
        except LifecycleError as exc:
            with service.condition:
                service.state = "error"
                service.last_error = str(exc)
                service.failed_generation = service.generation
            LOG.error("service=%s startup failed: %s", service.name, exc)
        finally:
            with service.condition:
                service.condition.notify_all()

    def discover(self, service, deadline):
        """Probe readiness without ownership changes, a lease, or TTL refresh."""
        if self._closing.is_set():
            raise LifecycleError(503, "lifecycle manager is stopping")
        with service.condition:
            if service.state in ("starting", "stopping"):
                raise LifecycleError(503, "service is not ready; discovery never starts models")
        if not self._ready(service, deadline):
            raise LifecycleError(503, "service is cold; discovery never starts models")

    def acquire(self, service, deadline):
        """Acquire a lease; caller MUST release on success, including proxy errors."""
        with service.condition:
            if self._closing.is_set():
                raise LifecycleError(503, "lifecycle manager is stopping")
            service.inflight += 1
            service.idle_since = None
            generation = service.generation - (service.state == "starting")
        load_deadline = min(deadline, time.monotonic() + service.config["startup_timeout"])
        try:
            while True:
                with service.condition:
                    if time.monotonic() >= load_deadline:
                        raise LifecycleError(504, "request deadline exceeded while loading")
                    if service.config["mode"] == "resident":
                        return
                    if (
                        service.failed_generation is not None
                        and service.failed_generation > generation
                    ):
                        raise LifecycleError(503, service.last_error or "service startup failed")
                    if service.state == "ready" and (
                        service.owned or service.generation > generation
                    ):
                        return
                    if service.state not in ("starting", "stopping"):
                        service.state = "starting"
                        service.generation += 1
                        worker = threading.Thread(
                            target=self._run_start,
                            args=(service, load_deadline),
                            name="lifecycle-start",
                        )
                        with self._starts_lock:
                            self._starts = [
                                pending for pending in self._starts if pending.is_alive()
                            ]
                            self._starts.append(worker)
                            worker.start()
                    service.condition.wait(timeout=min(0.5, load_deadline - time.monotonic()))
        except BaseException:
            self.release(service)
            raise

    def pin(self, service):
        with service.condition:
            service.pinned = True
            service.last_error = (
                "upstream completion unconfirmed; automatic unloading disabled until restart"
            )
        LOG.error("service=%s completion unconfirmed; pinned against unloading", service.name)

    def release(self, service):
        with service.condition:
            service.inflight -= 1
            if service.inflight == 0:
                service.idle_since = time.monotonic()
            service.condition.notify_all()

    def _reap(self):
        while not self._closing.wait(0.25):
            for service in self.services.values():
                with service.condition:
                    if (
                        service.config["mode"] != "dynamic"
                        or not service.owned
                        or service.pinned
                        or service.inflight
                        or service.state in ("starting", "stopping")
                        or service.idle_since is None
                        or time.monotonic() - service.idle_since < service.config["idle_timeout"]
                    ):
                        continue
                    service.state = "stopping"
                self._stop(service)

    def unload(self, service):
        """Explicit single-service stop, preserving the ownership state machine."""
        with service.condition:
            if service.inflight or service.state in ("starting", "stopping") or service.pinned:
                raise LifecycleError(409, "service is active or completion is unconfirmed")
            if not service.owned:
                return False
            service.state = "stopping"
        if not self._stop(service):
            raise LifecycleError(503, "service stop failed; ownership retained")
        return True

    def quiesce(self):
        """Atomically reject new leases only when every service is idle."""
        locked = []
        try:
            for service in self.services.values():
                service.condition.acquire()
                locked.append(service)
            if any(service.inflight for service in locked):
                raise LifecycleError(409, "requests are active; manager remains available")
            self.begin_shutdown()
        finally:
            for service in reversed(locked):
                service.condition.release()

    def begin_shutdown(self):
        self._closing.set()

    def close(self):
        self.begin_shutdown()
        self._reaper.join()
        with self._starts_lock:
            starts = tuple(self._starts)
        for worker in starts:
            worker.join()
        for service in self.services.values():
            with service.condition:
                if (
                    not service.owned
                    or service.pinned
                    or service.inflight
                    or service.state in ("starting", "stopping")
                ):
                    continue
                service.state = "stopping"
            self._stop(service)


class ProxyServer(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True
    allow_reuse_address = True

    def __init__(self, address, manager):
        if address[0] != "127.0.0.1":
            raise ValueError("lifecycle proxy must bind to 127.0.0.1")
        self.manager = manager
        super().__init__(address, _Handler)

    def get_request(self):
        connection, address = super().get_request()
        # Bound incomplete HTTP headers before a service timeout can be selected.
        connection.settimeout(15.0)
        return connection, address


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "ModelLifecycle"
    sys_version = ""

    def setup(self):
        super().setup()

        def expire_headers():
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        self._header_timer = threading.Timer(15.0, expire_headers)
        self._header_timer.daemon = True
        self._header_timer.start()

    def finish(self):
        self._header_timer.cancel()
        super().finish()

    def log_message(self, format, *args):
        # Do not log request paths/query strings, which can contain credentials.
        LOG.info(
            "HTTP peer=%s status=%s", self.client_address[0], args[1] if len(args) > 1 else "-"
        )

    def _json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
            self.wfile.flush()

    def send_error(self, code, message=None, explain=None):
        try:
            self._json(
                code,
                {
                    "error": {
                        "message": message or "invalid HTTP request",
                        "type": "lifecycle_error",
                    }
                },
            )
        except OSError:
            pass

    def _request_body(self, deadline):
        lengths = self.headers.get_all("Content-Length", [])
        encodings = self.headers.get_all("Transfer-Encoding", [])
        if len(lengths) > 1 or len(encodings) > 1 or (lengths and encodings):
            raise LifecycleError(400, "ambiguous request body framing")
        if encodings and encodings[0].strip().lower() != "chunked":
            raise LifecycleError(400, "unsupported transfer encoding")
        if self.headers.get("Trailer"):
            raise LifecycleError(400, "request trailers are unsupported")
        expect = self.headers.get("Expect")
        if expect and expect.lower() != "100-continue":
            raise LifecycleError(417, "unsupported Expect header")
        if expect:
            self.wfile.write(b"HTTP/1.1 100 Continue\r\n\r\n")
            self.wfile.flush()
        length = 0
        if lengths:
            value = lengths[0]
            if not value.isascii() or not value.isdigit():
                raise LifecycleError(400, "invalid Content-Length")
            try:
                length = int(value)
            except ValueError as exc:
                raise LifecycleError(400, "invalid Content-Length") from exc
            if length > MAX_BODY:
                raise LifecycleError(413, "request body exceeds 100 MiB")
        stream = tempfile.SpooledTemporaryFile(max_size=1024 * 1024)
        try:
            total = 0
            if encodings:
                while True:
                    self.connection.settimeout(max(0.001, deadline - time.monotonic()))
                    line = self.rfile.readline(8194)
                    if len(line) > 8192 or not line.endswith(b"\r\n"):
                        raise LifecycleError(400, "invalid chunk framing")
                    size_text = line[:-2].split(b";", 1)[0]
                    if not size_text or any(c not in b"0123456789abcdefABCDEF" for c in size_text):
                        raise LifecycleError(400, "invalid chunk size")
                    size = int(size_text, 16)
                    if size == 0:
                        if self.rfile.read(2) != b"\r\n":
                            raise LifecycleError(400, "request trailers are unsupported")
                        break
                    if total + size > MAX_BODY:
                        raise LifecycleError(413, "request body exceeds 100 MiB")
                    self._copy_body(stream, size, deadline)
                    if self.rfile.read(2) != b"\r\n":
                        raise LifecycleError(400, "invalid chunk terminator")
                    total += size
            else:
                self._copy_body(stream, length, deadline)
                total = length
            stream.seek(0)
            return stream, total
        except BaseException:
            stream.close()
            raise

    def _copy_body(self, stream, count, deadline):
        while count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LifecycleError(504, "request body deadline exceeded")
            self.connection.settimeout(remaining)
            data = self.rfile.read(min(65536, count))
            if not data:
                raise LifecycleError(400, "incomplete request body")
            stream.write(data)
            count -= len(data)

    @staticmethod
    def _excluded(headers):
        excluded = set(HOP_HEADERS)
        for value in headers.get_all("Connection", []):
            excluded.update(token.strip().lower() for token in value.split(","))
        return excluded

    def _forward(self, service, path, body, length, deadline, context):
        if time.monotonic() >= deadline or context["expired"]:
            raise LifecycleError(504, "request deadline exceeded before forwarding")
        upstream = service.config["path_upstreams"].get(path, service.config["upstream"])
        parsed = urlsplit(upstream)
        connection = _Connection(
            parsed.hostname, parsed.port, timeout=max(0.001, deadline - time.monotonic())
        )
        connection.deadline = deadline
        context["upstream"] = connection
        response = None
        try:
            excluded = self._excluded(self.headers) | {"host", "content-length", "expect"}
            connection.putrequest(
                self.command, self.path, skip_host=True, skip_accept_encoding=True
            )
            connection.putheader("Host", f"127.0.0.1:{parsed.port}")
            for name, value in self.headers.items():
                if name.lower() not in excluded and not name.lower().startswith(
                    "x-model-lifecycle-"
                ):
                    connection.putheader(name, value)
            connection.putheader("Content-Length", str(length))
            connection.putheader("Connection", "close")
            context["dispatched"] = True
            connection.endheaders()
            while True:
                data = body.read(65536)
                if not data:
                    break
                connection.send(data)
            response = connection.getresponse()
            if response.status == 101:
                raise LifecycleError(502, "upstream protocol upgrades are unsupported")
            excluded = self._excluded(response.headers)
            self.send_response_only(response.status, response.reason)
            for name, value in response.getheaders():
                if name.lower() not in excluded and not name.lower().startswith(
                    "x-model-lifecycle-"
                ):
                    self.send_header(name, value)
            self.send_header("Connection", "close")
            context["headers_sent"] = True
            disconnected = False
            try:
                context["client_io"] = True
                self.end_headers()
                self.wfile.flush()
            except OSError:
                disconnected = True
            finally:
                context["client_io"] = False
            while self.command != "HEAD":
                data = response.read1(65536)
                if not data:
                    if response.length not in (None, 0):
                        raise http.client.IncompleteRead(b"", response.length)
                    break
                if not disconnected:
                    try:
                        context["client_io"] = True
                        self.wfile.write(data)
                        self.wfile.flush()
                    except OSError:
                        # Keep reading until EOF/deadline: downstream loss is not
                        # permission to unload a backend still serving a request.
                        disconnected = True
                    finally:
                        context["client_io"] = False
            if context["expired"]:
                raise LifecycleError(504, "upstream response deadline exceeded")
            context["completed"] = True
        finally:
            context["upstream"] = None
            if response is not None:
                response.close()
            connection.close()

    def _handle(self):
        self._header_timer.cancel()
        self.close_connection = True
        context = {
            "upstream": None,
            "headers_sent": False,
            "dispatched": False,
            "completed": False,
            "expired": False,
            "client_io": False,
        }
        timer = None
        body = None
        leased = False
        service = None
        started = time.monotonic()
        try:
            if (
                not self.path.startswith("/")
                or self.path.startswith("//")
                or "#" in self.path
                or any(ord(c) <= 32 or ord(c) == 127 for c in self.path)
            ):
                raise LifecycleError(400, "only origin-form request targets are accepted")
            path = urlsplit(self.path).path
            if path == "/_lifecycle/unload":
                if self.command != "POST":
                    raise LifecycleError(405, "unload requires POST")
                selected = self.server.manager.select(path, self.headers.get_all(SELECTOR, []))
                handled = self.server.manager.unload(selected)
                self._json(200, {"object": "model_lifecycle", "handled": handled})
                return
            if path == "/_lifecycle/drain":
                if self.command != "POST":
                    raise LifecycleError(405, "drain requires POST")
                self.server.manager.quiesce()
                self._json(200, self.server.manager.status())
                return
            if path in ("/health", "/_lifecycle/status"):
                if self.command not in ("GET", "HEAD"):
                    raise LifecycleError(405, "management endpoints require GET or HEAD")
                payload = (
                    {"status": "ok", "scope": "model_lifecycle"}
                    if path == "/health"
                    else self.server.manager.status()
                )
                self._json(200, payload)
                return
            if self.command in ("CONNECT", "TRACE") or self.headers.get("Upgrade"):
                raise LifecycleError(405, "HTTP tunnels and upgrades are unsupported")
            if self.headers.get_all("X-Model-Lifecycle-Upstream"):
                raise LifecycleError(400, "upstream selection headers are forbidden")
            service = self.server.manager.select(path, self.headers.get_all(SELECTOR, []))
            budget = service.config["request_timeout"]
            if service.config["mode"] == "dynamic":
                budget += service.config["startup_timeout"]
            deadline = started + budget

            def expire():
                context["expired"] = True
                upstream = context["upstream"]
                if upstream is not None:
                    Manager._abort_connection(upstream)
                if context["client_io"]:
                    try:
                        self.connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass

            timer = threading.Timer(max(0.001, deadline - time.monotonic()), expire)
            timer.daemon = True
            timer.start()
            self.connection.settimeout(max(0.001, deadline - time.monotonic()))
            discovery = path == "/v1/models" and self.command in ("GET", "HEAD")
            if discovery:
                self.server.manager.discover(service, deadline)
            else:
                self.server.manager.acquire(service, deadline)
                leased = True
                # Loading has its own limit; forwarding keeps the full request budget.
                timer.cancel()
                deadline = time.monotonic() + service.config["request_timeout"]
                timer = threading.Timer(service.config["request_timeout"], expire)
                timer.daemon = True
                timer.start()
            context["client_io"] = True
            try:
                body, length = self._request_body(deadline)
            finally:
                context["client_io"] = False
            self._forward(service, path, body, length, deadline, context)
        except LifecycleError as exc:
            if not context["headers_sent"]:
                self.send_error(exc.status, str(exc))
        except (OSError, http.client.HTTPException, ValueError) as exc:
            if not context["headers_sent"]:
                expired = context["expired"] or isinstance(exc, TimeoutError)
                self.send_error(
                    504 if expired else 502,
                    "request deadline exceeded"
                    if expired
                    else "upstream or client connection failed",
                )
        finally:
            if timer is not None:
                timer.cancel()
            if body is not None:
                body.close()
            if leased:
                if context["dispatched"] and not context["completed"]:
                    self.server.manager.pin(service)
                self.server.manager.release(service)

    do_GET = _handle
    do_HEAD = _handle
    do_POST = _handle
    do_PUT = _handle
    do_PATCH = _handle
    do_DELETE = _handle
    do_OPTIONS = _handle
    do_CONNECT = _handle
    do_TRACE = _handle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--port", type=int, default=18100)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_config(args.config)
    except (OSError, ValueError) as exc:
        parser.error(f"invalid lifecycle configuration: {exc}")
    manager = Manager(config["services"], args.config.resolve().parent, config["routes"])
    try:
        server = ProxyServer(("127.0.0.1", args.port), manager)
    except BaseException:
        manager.close()
        raise
    stopping = threading.Event()

    def stop(signum, frame):
        if stopping.is_set():
            return
        stopping.set()
        manager.begin_shutdown()
        # shutdown() must not run on the serve_forever thread.
        threading.Thread(target=server.shutdown, name="lifecycle-shutdown", daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    LOG.info("listening on 127.0.0.1:%s; configuration requires restart to change", args.port)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        manager.begin_shutdown()
        server.server_close()  # Non-daemon request threads drain before cleanup.
        manager.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
