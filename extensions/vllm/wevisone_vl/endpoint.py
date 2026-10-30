"""Native vLLM HTTP surface for the fixed WeVisOne-VL tasks.

``POST /v1/chat/completions`` with a ``wevisone_task`` field is served here:
the middleware buffers the body, fetches and canonicalizes one image, runs
the native chat handler, and attaches a ``pixel`` object to the completion.
Requests without ``wevisone_task`` are replayed unchanged, so ordinary chat
(including SSE) keeps the upstream path. The older flat
``/v1/wevisone/predict`` route remains as a deprecated wrapper.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import re
from argparse import Namespace
from io import BytesIO
from typing import Literal
from urllib.parse import urlsplit

import aiohttp
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.datastructures import State

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.multimodal.image import convert_image_mode, normalize_image, open_image

from .render import overlay_png
from .tasks import (
    DEFAULT_TOKENS,
    MAX_OBJECTS,
    MAX_QUERY,
    MAX_TOKENS,
    SYSTEM_PROMPT,
    TASKS,
    MarkError,
    build_prompt,
    expected_artifacts,
    parse_marks,
    segment_labels,
    to_predictions,
)

logger = logging.getLogger(__name__)
_MODEL = "wevisone-vl-4b"
_PATH = "/v1/wevisone/predict"
_CHAT_PATH = "/v1/chat/completions"
_MAX_IMAGE_BYTES = 25 * 1024 * 1024
_MAX_PIXELS = 4_194_304
_MAX_BASE64 = 4 * ((_MAX_IMAGE_BYTES + 2) // 3)
_MAX_JSON_BYTES = _MAX_IMAGE_BYTES
_DEFAULT_SAMPLING = {"temperature": 0, "top_p": 0.95, "top_k": 50, "repetition_penalty": 1.05}
_CONTROLS = {
    "model",
    "messages",
    "wevisone_task",
    "stream",
    "stream_options",
    "max_tokens",
    "max_completion_tokens",
    "n",
    "stop",
    "skip_special_tokens",
}


class PixelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    model: str = Field(min_length=1, max_length=256)
    task: Literal["segmentation", "matting"]
    image_url: str = Field(min_length=1, max_length=_MAX_BASE64 + 256)
    query: str = Field(min_length=1, max_length=MAX_QUERY)
    max_new_tokens: int = Field(default=DEFAULT_TOKENS, ge=1, le=MAX_TOKENS)


class _ChatError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class _ChatBodyTooLarge(Exception):
    pass


class _ChatDisconnected(Exception):
    pass


def _text(status: int, message: str) -> PlainTextResponse:
    return PlainTextResponse(message, status_code=status)


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON fields are not supported")
        result[key] = value
    return result


async def _parse_request(raw_request: Request) -> PixelRequest:
    # Read before starting the disconnect listener, and before JSON allocation.
    media_type = raw_request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        raise HTTPException(415, "Only application/json is supported")
    if raw_request.query_params:
        raise HTTPException(400, "Parameters must be supplied only in the JSON body")
    content_length = raw_request.headers.get("content-length")
    if content_length is not None:
        try:
            length = int(content_length)
        except ValueError as exc:
            raise HTTPException(400, "Invalid Content-Length") from exc
        if length < 0:
            raise HTTPException(400, "Invalid Content-Length")
        if length > _MAX_JSON_BYTES:
            raise HTTPException(413, "JSON request exceeds the image upload limit")
    body = bytearray()
    async for chunk in raw_request.stream():
        if len(body) + len(chunk) > _MAX_JSON_BYTES:
            raise HTTPException(413, "JSON request exceeds the image upload limit")
        body.extend(chunk)
    try:
        data = json.loads(body, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise HTTPException(400, "Invalid JSON object or duplicate fields") from exc
    if not isinstance(data, dict):
        raise HTTPException(400, "The request body must be a JSON object")
    if isinstance(data.get("model"), str) and data["model"] != _MODEL:
        raise HTTPException(404, "The requested model is not served by this endpoint")
    try:
        request = PixelRequest.model_validate(data)
    except ValidationError as exc:
        # Do not echo an entire image data URL through Pydantic's input field.
        errors = exc.errors(include_url=False, include_input=False, include_context=False)
        raise HTTPException(400, errors) from exc
    if not request.query.strip():
        raise HTTPException(400, "query must contain a non-whitespace description")
    return request


async def _fetch_image(source: str, allowed_domains: tuple[str, ...]) -> bytes:
    if source.startswith("data:"):
        header, separator, encoded = source.partition(",")
        if not separator or not re.fullmatch(r"data:image/[A-Za-z0-9.+-]+;base64", header):
            raise _ChatError(400, "Expected a base64 image data URL")
        if len(encoded) > _MAX_BASE64:
            raise _ChatError(413, "Image exceeds 25 MiB")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise _ChatError(400, "Invalid base64 image") from exc
        if len(data) > _MAX_IMAGE_BYTES:
            raise _ChatError(413, "Image exceeds 25 MiB")
        return data

    if len(source) > 8192 or any(ord(char) <= 32 or ord(char) == 127 for char in source):
        raise _ChatError(400, "Invalid HTTPS image URL")
    try:
        url = urlsplit(source)
        valid = (
            url.scheme == "https" and bool(url.hostname) and not url.username
            and not url.password and not url.fragment and (url.port is None or url.port > 0)
        )
    except ValueError as exc:
        raise _ChatError(400, "Invalid HTTPS image URL") from exc
    if not valid:
        raise _ChatError(
            400,
            "Only HTTPS URLs and base64 image data URLs are supported; local paths are forbidden",
        )
    if allowed_domains and url.hostname not in allowed_domains:
        raise _ChatError(400, "The image host is not in allowed_media_domains")
    # The native MediaConnector/HTTPConnection retries transient fetch failures
    # and may cache URLs. This endpoint deliberately performs one bounded fetch,
    # without redirects (including HTTPS-to-local redirect chains) or a cache.
    try:
        timeout = aiohttp.ClientTimeout(total=30, connect=5)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.get(source, allow_redirects=False) as response:
                if 300 <= response.status < 400:
                    raise _ChatError(400, "Image URL redirects are not supported")
                response.raise_for_status()
                if response.content_length is not None and response.content_length > _MAX_IMAGE_BYTES:
                    raise _ChatError(413, "Image exceeds 25 MiB")
                data = bytearray()
                async for chunk in response.content.iter_chunked(64 * 1024):
                    if len(data) + len(chunk) > _MAX_IMAGE_BYTES:
                        raise _ChatError(413, "Image exceeds 25 MiB")
                    data.extend(chunk)
                return bytes(data)
    except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
        raise _ChatError(400, "Unable to fetch the HTTPS image within the bounded download limit") from exc


def _canonical_image(data: bytes) -> tuple[Image.Image, str]:
    """Apply native raster/EXIF/alpha semantics, then share one RGB PNG."""
    try:
        with open_image(BytesIO(data)) as original:
            width, height = original.size
            if width < 1 or height < 1 or width * height > _MAX_PIXELS:
                raise _ChatError(413, "Image exceeds 4,194,304 pixels")
            if getattr(original, "n_frames", 1) != 1:
                raise _ChatError(400, "Exactly one still image is required")
            normalized = normalize_image(original)
            rgb = None
            try:
                rgb = convert_image_mode(normalized, "RGB")
                if rgb is original:
                    # The caller keeps the image after this context closes it.
                    rgb = rgb.copy()
                rgb.load()
                # Do not copy EXIF or ancillary source metadata into the PNG.
                rgb.info.clear()
                with BytesIO() as output:
                    rgb.save(output, format="PNG")
                    if output.tell() > _MAX_IMAGE_BYTES:
                        raise _ChatError(413, "Canonical image exceeds 25 MiB")
                    encoded = base64.b64encode(output.getvalue()).decode("ascii")
                return rgb, encoded
            except BaseException:
                if rgb is not None and rgb is not original:
                    rgb.close()
                raise
            finally:
                if normalized is not original and normalized is not rgb:
                    normalized.close()
    except Image.DecompressionBombError as exc:
        raise _ChatError(413, "Image exceeds the pixel limit") from exc
    except _ChatError:
        raise
    except (OSError, ValueError, SyntaxError) as exc:
        raise _ChatError(400, "Unsupported or corrupt raster image") from exc


async def _decode_rpc(engine: EngineClient, payload: dict) -> object:
    # Cancelling an RPC await does not stop the worker's GPU operation. Drain it
    # before the caller releases the pixel lock, even on repeated cancellation.
    task = asyncio.create_task(engine.collective_rpc("wevisone_decode", args=(payload,)))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled() and task.exception() is not None:
            logger.error("WeVisOne worker failed while draining a cancelled request", exc_info=task.exception())
        raise


def _single_rank(result: object) -> dict:
    if not isinstance(result, (list, tuple)) or len(result) != 1 or not isinstance(result[0], dict):
        raise _ChatError(500, "WeVisOne requires exactly one worker rank and a dictionary RPC result")
    return result[0]


def _native_request(
    system: str, image_b64: str, prompt: str, max_tokens: int, overrides: dict
) -> ChatCompletionRequest:
    sampling = dict(_DEFAULT_SAMPLING)
    sampling.update(overrides)
    return ChatCompletionRequest(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_b64}},
                {"type": "text", "text": prompt},
            ]},
        ],
        max_completion_tokens=max_tokens,
        skip_special_tokens=False,
        stop=["<|im_end|>"],
        stream=False,
        **sampling,
    )


async def _call_chat(
    state: State, raw_request: Request, request: ChatCompletionRequest
) -> tuple[ChatCompletionResponse, str]:
    completion = await state.openai_serving_chat.create_chat_completion(request, raw_request)
    if isinstance(completion, ErrorResponse):
        raise _ChatError(completion.error.code, completion.error.message)
    if not isinstance(completion, ChatCompletionResponse) or len(completion.choices) != 1:
        raise _ChatError(500, "Native chat did not return one non-streaming completion")
    choice = completion.choices[0]
    if choice.finish_reason != "stop":
        raise _ChatError(
            400,
            "Generation did not finish with stop; no artifacts were produced. "
            "Increase max_tokens within 1...4096, shorten the query, or use a smaller image.",
        )
    response = choice.message.content
    if not isinstance(response, str) or not response.strip():
        raise _ChatError(500, "Native chat returned no generated response text")
    return completion, response


async def _decode_pixels(
    state: State, task: str, image_b64: str, size: list[int], prompt: str, response: str
) -> tuple[int, list, dict]:
    async with state.wevisone_pixel_lock:
        result = _single_rank(await _decode_rpc(state.wevisone_engine_client, {
            "model": _MODEL, "task": task, "image_b64": image_b64,
            "prompt": prompt, "response": response,
        }))
    if result.get("image_size") != size:
        raise _ChatError(500, "Worker image dimensions do not match the canonical image")
    objects, boxes, artifacts = (
        result.get("objects"), result.get("boxes_normalized_xyxy"), result.get("artifacts"))
    if (type(objects) is not int or not 0 <= objects <= MAX_OBJECTS
            or not isinstance(boxes, list) or len(boxes) != objects
            or not isinstance(artifacts, dict)):
        raise _ChatError(500, "Worker returned an invalid pixel result")
    for box in boxes:
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(type(value) not in (int, float) or not 0 <= value <= 1 for value in box)
                or not (box[2] > box[0] and box[3] > box[1])):
            raise _ChatError(500, "Worker returned an invalid normalized box")
    if set(artifacts) != expected_artifacts(task, objects):
        raise _ChatError(500, "Worker artifact set is incomplete or contains unknown names")
    for value in artifacts.values():
        if not isinstance(value, str) or not value:
            raise _ChatError(500, "Worker returned an empty artifact payload")
        try:
            base64.b64decode(value, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise _ChatError(500, "Worker returned a non-base64 artifact") from exc
    return objects, boxes, artifacts


def _object_result(task: str, image: Image.Image, size: list[int], response: str) -> dict:
    try:
        marks = parse_marks(response, bare=TASKS[task].kind in ("point", "any"))
        predictions = to_predictions(marks, size)
        artifacts = {}
        if predictions:
            artifacts["overlay.png"] = base64.b64encode(overlay_png(image, marks, size)).decode("ascii")
    except MarkError as error:
        raise _ChatError(500, str(error)) from error
    return {
        "task": task, "image_size": size, "objects": len(predictions),
        "predictions": predictions, "artifacts": artifacts,
    }


async def _pixel_result(
    state: State, task: str, image_b64: str, size: list[int], prompt: str, response: str
) -> dict:
    objects, boxes, artifacts = await _decode_pixels(state, task, image_b64, size, prompt, response)
    try:
        labels = segment_labels(response)
    except MarkError as error:
        raise _ChatError(500, str(error)) from error
    if len(labels) != objects:
        raise _ChatError(500, "Generated <OBJ> boxes and decoded masks differ; no artifacts were produced")
    width, height = size
    predictions = []
    for label, box in zip(labels, boxes):
        x1, y1, x2, y2 = box
        predictions.append({
            "label": label, "type": "box",
            "coords": [round(x1 * width, 1), round(y1 * height, 1),
                       round(x2 * width, 1), round(y2 * height, 1)],
        })
    return {
        "task": task, "image_size": size, "objects": objects,
        "predictions": predictions, "artifacts": artifacts,
    }


async def _infer(
    state: State,
    raw_request: Request,
    task: str,
    system: str,
    image_source: str,
    query: str,
    max_tokens: int,
    overrides: dict,
) -> JSONResponse:
    data = await _fetch_image(image_source, state.wevisone_allowed_domains)
    image, image_b64 = _canonical_image(data)
    size = list(image.size)
    prompt = build_prompt(task, query)
    completion, response = await _call_chat(
        state, raw_request, _native_request(system, image_b64, prompt, max_tokens, overrides))
    if TASKS[task].pixel:
        pixel = await _pixel_result(state, task, image_b64, size, prompt, response)
    else:
        pixel = _object_result(task, image, size, response)
    payload = completion.model_dump()
    payload["choices"][0]["message"]["pixel"] = pixel
    return JSONResponse(content=payload)


def _token_budget(data: dict) -> int:
    values = []
    for key in ("max_tokens", "max_completion_tokens"):
        if key in data:
            value = data[key]
            if type(value) is not int or not 1 <= value <= MAX_TOKENS:
                raise _ChatError(400, f"{key} must be an integer between 1 and {MAX_TOKENS}")
            values.append(value)
    if len(set(values)) > 1:
        raise _ChatError(400, "max_tokens and max_completion_tokens disagree")
    return values[0] if values else DEFAULT_TOKENS


def _messages(messages: object) -> tuple[str, str, str]:
    if not isinstance(messages, list) or not messages:
        raise _ChatError(400, "messages must be a non-empty list")
    system = None
    parts = None
    for message in messages:
        if not isinstance(message, dict):
            raise _ChatError(400, "each message must be an object")
        role, content = message.get("role"), message.get("content")
        if role == "system":
            if system is not None or not isinstance(content, str):
                raise _ChatError(400, "at most one system message with string content is supported")
            system = content
        elif role == "user":
            if parts is not None or not isinstance(content, list):
                raise _ChatError(400, "exactly one user message with a content list is required")
            parts = content
        else:
            raise _ChatError(400, "only system and user messages are supported")
    if parts is None:
        raise _ChatError(400, "exactly one user message with a content list is required")
    image_source = text = None
    for part in parts:
        if not isinstance(part, dict):
            raise _ChatError(400, "each content part must be an object")
        kind = part.get("type")
        if kind == "image_url":
            if image_source is not None:
                raise _ChatError(400, "exactly one image_url part is required")
            value = part.get("image_url")
            if not isinstance(value, dict) or not isinstance(value.get("url"), str):
                raise _ChatError(400, "image_url must contain a url string")
            image_source = value["url"]
        elif kind == "text":
            if text is not None or not isinstance(part.get("text"), str):
                raise _ChatError(400, "exactly one text part with string content is required")
            text = part["text"]
        else:
            raise _ChatError(400, "unsupported content part type")
    if image_source is None or text is None:
        raise _ChatError(400, "exactly one image_url and one text part are required")
    if not text.strip():
        raise _ChatError(400, "text must contain a non-whitespace description")
    if len(text) > MAX_QUERY:
        raise _ChatError(400, f"text must not exceed {MAX_QUERY} characters")
    return (SYSTEM_PROMPT if system is None else system), image_source, text


def _media_type(scope: dict) -> str:
    for name, value in scope["headers"]:
        if name == b"content-type":
            return value.decode("latin-1").split(";", 1)[0].strip().lower()
    return ""


async def _dispatch(scope: dict, receive, body: bytes) -> Response | None:
    try:
        data = json.loads(body, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise _ChatError(400, "Invalid JSON object or duplicate fields") from exc
    if not isinstance(data, dict):
        raise _ChatError(400, "The request body must be a JSON object")
    if "wevisone_task" not in data:
        # The byte scan can match a quoted string inside an ordinary message.
        return None
    if _media_type(scope) != "application/json":
        raise _ChatError(415, "Only application/json is supported")
    if scope.get("query_string"):
        raise _ChatError(400, "Parameters must be supplied only in the JSON body")
    task = data.get("wevisone_task")
    if not isinstance(task, str) or task not in TASKS:
        raise _ChatError(400, "wevisone_task must be one of: " + ", ".join(TASKS))
    model = data.get("model")
    if model is None:
        raise _ChatError(400, "model is required")
    if model != _MODEL:
        raise _ChatError(404, "The requested model is not served by this endpoint")
    if data.get("stream") not in (None, False):
        raise _ChatError(400, "stream is not supported with wevisone_task")
    if data.get("n") is not None and (type(data["n"]) is not int or data["n"] != 1):
        raise _ChatError(400, "n must be 1 for WeVisOne tasks")
    max_tokens = _token_budget(data)
    system, image_source, query = _messages(data.get("messages"))
    overrides = {
        key: value for key, value in data.items()
        if key not in _CONTROLS and value is not None
    }
    state = scope["app"].state
    raw_request = Request(scope, receive)

    async def work() -> Response:
        return await _infer(state, raw_request, task, system, image_source, query,
                            max_tokens, overrides)

    return await _with_disconnect(work(), scope, receive)


async def _chat_dispatch(scope: dict, receive, body: bytes) -> Response | None:
    try:
        return await _dispatch(scope, receive, body)
    except _ChatError as error:
        return _text(error.status, error.message)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("WeVisOne chat request failed")
        return _text(500, "WeVisOne inference failed; no artifacts were produced")


async def _with_disconnect(work_coro, scope: dict, receive) -> Response:
    # Like native chat's with_cancellation, listen only after body consumption.
    # Unlike that decorator, await cleanup so an in-flight decode cannot outlive
    # the handler or release the pixel lock before its GPU work finishes.
    work = asyncio.create_task(work_coro)
    disconnect = asyncio.create_task(_listen_for_disconnect(Request(scope, receive)))
    try:
        done, _ = await asyncio.wait((work, disconnect), return_when=asyncio.FIRST_COMPLETED)
        if work in done:
            return work.result()
        work.cancel()
        return _text(499, "Client disconnected")
    finally:
        disconnect.cancel()
        if not work.done():
            work.cancel()
        # Shield cleanup from ASGI cancellation; the worker RPC itself is also
        # drained inside _decode_rpc before the serialization lock is released.
        cleanup = asyncio.gather(work, disconnect, return_exceptions=True)
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                continue


async def _listen_for_disconnect(request: Request) -> None:
    # Native listen_for_disconnect also decrements native load metrics. These
    # requests do not own that counter, so retain only its ASGI receive loop.
    while True:
        if (await request.receive())["type"] == "http.disconnect":
            return


class _ReplayReceive:
    """Re-feed a buffered body, then defer to the real receive stream."""

    __slots__ = ("_body", "_receive", "_pending")

    def __init__(self, body: bytes, receive) -> None:
        self._body = body
        self._receive = receive
        self._pending = True

    async def __call__(self) -> dict:
        if self._pending:
            self._pending = False
            return {"type": "http.request", "body": self._body, "more_body": False}
        return await self._receive()


async def _read_body(receive) -> bytes:
    chunks = []
    total = 0
    while True:
        message = await receive()
        if message["type"] != "http.request":
            raise _ChatDisconnected
        chunk = message.get("body", b"")
        total += len(chunk)
        if total > _MAX_JSON_BYTES:
            raise _ChatBodyTooLarge
        chunks.append(chunk)
        if not message.get("more_body", False):
            return b"".join(chunks)


async def _deliver(response: Response, scope: dict, receive, send) -> None:
    try:
        await response(scope, receive, send)
    except (OSError, RuntimeError):
        logger.debug("WeVisOne response delivery failed", exc_info=True)


class WeVisOneChatMiddleware:
    """Dispatch ``wevisone_task`` chat requests and replay ordinary ones."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if not (scope["type"] == "http" and scope["method"] == "POST"
                and scope["path"] == _CHAT_PATH):
            return await self.app(scope, receive, send)
        state = getattr(scope.get("app"), "state", None)
        if getattr(state, "wevisone_engine_client", None) is None:
            return await self.app(scope, receive, send)
        try:
            body = await _read_body(receive)
        except _ChatBodyTooLarge:
            return await _deliver(
                _text(413, "JSON request exceeds the 25 MiB upload limit"), scope, receive, send)
        except _ChatDisconnected:
            return
        if b'"wevisone_task"' not in body:
            return await self.app(scope, _ReplayReceive(body, receive), send)
        response = await _chat_dispatch(scope, receive, body)
        if response is None:
            return await self.app(scope, _ReplayReceive(body, receive), send)
        await _deliver(response, scope, receive, send)


async def _predict(request: PixelRequest, raw_request: Request) -> JSONResponse:
    state = raw_request.app.state
    data = await _fetch_image(request.image_url, state.wevisone_allowed_domains)
    image, image_b64 = _canonical_image(data)
    size = list(image.size)
    query = request.query.strip()
    prompt = build_prompt(request.task, query)
    try:
        completion, response = await _call_chat(
            state, raw_request,
            _native_request(SYSTEM_PROMPT, image_b64, prompt, request.max_new_tokens, {}))
        objects, boxes, artifacts = await _decode_pixels(
            state, request.task, image_b64, size, prompt, response)
    except _ChatError as error:
        raise HTTPException(error.status, error.message) from error
    return JSONResponse(
        content={
            "model": _MODEL, "task": request.task, "response": response,
            "usage": completion.usage.model_dump(), "image_size": size,
            "boxes_normalized_xyxy": boxes, "objects": objects, "artifacts": artifacts,
        },
        headers={
            "Deprecation": "true",
            "Link": '</v1/chat/completions>; rel="successor-version"',
        },
    )


async def _predict_route(raw_request: Request, request: PixelRequest = Depends(_parse_request)):
    # Like native chat's with_cancellation, listen only after body consumption.
    # Unlike that decorator, await cleanup so an in-flight decode cannot outlive
    # the handler or release the lock before its GPU work finishes.
    work = asyncio.create_task(_predict(request, raw_request))
    disconnect = asyncio.create_task(_listen_for_disconnect(raw_request))
    try:
        done, _ = await asyncio.wait((work, disconnect), return_when=asyncio.FIRST_COMPLETED)
        if work in done:
            return work.result()
        work.cancel()
        raise HTTPException(499, "Client disconnected")
    except HTTPException:
        raise
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("WeVisOne pixel request failed")
        raise HTTPException(500, "WeVisOne pixel inference failed; no artifacts were produced") from exc
    finally:
        disconnect.cancel()
        if not work.done():
            work.cancel()
        # Shield cleanup from ASGI cancellation; the worker RPC itself is also
        # drained inside _decode_rpc before the serialization lock is released.
        cleanup = asyncio.gather(work, disconnect, return_exceptions=True)
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                continue


class PixelEndpointPlugin:
    """Zero-argument vllm.endpoint_plugins entry-point factory."""

    name = "wevisone_vl_metal"
    required_tasks = ("generate",)

    def attach_router(self, app: FastAPI) -> None:
        router = APIRouter()
        router.add_api_route(_PATH, _predict_route, methods=["POST"])
        app.include_router(router)
        # Registered before upstream auth/CORS middleware, so this stays the
        # innermost layer and only sees authenticated requests.
        app.add_middleware(WeVisOneChatMiddleware)

    async def init_state(self, engine_client: EngineClient | None, state: State, args: Namespace) -> None:
        if engine_client is None or getattr(state, "openai_serving_chat", None) is None:
            raise RuntimeError("WeVisOne pixel endpoints require the native generation/chat engine")
        if getattr(args, "served_model_name", None) != [_MODEL]:
            raise RuntimeError("WeVisOne pixel endpoints require explicit --served-model-name wevisone-vl-4b")
        if getattr(args, "api_server_count", 1) not in (None, 1):
            raise RuntimeError("WeVisOne pixel serialization requires one API server process")
        if engine_client.vllm_config.parallel_config.worker_extension_cls != "wevisone_vl_metal.worker.PixelWorkerExtension":
            raise RuntimeError("WeVisOne pixel endpoints require the fixed PixelWorkerExtension opt-in")
        initialized = _single_rank(await engine_client.collective_rpc("wevisone_initialize"))
        if initialized.get("initialized") is not True:
            raise RuntimeError("WeVisOne worker did not initialize its MPS pixel decoder")
        state.wevisone_engine_client = engine_client
        state.wevisone_pixel_lock = asyncio.Lock()
        state.wevisone_allowed_domains = tuple(engine_client.model_config.allowed_media_domains or ())
