"""Task prompts, mark parsing, and pixel conversion for WeVisOne-VL.

Prompt text mirrors the released ``wevisone_vl/prompts.py`` at commit
``9a6bcd85404505df0d05cfb5d97512ccacf35dee`` so the served checkpoint sees
the same instructions as the official engine. This module is pure CPU: it
imports neither vLLM nor model weights.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

SYSTEM_PROMPT = "You are a helpful assistant."
MAX_OBJECTS = 16
MAX_QUERY = 4096
MAX_TOKENS = 4096
DEFAULT_TOKENS = 2048

_REF = re.compile(
    r"<ref>(.*?)</ref>\s*<(box|point|polygon)>(.*?)(?:</(?:box|point|polygon)>|<ref>|$)",
    re.DOTALL | re.IGNORECASE,
)
_BARE = re.compile(
    r"<(box|point|polygon)>(.*?)(?:</\1>|<(?:box|point|polygon)>|$)",
    re.DOTALL | re.IGNORECASE,
)
_BIN = re.compile(r"<(\d+)>")
_COORD = re.compile(r"<(\d{1,4})>")
_OBJ = re.compile(r"<OBJ>", re.IGNORECASE)
_NONE = re.compile(
    r"(?:<ref>[^<>]*</ref>\s*)?(?:none|<box>\s*none\s*</box>)", re.IGNORECASE
)
_SPECIAL = ("<|im_end|>", "<|endoftext|>")


class MarkError(ValueError):
    """Generated marks are missing, malformed, or outside the task contract."""


@dataclass(frozen=True)
class Task:
    kind: Literal["box", "point", "any"]
    pixel: bool


TASKS: dict[str, Task] = {
    "detection": Task("box", False),
    "referring": Task("box", False),
    "pointing": Task("point", False),
    "counting": Task("any", False),
    "ocr": Task("box", False),
    "segmentation": Task("box", True),
    "matting": Task("box", True),
}


@dataclass(frozen=True)
class Mark:
    label: str
    kind: Literal["box", "point"]
    shapes: tuple[tuple[int, ...], ...]


def _labels(query: str) -> list[str]:
    return [item.strip() for item in query.splitlines() if item.strip()]


def _text(query: str) -> str:
    return query.strip()


def _phrase(query: str) -> str:
    text = _text(query)
    return text if not text or text[-1] in ".?!" else text + "."


def build_prompt(task: str, query: str) -> str:
    """Return the official instruction for ``task`` and ``query``."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}")
    if task == "detection":
        return (
            "Locate all the instances that match the following description: "
            + ". ".join(_labels(query))
            + "."
        )
    if task == "referring":
        return (
            "Locate the instances that match the following description: "
            + _phrase(query)
        )
    if task == "pointing":
        return "point to: " + ". ".join(_labels(query))
    if task == "counting":
        return f"How many {_text(query)} in the image? Please count them with boxes."
    if task in ("segmentation", "matting"):
        return (
            "Locate the instances that match the following description and "
            "provide the alpha mattes: " + _phrase(query)
        )
    return "detect all the word in the box format"


def strip_special(text: str) -> str:
    for token in _SPECIAL:
        text = text.split(token, 1)[0]
    return text


def parse_marks(text: str, *, bare: bool = False) -> tuple[Mark, ...]:
    """Parse ``<ref>`` marks; ``<box>None</box>`` means an explicit no-target."""
    text = strip_special(text)
    groups = _REF.findall(text)
    if groups:
        return tuple(
            Mark(
                (label.strip() or "object"),
                kind.lower(),  # type: ignore[arg-type]
                _shapes(kind.lower(), body),
            )
            for label, kind, body in groups
        )
    if _NONE.fullmatch(text.strip()):
        return ()
    if bare:
        marks = tuple(
            Mark("object", kind.lower(), _shapes(kind.lower(), body))  # type: ignore[arg-type]
            for kind, body in _BARE.findall(text)
        )
        if marks:
            return marks
    if not text.strip():
        raise MarkError("the generated response is empty")
    raise MarkError("the generated response contains neither object marks nor an explicit none")


def _shapes(kind: str, body: str) -> tuple[tuple[int, ...], ...]:
    if kind == "polygon":
        raise MarkError("polygon marks are not part of this endpoint contract")
    if body.strip().casefold() == "none":
        return ()
    size = 2 if kind == "point" else 4
    values = [int(value) for value in _BIN.findall(body)]
    if not values or len(values) % size:
        raise MarkError("object coordinates are incomplete")
    if any(not 0 <= value <= 1000 for value in values):
        raise MarkError("object coordinates are outside 0...1000")
    shapes = []
    for offset in range(0, len(values), size):
        shape = tuple(values[offset : offset + size])
        if kind == "box" and not (shape[2] > shape[0] and shape[3] > shape[1]):
            raise MarkError("a generated box has no positive area")
        shapes.append(shape)
    return tuple(shapes)


def shape_pixels(shape: tuple[int, ...], size: list[int]) -> list[float]:
    """Convert bins in ``shape`` to unrounded original-image pixels."""
    width, height = size
    return [
        value / 1000 * (width if axis % 2 == 0 else height)
        for axis, value in enumerate(shape)
    ]


def to_predictions(marks: tuple[Mark, ...], size: list[int]) -> list[dict]:
    """Flatten marks into pixel predictions rounded to one decimal."""
    predictions = []
    for mark in marks:
        for shape in mark.shapes:
            coords = [round(value, 1) for value in shape_pixels(shape, size)]
            predictions.append({"label": mark.label, "type": mark.kind, "coords": coords})
    if len(predictions) > MAX_OBJECTS:
        raise MarkError(f"the response produced more than {MAX_OBJECTS} objects")
    return predictions


def segment_labels(text: str) -> list[str]:
    """Labels for the ``<OBJ>`` boxes, in the worker's exact scan order."""
    labels = []
    for label, kind, body in _REF.findall(strip_special(text)):
        if kind.lower() != "box":
            raise MarkError("segmentation responses must contain boxes")
        for prefix in _OBJ.split(body)[:-1]:
            values = [int(value) for value in _COORD.findall(prefix)]
            if len(values) < 4:
                continue
            x1, y1, x2, y2 = values[-4:]
            if all(0 <= value <= 1000 for value in (x1, y1, x2, y2)) and x2 > x1 and y2 > y1:
                labels.append(label.strip() or "object")
    return labels


def expected_artifacts(task: str, objects: int) -> set[str]:
    """Exact artifact name set for a successful response."""
    if objects == 0:
        return set()
    if not TASKS[task].pixel:
        return {"overlay.png"}
    names = {"overlay.png", "alpha.png", "cutout.png"}
    names.update(f"masks/mask_{index:03d}.png" for index in range(objects))
    if task == "segmentation":
        names.add("segmentation.png")
    return names
