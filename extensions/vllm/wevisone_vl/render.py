"""Server-side object overlay using the documented client geometry.

Colours, line widths, point radii, text placement, and the default bitmap
font match the Bash ``save_result`` client so migrated requests keep an
identical overlay for the same image and marks.
"""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw

from .tasks import Mark, shape_pixels

COLORS = ("red", "lime", "dodgerblue", "orange", "magenta", "cyan")


def overlay_png(image: Image.Image, marks: tuple[Mark, ...], size: list[int]) -> bytes:
    """Draw boxes, points, and labels on a copy of the canonical RGB image."""
    canvas = image.copy()
    canvas.load()
    draw = ImageDraw.Draw(canvas)
    width = size[0]
    thickness = max(2, round(width / 400))
    radius = max(6, round(width / 85))
    for index, mark in enumerate(marks):
        color = COLORS[index % len(COLORS)]
        for shape in mark.shapes:
            coords = shape_pixels(shape, size)
            if mark.kind == "box":
                draw.rectangle(coords, outline=color, width=thickness)
            else:
                x, y = coords
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill=color,
                    outline="white",
                    width=thickness,
                )
            draw.text((coords[0], max(0, coords[1] - 14)), mark.label, fill=color)
    output = BytesIO()
    canvas.save(output, format="PNG")
    return output.getvalue()
