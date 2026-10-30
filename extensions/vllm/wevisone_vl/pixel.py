"""Capture the released decoder's inputs from the existing MLX backbone."""

import base64
import io
import re

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from vllm_metal.attention.context import get_context
from vllm_metal.tensor_bridge import mlx_to_torch

_BOX = re.compile(r"<box>(.*?)(?:</box>|$)", re.DOTALL | re.IGNORECASE)
_OBJ = re.compile(r"<OBJ>", re.IGNORECASE)
_COORD = re.compile(r"<(\d{1,4})>")


def adaptive_avg_pool2d_mps(value, output_size):
    """Exact adaptive bin boundaries, without the MPS divisibility restriction."""
    if value.device.type != "mps" or value.dtype != torch.float32 or value.ndim != 4:
        raise ValueError("pixel pooling requires an NCHW FP32 MPS tensor")
    if any(type(size) is not int or size < 1 for size in output_size):
        raise ValueError("invalid adaptive pooling size")
    if all(source % target == 0 for source, target in zip(value.shape[-2:], output_size)):
        return F.adaptive_avg_pool2d(value, output_size)

    def reduce_axis(current, target, axis):
        source = current.shape[axis]
        if source % target == 0:
            shape = list(current.shape)
            shape[axis:axis + 1] = [target, source // target]
            return current.reshape(shape).mean(axis + 1)
        bins = torch.arange(target, device=current.device)
        start = bins * source // target
        end = ((bins + 1) * source + target - 1) // target
        offsets = torch.arange((source + target - 1) // target + 1, device=current.device)
        indices = start[:, None] + offsets
        valid = indices < end[:, None]
        selected = current.index_select(axis, indices.clamp(max=source - 1).flatten())
        shape = list(current.shape)
        shape[axis:axis + 1] = [target, offsets.numel()]
        mask_shape = [1] * len(shape)
        mask_shape[axis:axis + 2] = [target, offsets.numel()]
        summed = (selected.reshape(shape) * valid.reshape(mask_shape)).sum(axis + 1)
        divisor_shape = [1] * summed.ndim
        divisor_shape[axis] = target
        return summed / (end - start).reshape(divisor_shape)

    return reduce_axis(reduce_axis(value, output_size[1], 3), output_size[0], 2)


def pixel_fusion_forward(self, image, encoded_levels):
    """Pinned AdditivePixelFusion.forward, changing only adaptive pooling."""
    fpn = self.upsample(self.fpn(encoded_levels))
    cnn = self.cnn(image)
    target_hw = tuple(max(1, int(value) // 2) for value in cnn.shape[-2:])
    if tuple(fpn.shape[-2:]) != target_hw:
        src_h, src_w = (int(value) for value in fpn.shape[-2:])
        dst_h, dst_w = target_hw
        resize = fpn.float()
        pool_h, pool_w = min(src_h, dst_h), min(src_w, dst_w)
        if (pool_h, pool_w) != (src_h, src_w):
            resize = adaptive_avg_pool2d_mps(resize, (pool_h, pool_w))
        if (pool_h, pool_w) != (dst_h, dst_w):
            resize = F.interpolate(resize, size=(dst_h, dst_w),
                                   mode="bilinear", align_corners=False)
        fpn = resize.to(dtype=fpn.dtype)
    fused = self.smooth(self.fpn_proj(fpn))
    fused = self.s2_upsample(fused)
    return fused + self.cnn_proj(cnn)


def parse_boxes(response):
    boxes = []
    for body in _BOX.findall(response.split("<|im_end|>", 1)[0]):
        for prefix in _OBJ.split(body)[:-1]:
            values = [int(v) for v in _COORD.findall(prefix)]
            if len(values) >= 4:
                x1, y1, x2, y2 = values[-4:]
                if all(0 <= v <= 1000 for v in (x1, y1, x2, y2)) and x2 > x1 and y2 > y1:
                    boxes.append([v / 1000 for v in (x1, y1, x2, y2)])
    return boxes


class _Capture(nn.Module):
    def __init__(self, inner, rows, index):
        super().__init__()
        self.inner = inner
        # Private attributes are not MLX module state.
        object.__setattr__(self, "_rows", rows)
        object.__setattr__(self, "_index", index)

    def __call__(self, x):
        self._rows[self._index] = x
        return self.inner(x)


def capture_features(model, processor, image, prompt, response, obj_id):
    """Return MLX arrays for pre-norm OBJ rows and four raw vision maps.

    The caller must retain these arrays through the Torch/MPS synchronization.
    """
    if get_context() is not None:
        raise RuntimeError("pixel capture cannot run with a paged attention context")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "image", "image": image},
                                      {"type": "text", "text": prompt}]},
        {"role": "assistant", "content": response},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="np",
                       images_kwargs={"size": {"shortest_edge": 16384, "longest_edge": 2073600}})
    ids_np = inputs["input_ids"]
    if ids_np.shape[0] != 1 or ids_np.shape[1] > 8192:
        raise ValueError("teacher-forced input exceeds the single-request context")
    positions = np.flatnonzero(ids_np[0] == obj_id).tolist()
    if not positions or len(positions) > 16:
        raise RuntimeError("teacher-forced input must contain 1 to 16 OBJ tokens")
    grid = np.asarray(inputs["image_grid_thw"])
    if grid.shape != (1, 3) or int(grid[0, 0]) != 1:
        raise RuntimeError("expected one still-image grid")
    h, w = map(int, grid[0, 1:])
    if h % 2 or w % 2:
        raise RuntimeError("image grid is not merge-aligned")
    ids = mx.array(ids_np)
    pixels = mx.array(inputs["pixel_values"])
    grid_mx = mx.array(grid)
    tower = model.vision_tower
    original_list = tower.deepstack_merger_list
    original_final = tower.merger
    captured = [None] * 4
    try:
        tower.deepstack_merger_list = [_Capture(m, captured, i) for i, m in enumerate(original_list)]
        tower.merger = _Capture(original_final, captured, 3)
        hidden = model.hidden_state_at_layer(ids, layer=36, pixel_values=pixels,
                                             image_grid_thw=grid_mx, mask=None)
        if any(x is None or x.shape != (h * w, 1024) for x in captured):
            raise RuntimeError("unexpected pre-merger feature shape")
        rows = hidden[0, mx.array(positions), :]
        maps = [x.reshape(h // 2, w // 2, 2, 2, 1024)
                .transpose(0, 2, 1, 3, 4).reshape(h, w, 1024)
                .transpose(2, 0, 1)[None] for x in captured]
        mx.eval(rows, *maps)
        mx.synchronize()
        return rows, maps, (h, w)
    finally:
        tower.deepstack_merger_list = original_list
        tower.merger = original_final
        mx.synchronize()


def _png(image):
    output = io.BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def decode(decoder, model, processor, obj_id, cnn_size, payload):
    if payload.get("model") != "wevisone-vl-4b" or payload.get("task") not in ("segmentation", "matting"):
        raise ValueError("unsupported model or task")
    prompt, response = payload["prompt"], payload["response"]
    if not isinstance(prompt, str) or not isinstance(response, str):
        raise ValueError("prompt and response must be strings")
    data = base64.b64decode(payload["image_b64"], validate=True)
    if len(data) > 25 * 1024 * 1024:
        raise ValueError("image too large")
    with Image.open(io.BytesIO(data)) as source:
        if source.format != "PNG" or source.mode != "RGB" or source.width * source.height > 4194304:
            raise ValueError("expected bounded canonical RGB PNG")
        image = source.copy()
    empty = {"image_size": list(image.size), "boxes_normalized_xyxy": [], "objects": 0, "artifacts": {}}
    boxes = parse_boxes(response)
    if not boxes:
        absent = response.split("<|im_end|>", 1)[0].split("<|endoftext|>", 1)[0].strip()
        explicit_none = re.fullmatch(
            r"(?:<ref>[^<>]*</ref>\s*)?(?:none|<box>\s*none\s*</box>)",
            absent, re.IGNORECASE)
        if _OBJ.search(response) or explicit_none is None:
            raise ValueError("response is neither valid OBJ boxes nor explicit None")
        return empty
    if len(boxes) > 16:
        raise ValueError("more than 16 objects")
    arrays = None
    try:
        arrays = capture_features(model, processor, image, prompt, response, obj_id)
        rows, maps, grid = arrays
        if rows.shape[0] != len(boxes):
            raise ValueError("OBJ token and valid box counts differ")
        with torch.inference_mode():
            visual = [mlx_to_torch(x, device="mps").float() for x in maps]
            text = mlx_to_torch(rows, device="mps").float()
            rgb = np.asarray(image.resize((cnn_size, cnn_size), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
            cnn = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().unsqueeze(0).to("mps")
            masks = []
            for index, box in enumerate(boxes):
                result = decoder(image=cnn, visual_features_2d=visual, t_seg=text[index:index + 1],
                                 boxes01=torch.tensor([box], dtype=torch.float32, device="mps"), grid_hw=grid)
                logits = result["mask_logits"]
                if logits.ndim == 4 and logits.shape[1] == 1:
                    logits = logits[:, 0]
                if logits.ndim == 2:
                    logits = logits[None]
                if logits.ndim != 3 or logits.shape[0] != 1:
                    raise RuntimeError("unexpected decoder mask shape")
                probability = F.interpolate(logits.float().sigmoid()[None], size=(image.height, image.width),
                                            mode="bilinear", align_corners=False)[0, 0]
                if not bool(torch.isfinite(probability).all().item()):
                    raise RuntimeError("decoder produced non-finite probabilities")
                masks.append(probability.cpu().numpy())
        union = np.maximum.reduce(masks)
        alpha = Image.fromarray(np.uint8(np.clip(union, 0, 1) * 255))
        cutout = image.copy().convert("RGBA")
        cutout.putalpha(alpha)
        overlay_alpha = alpha
        if payload["task"] == "segmentation":
            overlay_alpha = Image.fromarray((union >= 0.5).astype(np.uint8) * 255)
        overlay = Image.blend(image, Image.composite(
            Image.new("RGB", image.size, (255, 0, 0)), image, overlay_alpha), 0.5)
        artifacts = {"alpha.png": _png(alpha), "cutout.png": _png(cutout), "overlay.png": _png(overlay)}
        for i, mask in enumerate(masks):
            artifacts[f"masks/mask_{i:03d}.png"] = _png(
                Image.fromarray(np.uint8(np.clip(mask, 0, 1) * 255)))
        if payload["task"] == "segmentation":
            artifacts["segmentation.png"] = _png(overlay_alpha)
        return {"image_size": list(image.size), "boxes_normalized_xyxy": boxes, "objects": len(boxes), "artifacts": artifacts}
    finally:
        try:
            mx.synchronize()
        finally:
            torch.mps.synchronize()
