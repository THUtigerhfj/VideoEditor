from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SCRIBBLE_CONTROLNET_ID = "xinsir/controlnet-scribble-sdxl-1.0"


def _cleanup_incomplete_snapshot(repo_id: str, cache_dir) -> int:
    from pathlib import Path

    cache_path = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}"
    removed = 0
    if not cache_path.exists():
        return removed
    for partial in cache_path.rglob("*.incomplete"):
        partial.unlink(missing_ok=True)
        removed += 1
    return removed


def _is_cuda_device(device: str) -> bool:
    return str(device).startswith("cuda")


def required_snapshot_patterns(repo_id: str, device: str = "cuda") -> tuple[str, ...] | None:
    if repo_id == SCRIBBLE_CONTROLNET_ID:
        return (
            "config.json",
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.fp16.safetensors",
        )
    if repo_id == SDXL_MODEL_ID:
        fp16 = _is_cuda_device(device)
        text_weight = "model.fp16.safetensors" if fp16 else "model.safetensors"
        unet_weight = "diffusion_pytorch_model.fp16.safetensors" if fp16 else "diffusion_pytorch_model.safetensors"
        vae_weight = "diffusion_pytorch_model.fp16.safetensors" if fp16 else "diffusion_pytorch_model.safetensors"
        return (
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            f"text_encoder/{text_weight}",
            "text_encoder_2/config.json",
            f"text_encoder_2/{text_weight}",
            "tokenizer/*",
            "tokenizer_2/*",
            "unet/config.json",
            f"unet/{unet_weight}",
            "vae/config.json",
            f"vae/{vae_weight}",
        )
    return None


def _ensure_snapshot(repo_id: str, cache_dir, device: str = "cuda"):
    from huggingface_hub import snapshot_download

    _cleanup_incomplete_snapshot(repo_id, cache_dir)
    kwargs = {
        "repo_id": repo_id,
        "cache_dir": cache_dir,
        "resume_download": True,
    }
    allow_patterns = required_snapshot_patterns(repo_id, device=device)
    if allow_patterns:
        kwargs["allow_patterns"] = list(allow_patterns)

    try:
        return snapshot_download(local_files_only=True, **kwargs)
    except Exception:
        return snapshot_download(**kwargs)


def _from_pretrained_cache_first(loader, model_id: str, **kwargs):
    try:
        return loader.from_pretrained(model_id, local_files_only=True, **kwargs)
    except Exception as local_error:
        print(f"Local cache load failed for {model_id}; trying Hugging Face Hub: {local_error}")
        return loader.from_pretrained(model_id, **kwargs)


def _to_rgb(image: Image.Image) -> Image.Image:
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    return image.convert("RGB")


PRODUCT_LABELS = {
    "can",
    "cup",
    "yogurt cup",
    "bottle",
    "jar",
    "carton",
    "container",
    "package",
    "packaging",
}

FRUIT_LABELS = {
    "apple",
    "banana",
    "orange",
    "pear",
    "peach",
    "lemon",
    "mango",
    "fruit",
}


def _label_matches(label: str, vocabulary: set[str]) -> bool:
    normalized = " ".join(str(label).lower().replace("_", " ").split())
    return any(term in normalized for term in vocabulary)


def build_reference_prompt(label: str, attrs: str | None) -> tuple[str, str]:
    clean_label = str(label).strip()
    if not clean_label:
        raise ValueError("label is required")
    attrs = (attrs or "").strip()
    attr_phrase = f", {attrs}" if attrs else ""
    base = (
        f"single isolated {clean_label}{attr_phrase}, photorealistic studio product photo, "
        "centered composition, full object visible, clean silhouette, plain light background, "
        "real camera photograph, high detail, realistic shadows, realistic surface reflections"
    )
    negative_common = (
        "cluttered background, multiple objects, cropped object, occlusion, blur, low detail, "
        "watermark, busy scene, cartoon, illustration, anime, cgi, 3d render, toy-like, flat color, "
        "line drawing, sketch, black outline"
    )

    if _label_matches(clean_label, PRODUCT_LABELS):
        prompt = (
            f"{base}, cylindrical product packaging, realistic printed label area, "
            "subtle material imperfections, metal or plastic surface, commercial product photography"
        )
        negative = negative_common + ", unreadable messy text, deformed packaging"
        return prompt, negative

    if _label_matches(clean_label, FRUIT_LABELS):
        prompt = (
            f"{base}, natural fruit texture, subtle color variation, realistic surface imperfections, "
            "visible stem, real produce photography, true-to-life shading"
        )
        negative = negative_common + ", wax fruit, synthetic surface"
        return prompt, negative

    prompt = (
        f"{base}, physically plausible material, subtle texture variation, natural lighting, "
        "true-to-life shading"
    )
    negative = negative_common + ", synthetic surface"
    return prompt, negative

def preprocess_sketch_for_scribble(sketch: Image.Image, size: int = 1024) -> Image.Image:
    if size <= 0:
        raise ValueError("size must be positive")
    rgb = np.asarray(_to_rgb(sketch), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if gray.mean() < 127:
        gray = 255 - gray
    _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    coords = cv2.findNonZero(binary_inv)
    if coords is None:
        canvas = np.full((size, size, 3), 255, dtype=np.uint8)
        return Image.fromarray(canvas)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = binary_inv[y : y + h, x : x + w]
    target_side = max(1, int(size * 0.78))
    scale = min(target_side / max(w, 1), target_side / max(h, 1))
    resized = cv2.resize(cropped, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((size, size), 255, dtype=np.uint8)
    y0 = (size - resized.shape[0]) // 2
    x0 = (size - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = np.where(resized > 0, 0, 255).astype(np.uint8)
    canvas = np.stack([canvas, canvas, canvas], axis=-1)
    return Image.fromarray(canvas)


def _mask_bbox(mask_arr: np.ndarray):
    binary = mask_arr > 0
    ys, xs = np.where(binary)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def build_mask_shape_scribble(target_mask, size: int = 1024, contour_width: int = 5) -> Image.Image:
    if size <= 0:
        raise ValueError("size must be positive")
    mask = np.asarray(target_mask, dtype=np.uint8)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 0).astype(np.uint8) * 255
    bbox = _mask_bbox(mask)
    canvas = np.full((size, size), 255, dtype=np.uint8)
    if bbox is None:
        return Image.fromarray(np.stack([canvas, canvas, canvas], axis=-1)).convert("RGB")

    x0, y0, x1, y1 = bbox
    crop = mask[y0 : y1 + 1, x0 : x1 + 1]
    target_side = max(1, int(size * 0.78))
    scale = min(target_side / max(crop.shape[1], 1), target_side / max(crop.shape[0], 1))
    new_w = max(1, int(round(crop.shape[1] * scale)))
    new_h = max(1, int(round(crop.shape[0] * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    contour = np.zeros_like(resized)
    contours, _ = cv2.findContours((resized > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour, contours, -1, 255, max(1, int(contour_width)))
    y = (size - new_h) // 2
    x = (size - new_w) // 2
    canvas[y : y + new_h, x : x + new_w] = np.where(contour > 0, 0, canvas[y : y + new_h, x : x + new_w])
    return Image.fromarray(np.stack([canvas, canvas, canvas], axis=-1)).convert("RGB")


def build_shape_conditioned_scribble(
    sketch: Image.Image,
    target_mask,
    size: int = 1024,
    sketch_mask_fit_strength: float = 0.5,
    mask_contour_weight: float = 0.6,
) -> Image.Image:
    sketch_scribble = preprocess_sketch_for_scribble(sketch, size=size).convert("L")
    if target_mask is None or mask_contour_weight <= 0:
        return Image.merge("RGB", (sketch_scribble, sketch_scribble, sketch_scribble))

    mask_scribble = build_mask_shape_scribble(target_mask, size=size).convert("L")
    sketch_arr = np.asarray(sketch_scribble, dtype=np.uint8)
    mask_arr = np.asarray(mask_scribble, dtype=np.uint8)
    sketch_lines = sketch_arr < 128
    mask_lines = mask_arr < 128

    # Keep user sketch semantics, then add target contour as spatial guidance.
    combined = np.full((size, size), 255, dtype=np.uint8)
    combined[np.logical_or(sketch_lines, mask_lines)] = 0
    return Image.fromarray(np.stack([combined, combined, combined], axis=-1)).convert("RGB")


@lru_cache(maxsize=2)
def build_sdxl_scribble_pipeline(cache_dir: str, device: str = "cuda"):
    import torch
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

    torch_dtype = torch.float16 if _is_cuda_device(device) else torch.float32
    variant = "fp16" if _is_cuda_device(device) else None
    _ensure_snapshot(SCRIBBLE_CONTROLNET_ID, cache_dir, device=device)
    _ensure_snapshot(SDXL_MODEL_ID, cache_dir, device=device)
    controlnet = _from_pretrained_cache_first(
        ControlNetModel,
        SCRIBBLE_CONTROLNET_ID,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
        use_safetensors=True,
    )
    pipe = _from_pretrained_cache_first(
        StableDiffusionXLControlNetPipeline,
        SDXL_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
        use_safetensors=True,
        variant=variant,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)
    return pipe


def generate_reference_candidates(
    pipe,
    sketch: Image.Image,
    label: str,
    attrs: str | None,
    seeds: Iterable[int] | None = None,
    num_candidates: int = 2,
    num_inference_steps: int = 40,
    guidance_scale: float = 6.5,
    controlnet_conditioning_scale: float = 0.7,
    size: int = 1024,
    target_mask_for_shape=None,
    shape_conditioned_scribble: bool = False,
    sketch_mask_fit_strength: float = 0.5,
    mask_contour_weight: float = 0.6,
):
    import torch

    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive")
    prompt, negative_prompt = build_reference_prompt(label, attrs)
    if shape_conditioned_scribble and target_mask_for_shape is not None:
        scribble = build_shape_conditioned_scribble(
            sketch,
            target_mask=target_mask_for_shape,
            size=size,
            sketch_mask_fit_strength=sketch_mask_fit_strength,
            mask_contour_weight=mask_contour_weight,
        )
    else:
        scribble = preprocess_sketch_for_scribble(sketch, size=size)
    seeds = list(seeds or range(42, 42 + num_candidates))[:num_candidates]
    candidates = []
    for seed in seeds:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=scribble,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            height=size,
            width=size,
            generator=torch.Generator(device="cpu").manual_seed(int(seed)),
        )
        candidates.append({
            "seed": int(seed),
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "scribble": scribble.copy(),
            "image": result.images[0].convert("RGB"),
        })
    return candidates
