from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage
from PIL import Image

GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
BIREFNET_MODEL_ID = "ZhengPeng7/BiRefNet"


def model_cache_dir(repo_id: str, cache_dir) -> Path:
    return Path(cache_dir) / f"models--{repo_id.replace('/', '--')}"


def cleanup_incomplete_model_blobs(repo_id: str, cache_dir) -> int:
    cache_path = model_cache_dir(repo_id, cache_dir)
    removed = 0
    if not cache_path.exists():
        return removed
    for partial in cache_path.rglob('*.incomplete'):
        partial.unlink(missing_ok=True)
        removed += 1
    return removed


def ensure_model_snapshot(repo_id: str, cache_dir):
    from huggingface_hub import snapshot_download

    cleanup_incomplete_model_blobs(repo_id, cache_dir)
    return snapshot_download(repo_id=repo_id, cache_dir=cache_dir, resume_download=True)


def _to_uint8_mask(mask) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.uint8)


def clean_binary_mask(mask) -> np.ndarray:
    binary = _to_uint8_mask(mask)
    labeled, count = scipy.ndimage.label(binary)
    if count == 0:
        return (binary * 255).astype(np.uint8)
    component_sizes = scipy.ndimage.sum(binary, labeled, range(1, count + 1))
    largest_component = 1 + int(np.argmax(component_sizes))
    cleaned = (labeled == largest_component).astype(np.uint8)
    cleaned = scipy.ndimage.binary_fill_holes(cleaned).astype(np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return (cleaned * 255).astype(np.uint8)


def _largest_component_ratio(mask_binary: np.ndarray) -> float:
    total = float(mask_binary.sum())
    if total <= 0:
        return 0.0
    labeled, count = scipy.ndimage.label(mask_binary)
    if count == 0:
        return 0.0
    component_sizes = scipy.ndimage.sum(mask_binary, labeled, range(1, count + 1))
    return float(np.max(component_sizes) / total)


def _bbox_from_mask(mask_binary: np.ndarray):
    ys, xs = np.where(mask_binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def score_reference_candidate(image, mask, detection_score: float) -> dict:
    image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
    image_arr = np.asarray(image, dtype=np.uint8)
    cleaned = clean_binary_mask(mask)
    mask_binary = (cleaned > 0).astype(np.uint8)
    area_ratio = float(mask_binary.mean())
    largest_component_ratio = _largest_component_ratio(mask_binary)
    touches_border = bool(mask_binary[0].any() or mask_binary[-1].any() or mask_binary[:, 0].any() or mask_binary[:, -1].any())

    bbox = _bbox_from_mask(mask_binary)
    centeredness = 0.0
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dx = abs(cx - (image_arr.shape[1] / 2.0)) / max(image_arr.shape[1] / 2.0, 1.0)
        dy = abs(cy - (image_arr.shape[0] / 2.0)) / max(image_arr.shape[0] / 2.0, 1.0)
        centeredness = max(0.0, 1.0 - (dx + dy) / 2.0)

    gray = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)
    background = gray[mask_binary == 0]
    background_complexity = float(background.std() / 255.0) if background.size else 0.0
    foreground = gray[mask_binary > 0]
    object_texture = float(foreground.std() / 255.0) if foreground.size else 0.0

    if area_ratio <= 0:
        area_score = 0.0
    else:
        area_score = max(0.0, 1.0 - abs(area_ratio - 0.28) / 0.28)

    background_score = max(0.0, 1.0 - min(background_complexity / 0.35, 1.0))
    texture_score = min(object_texture / 0.12, 1.0)
    score = (
        float(detection_score) * 0.4
        + largest_component_ratio * 0.2
        + area_score * 0.15
        + centeredness * 0.1
        + background_score * 0.05
        + texture_score * 0.1
    )
    if touches_border:
        score -= 0.35

    return {
        "score": float(score),
        "detection_score": float(detection_score),
        "mask_area_ratio": area_ratio,
        "largest_component_ratio": largest_component_ratio,
        "touches_border": touches_border,
        "background_complexity": background_complexity,
        "object_texture": object_texture,
        "centeredness": centeredness,
        "mask": cleaned,
    }


@lru_cache(maxsize=2)
def _load_grounding_dino(cache_dir: str, device: str):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    cleanup_incomplete_model_blobs(GROUNDING_DINO_MODEL_ID, cache_dir)
    processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL_ID, cache_dir=cache_dir)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_MODEL_ID, cache_dir=cache_dir)
    model.to(device)
    model.eval()
    return processor, model


def detect_object_box_with_grounding_dino(image, label, cache_dir: str, device: str = "cuda", box_threshold: float = 0.25, text_threshold: float = 0.25):
    import torch

    image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
    processor, model = _load_grounding_dino(cache_dir, device)
    query = str(label).strip()
    if not query:
        raise ValueError("label is required")
    if not query.endswith('.'):
        query = f"{query}."
    inputs = processor(images=image, text=query, return_tensors='pt')
    inputs = {key: value.to(device) if hasattr(value, 'to') else value for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs['input_ids'],
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]
    if len(result['boxes']) == 0:
        return None
    best_idx = int(result['scores'].argmax().item())
    return {
        'box': result['boxes'][best_idx].detach().cpu().numpy().astype(np.float32),
        'score': float(result['scores'][best_idx].item()),
        'label': result['labels'][best_idx],
    }


def segment_box_with_sam2(image, box, image_predictor):
    image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
    image_arr = np.asarray(image, dtype=np.uint8)
    image_predictor.set_image(image_arr)
    masks, scores, _ = image_predictor.predict(box=np.asarray(box, dtype=np.float32)[None, :], multimask_output=True)
    best_idx = int(np.argmax(scores))
    return clean_binary_mask(masks[best_idx])


@lru_cache(maxsize=1)
def _load_birefnet_pipeline(cache_dir: str, device: str):
    from transformers import pipeline

    cleanup_incomplete_model_blobs(BIREFNET_MODEL_ID, cache_dir)
    device_id = 0 if str(device).startswith('cuda') else -1
    return pipeline('image-segmentation', model=BIREFNET_MODEL_ID, trust_remote_code=True, device=device_id, cache_dir=cache_dir)


def fallback_segment_with_birefnet(image, cache_dir: str, device: str = "cuda"):
    image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
    segmenter = _load_birefnet_pipeline(cache_dir, device)
    outputs = segmenter(image)
    if not outputs:
        raise RuntimeError('BiRefNet returned no segmentation outputs')
    best_mask = None
    best_area = -1
    for item in outputs:
        mask = item.get('mask')
        if mask is None:
            continue
        cleaned = clean_binary_mask(mask)
        area = int((cleaned > 0).sum())
        if area > best_area:
            best_area = area
            best_mask = cleaned
    if best_mask is None:
        raise RuntimeError('BiRefNet did not yield a usable mask')
    return best_mask


def select_best_reference_candidate(candidates):
    if not candidates:
        raise ValueError('candidates must not be empty')
    return max(candidates, key=lambda item: item['score'])
