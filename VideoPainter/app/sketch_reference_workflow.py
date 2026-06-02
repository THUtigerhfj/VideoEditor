from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from frame_conditioning import mask_geometry
from reference_segmenter import (
    detect_object_box_with_grounding_dino,
    fallback_segment_with_birefnet,
    score_reference_candidate,
    segment_box_with_sam2,
    select_best_reference_candidate,
)
from sketch_generator import (
    build_sdxl_scribble_pipeline,
    generate_reference_candidates,
)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def _ensure_rgb_image(image_or_path) -> Image.Image:
    if isinstance(image_or_path, Image.Image):
        return image_or_path.convert("RGB")
    return Image.open(image_or_path).convert("RGB")


def _seed_list(seed: int, candidate_count: int) -> list[int]:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if seed >= 0:
        return [int(seed) + idx for idx in range(candidate_count)]
    rng = np.random.default_rng()
    return [int(rng.integers(0, 2**31 - 1)) for _ in range(candidate_count)]


@lru_cache(maxsize=2)
def build_sam2_image_predictor(sam2_cfg: str, sam2_ckpt: str, device: str = "cuda"):
    sam2_model = build_sam2(sam2_cfg, str(sam2_ckpt), device=device, apply_postprocessing=False)
    return SAM2ImagePredictor(sam2_model)


def build_reference_preview(image, mask):
    image_arr = np.asarray(Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB"), dtype=np.uint8)
    if mask is None:
        return image_arr
    mask_arr = np.asarray(mask, dtype=np.uint8)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[..., 0]
    mask_binary = (mask_arr > 0).astype(np.uint8)
    color = np.zeros_like(image_arr)
    color[:, :, 1] = 255
    return np.uint8((1 - 0.45 * mask_binary[:, :, None]) * image_arr + 0.45 * mask_binary[:, :, None] * color)




def _mask_bbox(mask_arr: np.ndarray):
    binary = mask_arr > 0
    ys, xs = np.where(binary)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def fit_reference_to_frame_mask(reference_image: Image.Image, reference_mask, target_mask, output_size=(720, 480), object_scale: float = 1.0):
    image_arr = np.asarray(reference_image.convert("RGB"), dtype=np.uint8)
    ref = np.asarray(reference_mask, dtype=np.uint8)
    if ref.ndim == 3:
        ref = ref[..., 0]
    ref = (ref > 0).astype(np.uint8) * 255
    target = np.asarray(target_mask, dtype=np.uint8)
    if target.ndim == 3:
        target = target[..., 0]
    target = (target > 0).astype(np.uint8) * 255

    ref_bbox = _mask_bbox(ref)
    target_bbox = _mask_bbox(target)
    if ref_bbox is None or target_bbox is None:
        raise ValueError("reference_mask and target_mask must be non-empty")

    rx0, ry0, rx1, ry1 = ref_bbox
    tx0, ty0, tx1, ty1 = target_bbox
    object_crop = image_arr[ry0 : ry1 + 1, rx0 : rx1 + 1]
    alpha_crop = ref[ry0 : ry1 + 1, rx0 : rx1 + 1]
    target_w = tx1 - tx0 + 1
    target_h = ty1 - ty0 + 1
    object_scale = min(1.0, max(0.05, float(object_scale)))
    scaled_w = max(1, int(round(target_w * object_scale)))
    scaled_h = max(1, int(round(target_h * object_scale)))
    resized_rgb = np.asarray(Image.fromarray(object_crop).resize((target_w, target_h), Image.BICUBIC), dtype=np.uint8)
    resized_alpha = np.asarray(Image.fromarray(alpha_crop).resize((target_w, target_h), Image.NEAREST), dtype=np.uint8)
    if object_scale >= 1.0:
        resized_alpha = target[ty0 : ty1 + 1, tx0 : tx1 + 1]
    else:
        resized_rgb = np.asarray(Image.fromarray(resized_rgb).resize((scaled_w, scaled_h), Image.BICUBIC), dtype=np.uint8)
        resized_alpha = np.asarray(Image.fromarray(resized_alpha).resize((scaled_w, scaled_h), Image.NEAREST), dtype=np.uint8)

    canvas = np.full((output_size[1], output_size[0], 3), 255, dtype=np.uint8)
    mask_canvas = np.zeros((output_size[1], output_size[0]), dtype=np.uint8)
    x0 = int(round((tx0 + tx1 + 1 - resized_rgb.shape[1]) / 2.0))
    y0 = int(round((ty0 + ty1 + 1 - resized_rgb.shape[0]) / 2.0))
    x0 = max(0, min(output_size[0] - resized_rgb.shape[1], x0))
    y0 = max(0, min(output_size[1] - resized_rgb.shape[0], y0))
    x1 = x0 + resized_rgb.shape[1]
    y1 = y0 + resized_rgb.shape[0]
    alpha = (resized_alpha > 0).astype(np.float32)[:, :, None]
    patch = canvas[y0:y1, x0:x1]
    patch[:] = np.clip(alpha * resized_rgb.astype(np.float32) + (1.0 - alpha) * patch.astype(np.float32), 0, 255).astype(np.uint8)
    mask_canvas[y0:y1, x0:x1] = np.where(resized_alpha > 0, resized_alpha, mask_canvas[y0:y1, x0:x1]).astype(np.uint8)
    return Image.fromarray(canvas).convert("RGB"), Image.fromarray(mask_canvas).convert("L")


def score_shape_compatibility(candidate_mask, target_mask):
    candidate = mask_geometry(candidate_mask)
    target = mask_geometry(target_mask)
    if candidate["bbox"] is None or target["bbox"] is None:
        return 0.0
    aspect_delta = abs(candidate["aspect"] - target["aspect"]) / max(target["aspect"], 1e-6)
    area_delta = abs(candidate["area_ratio"] - target["area_ratio"]) / max(target["area_ratio"], 1e-6)
    aspect_score = max(0.0, 1.0 - min(aspect_delta, 1.0))
    area_score = max(0.0, 1.0 - min(area_delta, 1.0))
    return float(0.7 * aspect_score + 0.3 * area_score)


def load_reference_assets_for_ui(image_path, mask_path):
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    mask = None
    if mask_path:
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    return image, mask


def _segment_candidate(image: Image.Image, label: str, image_predictor, cache_dir: str, device: str, allow_birefnet: bool):
    detection = None
    detection_error = None
    mask = None
    segmentation_source = None

    try:
        detection = detect_object_box_with_grounding_dino(image, label, cache_dir=cache_dir, device=device)
    except Exception as exc:  # pragma: no cover - exercised in integration
        detection_error = str(exc)

    if detection is not None:
        try:
            mask = segment_box_with_sam2(image, detection["box"], image_predictor)
            segmentation_source = "groundingdino_sam2"
        except Exception as exc:  # pragma: no cover - exercised in integration
            detection_error = str(exc)
            mask = None

    if mask is None and allow_birefnet:
        try:
            mask = fallback_segment_with_birefnet(image, cache_dir=cache_dir, device=device)
            segmentation_source = "birefnet"
        except Exception as exc:  # pragma: no cover - exercised in integration
            detection_error = str(exc)
            mask = None

    if mask is None:
        raise RuntimeError(detection_error or "Automatic segmentation failed")

    score = score_reference_candidate(image, mask, detection_score=float(detection["score"]) if detection else 0.0)
    score.update(
        {
            "segmentation_source": segmentation_source,
            "detection_box": None if detection is None else [float(v) for v in detection["box"].tolist()],
            "detection_label": None if detection is None else detection["label"],
            "detection_error": detection_error,
        }
    )
    return score


def generate_reference_assets(
    sketch_image,
    label: str,
    attrs: str | None,
    output_dir,
    cache_dir,
    sam2_cfg: str,
    sam2_ckpt,
    device: str = "cuda",
    candidate_count: int = 2,
    seed: int = 42,
    allow_birefnet: bool = True,
    allow_missing_mask: bool = False,
    target_mask_for_shape=None,
    shape_compatibility_weight: float = 0.25,
    reference_num_inference_steps: int = 40,
    reference_guidance_scale: float = 6.5,
    reference_controlnet_scale: float = 0.7,
    shape_conditioned_scribble: bool = False,
    sketch_mask_fit_strength: float = 0.5,
    mask_contour_weight: float = 0.6,
    frame_shaped_reference: bool = False,
    frame_shaped_reference_object_scale: float = 1.0,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = output_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    source_image = _ensure_rgb_image(sketch_image)
    input_is_sketch = True
    seeds = _seed_list(seed, candidate_count)

    pipe = build_sdxl_scribble_pipeline(str(cache_dir), device=device)
    raw_candidates = generate_reference_candidates(
        pipe=pipe,
        sketch=source_image,
        label=label,
        attrs=attrs,
        seeds=seeds,
        num_candidates=candidate_count,
        num_inference_steps=reference_num_inference_steps,
        guidance_scale=reference_guidance_scale,
        controlnet_conditioning_scale=reference_controlnet_scale,
        target_mask_for_shape=target_mask_for_shape,
        shape_conditioned_scribble=shape_conditioned_scribble,
        sketch_mask_fit_strength=sketch_mask_fit_strength,
        mask_contour_weight=mask_contour_weight,
    )

    image_predictor = build_sam2_image_predictor(sam2_cfg, str(sam2_ckpt), device=device)

    candidate_records = []
    scored_candidates = []
    for idx, candidate in enumerate(raw_candidates):
        image_path = candidates_dir / f"candidate_{idx:02d}_image.png"
        candidate["image"].save(image_path)
        record: dict[str, Any] = {
            "index": idx,
            "seed": candidate.get("seed"),
            "prompt": candidate.get("prompt"),
            "negative_prompt": candidate.get("negative_prompt"),
            "image_path": str(image_path),
            "status": "failed",
        }
        scribble = candidate.get("scribble")
        if scribble is not None:
            scribble_path = candidates_dir / f"candidate_{idx:02d}_scribble.png"
            scribble.save(scribble_path)
            record["scribble_path"] = str(scribble_path)
        try:
            score = _segment_candidate(
                candidate["image"],
                label=label,
                image_predictor=image_predictor,
                cache_dir=str(cache_dir),
                device=device,
                allow_birefnet=allow_birefnet,
            )
            mask_path = candidates_dir / f"candidate_{idx:02d}_mask.png"
            Image.fromarray(score["mask"]).convert("L").save(mask_path)
            base_score = float(score["score"])
            shape_compatibility = None
            shape_weight = 0.0
            if target_mask_for_shape is not None and shape_compatibility_weight > 0:
                shape_compatibility = score_shape_compatibility(score["mask"], target_mask_for_shape)
                shape_weight = float(shape_compatibility_weight)
                score["score"] = base_score + shape_weight * shape_compatibility
            record.update(
                {
                    "status": "ok",
                    "mask_path": str(mask_path),
                    "score": float(score["score"]),
                    "base_score": base_score,
                    "shape_compatibility": None if shape_compatibility is None else float(shape_compatibility),
                    "shape_compatibility_weight": shape_weight,
                    "mask_area_ratio": float(score["mask_area_ratio"]),
                    "largest_component_ratio": float(score["largest_component_ratio"]),
                    "touches_border": bool(score["touches_border"]),
                    "background_complexity": float(score["background_complexity"]),
                    "object_texture": float(score.get("object_texture", 0.0)),
                    "centeredness": float(score["centeredness"]),
                    "segmentation_source": score["segmentation_source"],
                    "detection_box": score["detection_box"],
                    "detection_label": score["detection_label"],
                    "detection_error": score["detection_error"],
                }
            )
            scored_candidates.append(
                {
                    "index": idx,
                    "image": candidate["image"].copy(),
                    "mask": score["mask"],
                    "score": float(score["score"]),
                    "record": record,
                }
            )
        except Exception as exc:  # pragma: no cover - integration fallback path
            record["error"] = str(exc)
        candidate_records.append(record)

    reference_image_path = output_dir / "reference_image.png"
    reference_mask_path = output_dir / "reference_mask.png"
    reference_meta_path = output_dir / "reference_meta.json"

    if not scored_candidates:
        raw_candidates[0]["image"].save(reference_image_path)
        meta = {
            "label": label,
            "attrs": attrs or "",
            "input_is_sketch": bool(input_is_sketch),
            "skipped_generation": not input_is_sketch,
            "seed": int(seed),
            "candidate_count": int(candidate_count),
            "reference_num_inference_steps": int(reference_num_inference_steps),
            "reference_guidance_scale": float(reference_guidance_scale),
            "reference_controlnet_scale": float(reference_controlnet_scale),
            "shape_conditioned_scribble": bool(shape_conditioned_scribble),
            "sketch_mask_fit_strength": float(sketch_mask_fit_strength),
            "mask_contour_weight": float(mask_contour_weight),
            "frame_shaped_reference": bool(frame_shaped_reference),
            "frame_shaped_reference_object_scale": float(frame_shaped_reference_object_scale),
            "cache_dir": str(cache_dir),
            "best_candidate_index": None,
            "reference_image": str(reference_image_path),
            "reference_mask": None,
            "automatic_segmentation_succeeded": False,
            "candidates": candidate_records,
        }
        reference_meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        if not allow_missing_mask:
            raise RuntimeError("No reference candidate produced a usable mask")
        return {
            "reference_image_path": reference_image_path,
            "reference_mask_path": None,
            "reference_meta_path": reference_meta_path,
            "metadata": meta,
            "best_candidate": None,
            "candidate_records": candidate_records,
        }

    best = select_best_reference_candidate(scored_candidates)
    best_image = best["image"]
    best_mask = best["mask"]
    if frame_shaped_reference:
        if target_mask_for_shape is None:
            raise ValueError("frame_shaped_reference requires target_mask_for_shape")
        best_image, best_mask_pil = fit_reference_to_frame_mask(
            best_image,
            best_mask,
            target_mask_for_shape,
            output_size=(720, 480),
            object_scale=frame_shaped_reference_object_scale,
        )
        best_mask = np.asarray(best_mask_pil.convert("L"), dtype=np.uint8)
        best["image"] = best_image.copy()
        best["mask"] = best_mask

    best_image.save(reference_image_path)
    Image.fromarray(best_mask).convert("L").save(reference_mask_path)

    meta = {
        "label": label,
        "attrs": attrs or "",
        "input_is_sketch": bool(input_is_sketch),
        "skipped_generation": not input_is_sketch,
        "seed": int(seed),
        "candidate_count": int(candidate_count),
        "reference_num_inference_steps": int(reference_num_inference_steps),
        "reference_guidance_scale": float(reference_guidance_scale),
        "reference_controlnet_scale": float(reference_controlnet_scale),
        "shape_conditioned_scribble": bool(shape_conditioned_scribble),
        "sketch_mask_fit_strength": float(sketch_mask_fit_strength),
        "mask_contour_weight": float(mask_contour_weight),
        "frame_shaped_reference": bool(frame_shaped_reference),
        "frame_shaped_reference_object_scale": float(frame_shaped_reference_object_scale),
        "cache_dir": str(cache_dir),
        "best_candidate_index": int(best["index"]),
        "reference_image": str(reference_image_path),
        "reference_mask": str(reference_mask_path),
        "automatic_segmentation_succeeded": True,
        "candidates": candidate_records,
    }
    reference_meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    return {
        "reference_image_path": reference_image_path,
        "reference_mask_path": reference_mask_path,
        "reference_meta_path": reference_meta_path,
        "metadata": meta,
        "best_candidate": best,
        "candidate_records": candidate_records,
    }
