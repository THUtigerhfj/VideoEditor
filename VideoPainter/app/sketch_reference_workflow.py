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
    best["image"].save(reference_image_path)
    Image.fromarray(best["mask"]).convert("L").save(reference_mask_path)

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
