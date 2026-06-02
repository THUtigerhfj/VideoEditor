#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
DIFFUSERS = ROOT / "diffusers" / "src"

sys.path.insert(0, str(APP))
sys.path.insert(0, str(DIFFUSERS))
os.chdir(APP)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from anydoor_bridge import replace_first_frame_with_anydoor  # noqa: E402
from frame_conditioning import build_anydoor_background_inpaint_mask, build_edge_refine_masks, build_lama_background_inpaint_mask, build_union_edit_masks, composite_object_guide_frames, finalize_propagation_sequence, mask_bbox, mask_bbox_trajectory, prepare_anydoor_target_image, select_anydoor_target_mask, smooth_mask_bboxes  # noqa: E402
from reference_segmenter import clean_binary_mask, detect_object_box_with_grounding_dino, segment_box_with_sam2  # noqa: E402
from sam2.build_sam import build_sam2, build_sam2_video_predictor  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402
from sketch_reference_workflow import generate_reference_assets  # noqa: E402
import utils as vp_utils  # noqa: E402
from utils import REFERENCE_PREV_CLIP_WEIGHT, build_reference_propagation_prompt, generate_frames, load_model, run_flux_fill_inpaint, run_lama_inpaint, run_lama_video_inpaint  # noqa: E402


def parse_points(value):
    """Parse points from JSON file, JSON string, or 'x,y,label;x,y,label'."""
    if value is None:
        return None, None
    path = Path(value)
    if path.exists():
        data = json.loads(path.read_text())
    else:
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            data = []
            for item in value.split(";"):
                item = item.strip()
                if not item:
                    continue
                x, y, label = item.split(",")
                data.append([float(x), float(y), int(label)])

    if isinstance(data, dict):
        points = data.get("points")
        labels = data.get("labels")
    else:
        points = [[row[0], row[1]] for row in data]
        labels = [row[2] for row in data]
    if not points or not labels or len(points) != len(labels):
        raise ValueError(f"Invalid points: {value}")
    return np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def parse_points_with_frame_size(value, width: int, height: int):
    if value is None:
        return None, None
    if str(value).strip().lower() == "center":
        return np.asarray([[width / 2.0, height / 2.0]], dtype=np.float32), np.asarray([1], dtype=np.int32)
    return parse_points(value)


def ensure_run_dirs(output_dir):
    output_dir = Path(output_dir)
    dirs = {
        "root": output_dir,
        "inputs": output_dir / "inputs",
        "videos": output_dir / "inputs" / "videos",
        "refs": output_dir / "inputs" / "refs",
        "prompts": output_dir / "inputs" / "prompts",
        "masks": output_dir / "masks",
        "overlays": output_dir / "masks" / "overlays",
        "first_frames": output_dir / "first_frames",
        "outputs": output_dir / "outputs",
        "logs": output_dir / "logs",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def load_video_frames(path, frame_count=49, fps_out=8, size=(720, 480)):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    sample_interval = max(1, int(fps / fps_out)) if fps > fps_out else 1
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(frame, size))
        idx += 1
        if len(frames) >= frame_count:
            break
    cap.release()
    if len(frames) != frame_count:
        raise RuntimeError(
            f"Need {frame_count} sampled frames for the restored VideoPainter path, got {len(frames)}. "
            f"Use a longer input video or lower source FPS sampling logic."
        )
    return np.asarray(frames, dtype=np.uint8)


def save_video(frames, path, fps=8):
    arr = np.asarray(frames)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    h, w = arr[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in arr:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return Path(path)


def save_run_inputs(args, dirs, sampled_frames, video_points, video_labels, ref_points=None, ref_labels=None):
    input_video = Path(args.input_video)
    if input_video.exists():
        shutil.copy2(input_video, dirs["videos"] / input_video.name)
    sampled_video_path = save_video(sampled_frames, dirs["videos"] / "sampled_49f_720x480.mp4", fps=8)

    effective_reference_prompt = None
    if args.mode in {"image_reference", "sketch_reference"}:
        effective_reference_prompt = build_reference_propagation_prompt(args.video_caption, args.target_object_caption)

    prompt_payload = {
        "mode": args.mode,
        "video_caption": args.video_caption,
        "target_object_caption": args.target_object_caption,
        "effective_reference_propagation_prompt": effective_reference_prompt,
        "note": (
            "video_caption is passed to VideoPainter as the edited-video description. "
            "For image_reference and sketch_reference modes, AnyDoor creates the first frame, "
            "and VideoPainter uses that first frame plus an optional target_object_caption emphasis "
            "to propagate the replacement."
        ),
    }
    (dirs["prompts"] / "captions.json").write_text(json.dumps(prompt_payload, indent=2, ensure_ascii=False))
    (dirs["prompts"] / "video_caption.txt").write_text(str(args.video_caption) + "\n")
    if args.target_object_caption:
        (dirs["prompts"] / "target_object_caption.txt").write_text(str(args.target_object_caption) + "\n")
    if effective_reference_prompt:
        (dirs["prompts"] / "effective_reference_propagation_prompt.txt").write_text(str(effective_reference_prompt) + "\n")

    point_payload = {
        "video_points_720x480": None if video_points is None else video_points.tolist(),
        "video_labels": None if video_labels is None else video_labels.tolist(),
        "reference_points_original_image": None if ref_points is None else ref_points.tolist(),
        "reference_labels": None if ref_labels is None else ref_labels.tolist(),
    }
    (dirs["inputs"] / "points.json").write_text(json.dumps(point_payload, indent=2, ensure_ascii=False))
    return sampled_video_path


def save_mask_overlay(frames, masks, path):
    overlay = np.uint8((1 - 0.45 * masks) * frames + 0.45 * masks * np.array([0, 255, 255], dtype=np.uint8))
    return save_video(overlay, path)


def segment_video(frames, points, labels, sam2_cfg, sam2_ckpt, dilate_iter=6):
    predictor = build_sam2_video_predictor(sam2_cfg, str(sam2_ckpt), apply_postprocessing=False)
    state = predictor.init_state(images=frames, offload_video_to_cpu=True, async_loading_frames=True)
    predictor.add_new_points(
        inference_state=state,
        frame_idx=0,
        obj_id=0,
        points=points,
        labels=labels,
    )
    masks = []
    for _, _, logits in predictor.propagate_in_video(state):
        mask = np.zeros((480, 720, 1), dtype=np.float32)
        for logit in logits:
            mask += (logit.cpu().squeeze().detach().numpy() > 0).astype(np.float32)[:, :, None]
        mask = (mask > 0.5).astype(np.uint8)
        if dilate_iter > 0:
            mask = scipy.ndimage.binary_dilation(mask, iterations=dilate_iter).astype(np.uint8)
        masks.append(mask)
    masks = np.asarray(masks, dtype=np.uint8)
    if masks.max() == 0:
        raise RuntimeError("Video target mask is empty")
    return masks


def _mask_bbox(mask):
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim == 3:
        binary = binary[..., 0]
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def fit_reference_mask_to_target_bbox(reference_mask, target_mask, output_size=(720, 480)):
    ref = clean_binary_mask(reference_mask)
    target = (np.asarray(target_mask) > 0).astype(np.uint8)
    if target.ndim == 3:
        target = target[..., 0]
    ref_bbox = _mask_bbox(ref)
    target_bbox = _mask_bbox(target)
    if ref_bbox is None or target_bbox is None:
        raise RuntimeError("Cannot fit reference mask because a source mask is empty")

    rx0, ry0, rx1, ry1 = ref_bbox
    tx0, ty0, tx1, ty1 = target_bbox
    ref_crop = (ref[ry0 : ry1 + 1, rx0 : rx1 + 1] > 0).astype(np.uint8) * 255
    tw = tx1 - tx0 + 1
    th = ty1 - ty0 + 1
    scale = min(tw / max(ref_crop.shape[1], 1), th / max(ref_crop.shape[0], 1))
    new_w = max(1, int(round(ref_crop.shape[1] * scale)))
    new_h = max(1, int(round(ref_crop.shape[0] * scale)))
    resized = cv2.resize(ref_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((output_size[1], output_size[0]), dtype=np.uint8)
    cx = int(round((tx0 + tx1) / 2.0))
    cy = int(round((ty0 + ty1) / 2.0))
    x0 = max(0, min(output_size[0] - new_w, cx - new_w // 2))
    y0 = max(0, min(output_size[1] - new_h, cy - new_h // 2))
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return clean_binary_mask(canvas)


def segment_anydoor_object_mask(image, label, cache_dir, sam2_cfg, sam2_ckpt, device="cuda:0"):
    label = str(label or "").strip()
    if not label:
        raise ValueError("A label or target_object_caption is required for AnyDoor object mask segmentation")
    image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
    detection = detect_object_box_with_grounding_dino(image, label, cache_dir=cache_dir, device=device)
    if detection is None:
        raise RuntimeError(f"Could not detect replacement object in AnyDoor first frame: {label}")
    predictor = SAM2ImagePredictor(build_sam2(sam2_cfg, str(sam2_ckpt), device=device, apply_postprocessing=False))
    return segment_box_with_sam2(image, detection["box"], predictor)


def segment_video_from_initial_mask(frames, initial_mask, sam2_cfg, sam2_ckpt, dilate_iter=6):
    predictor = build_sam2_video_predictor(sam2_cfg, str(sam2_ckpt), apply_postprocessing=False)
    state = predictor.init_state(images=frames, offload_video_to_cpu=True, async_loading_frames=True)
    init = (np.asarray(initial_mask) > 0).astype(np.uint8)
    if init.ndim == 3:
        init = init[..., 0]
    if init.max() == 0:
        raise RuntimeError("Initial propagation mask is empty")
    predictor.add_new_mask(
        inference_state=state,
        frame_idx=0,
        obj_id=0,
        mask=init,
    )
    masks = []
    for _, _, logits in predictor.propagate_in_video(state):
        mask = np.zeros((480, 720, 1), dtype=np.float32)
        for logit in logits:
            mask += (logit.cpu().squeeze().detach().numpy() > 0).astype(np.float32)[:, :, None]
        mask = (mask > 0.5).astype(np.uint8)
        if dilate_iter > 0:
            mask = scipy.ndimage.binary_dilation(mask, iterations=dilate_iter).astype(np.uint8)
        masks.append(mask)
    masks = np.asarray(masks, dtype=np.uint8)
    if masks.max() == 0:
        raise RuntimeError("Propagation mask tracking returned an empty mask sequence")
    return masks


def build_target_bboxes_for_guides(target_masks, args, output_size=(720, 480)):
    raw_bboxes = [mask_bbox(mask) for mask in target_masks]
    smoothing = getattr(args, "mask_bbox_smoothing", "off")
    if smoothing == "median":
        return smooth_mask_bboxes(
            raw_bboxes,
            image_size=output_size,
            window=getattr(args, "mask_bbox_smoothing_window", 5),
            max_scale_delta=getattr(args, "mask_bbox_max_scale_delta", 0.08),
        )
    return raw_bboxes


def save_mask_bbox_diagnostics(path, target_masks, target_bboxes):
    payload = {
        "raw": mask_bbox_trajectory(target_masks),
        "used_bboxes": [None if box is None else [int(v) for v in box] for box in target_bboxes],
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def fit_object_mask_to_video_motion(object_mask, video_masks):
    fitted = []
    for video_mask in video_masks:
        fitted_mask = fit_reference_mask_to_target_bbox(object_mask, video_mask.squeeze(), output_size=(720, 480))
        fitted.append((fitted_mask > 0).astype(np.uint8)[:, :, None])
    masks = np.asarray(fitted, dtype=np.uint8)
    if masks.max() == 0:
        raise RuntimeError("Object-shaped propagation mask sequence is empty")
    return masks


def segment_reference_image(reference_image, points, labels, sam2_cfg, sam2_ckpt):
    image = Image.open(reference_image).convert("RGB")
    arr = np.asarray(image, dtype=np.uint8)
    predictor = SAM2ImagePredictor(build_sam2(sam2_cfg, str(sam2_ckpt), apply_postprocessing=False))
    predictor.set_image(arr)
    masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels, multimask_output=True)
    mask = (masks[int(np.argmax(scores))] > 0).astype(np.uint8)
    if mask.max() == 0:
        raise RuntimeError("Reference image mask is empty")
    return image, Image.fromarray(mask * 255).convert("L"), mask


def load_reference_image_and_mask(args, dirs):
    if args.reference_mask:
        reference_image = Image.open(args.reference_image).convert("RGB")
        reference_mask = Image.open(args.reference_mask).convert("L")
        reference_mask_np = (np.asarray(reference_mask, dtype=np.uint8) > 0).astype(np.uint8)
        if reference_mask_np.max() == 0:
            raise RuntimeError("Reference image mask is empty")
        return reference_image, reference_mask, reference_mask_np

    ref_points, ref_labels = parse_points(args.reference_points)
    save_run_inputs(args, dirs, load_video_frames(args.input_video), *parse_points_with_frame_size(args.video_points, 720, 480), ref_points, ref_labels)
    return segment_reference_image(
        args.reference_image,
        ref_points,
        ref_labels,
        args.sam2_cfg,
        args.sam2_ckpt,
    )


def pil_frames_and_masks(frames, masks):
    pil_frames = [Image.fromarray(frame).convert("RGB") for frame in frames]
    pil_masks = []
    for mask in masks:
        mask_gray = (mask.squeeze() > 0).astype(np.uint8) * 255
        pil_masks.append(Image.fromarray(np.stack([mask_gray, mask_gray, mask_gray], axis=-1)).convert("RGB"))
    return pil_frames, pil_masks


def run_reference_replacement(args, dirs, frames, masks, pil_frames, pil_masks, reference_image, reference_mask, metadata, output_prefix):
    reference_image.save(dirs["refs"] / "reference_image.png")
    reference_mask.save(dirs["masks"] / "reference_object_mask.png")

    effective_prompt = build_reference_propagation_prompt(args.video_caption, args.target_object_caption)
    metadata["effective_reference_propagation_prompt"] = effective_prompt

    target_first = Image.fromarray(frames[0]).convert("RGB")
    target_mask = Image.fromarray((masks[0].squeeze() * 255).astype(np.uint8)).convert("L")
    target_mask_arr = np.asarray(target_mask, dtype=np.uint8)
    reference_mask_arr = np.asarray(reference_mask.convert("L"), dtype=np.uint8)
    anydoor_target_mask_arr = select_anydoor_target_mask(reference_mask_arr, target_mask_arr, output_size=(720, 480))
    pre_inpaint_mode = getattr(args, "anydoor_pre_inpaint_mode", "lama")
    metadata["anydoor_pre_inpaint_mode"] = pre_inpaint_mode
    if pre_inpaint_mode == "flux":
        background_mask_arr = build_anydoor_background_inpaint_mask(
            target_mask_arr,
            dilate_size=getattr(args, "anydoor_background_mask_dilate", 0),
        )
        Image.fromarray(background_mask_arr).convert("L").save(dirs["masks"] / "flux_background_mask.png")
        background_prompt = getattr(args, "anydoor_background_prompt", "") or (
            "unoccupied tabletop and laptop keyboard area, continuous desk and laptop surfaces matching the surrounding scene, same lighting, perspective, reflections, focus, and background texture"
        )
        metadata["anydoor_background_prompt"] = background_prompt
        flux_background = run_flux_fill_inpaint(
            image=target_first,
            mask_image=Image.fromarray(background_mask_arr).convert("L"),
            pipe_img_inpainting=args.img_inpainting_model,
            prompt=background_prompt,
            seed=args.seed,
            guidance_scale=getattr(args, "anydoor_background_guidance_scale", 30.0),
            num_inference_steps=getattr(args, "anydoor_background_num_inference_steps", 50),
            device=args.device,
        )
        flux_background.save(dirs["first_frames"] / "flux_background_first_frame.png")
        anydoor_target_image_arr = np.asarray(flux_background.convert("RGB"), dtype=np.uint8)
    elif pre_inpaint_mode == "lama":
        lama_mask_arr = build_lama_background_inpaint_mask(
            target_mask_arr,
            mode=getattr(args, "anydoor_lama_mask_mode", "rect"),
            padding=getattr(args, "anydoor_lama_mask_padding", 24),
            dilate_size=getattr(args, "anydoor_lama_mask_dilate", 31),
        )
        Image.fromarray(lama_mask_arr).convert("L").save(dirs["masks"] / "lama_background_mask.png")
        metadata["anydoor_lama_mask_mode"] = getattr(args, "anydoor_lama_mask_mode", "rect")
        metadata["anydoor_lama_mask_padding"] = getattr(args, "anydoor_lama_mask_padding", 24)
        metadata["anydoor_lama_mask_dilate"] = getattr(args, "anydoor_lama_mask_dilate", 31)
        lama_background = run_lama_inpaint(
            image=target_first,
            mask_image=Image.fromarray(lama_mask_arr).convert("L"),
            lama_model=getattr(args, "lama_model", None),
            device=args.device,
        )
        lama_background.save(dirs["first_frames"] / "lama_background_first_frame.png")
        anydoor_target_image_arr = np.asarray(lama_background.convert("RGB"), dtype=np.uint8)
    elif pre_inpaint_mode == "off":
        anydoor_target_image_arr = frames[0].copy()
    elif pre_inpaint_mode == "opencv":
        anydoor_target_image_arr = prepare_anydoor_target_image(frames[0], target_mask_arr, anydoor_target_mask_arr)
    else:
        raise ValueError(f"Unsupported anydoor_pre_inpaint_mode: {pre_inpaint_mode}")
    Image.fromarray(anydoor_target_image_arr).convert("RGB").save(dirs["first_frames"] / "anydoor_target_input.png")
    Image.fromarray(anydoor_target_mask_arr).convert("L").save(dirs["masks"] / "anydoor_target_mask.png")
    target_mask.save(dirs["masks"] / "cogvideox_video_target_first_mask.png")
    replaced_first = replace_first_frame_with_anydoor(
        target_image=Image.fromarray(anydoor_target_image_arr).convert("RGB"),
        target_mask=Image.fromarray(anydoor_target_mask_arr).convert("L"),
        reference_image=reference_image,
        reference_mask=reference_mask,
        guidance_scale=args.anydoor_guidance_scale,
    )
    anydoor_path = dirs["first_frames"] / "anydoor_first_frame.png"
    replaced_first.save(anydoor_path)
    torch.cuda.empty_cache()

    propagation_masks = masks
    propagation_mask_source = "video_target"
    guide_object_mask = anydoor_target_mask_arr
    if args.reference_propagation_mask_source in {"anydoor_object", "anydoor_object_sam2"}:
        tracking_frames = frames.copy()
        tracking_frames[0] = np.asarray(replaced_first.resize((720, 480)).convert("RGB"), dtype=np.uint8)
        object_label = args.target_object_caption or args.label or "object"
        try:
            first_object_mask = segment_anydoor_object_mask(
                replaced_first.resize((720, 480)),
                object_label,
                cache_dir=args.cache_dir,
                sam2_cfg=args.sam2_cfg,
                sam2_ckpt=args.sam2_ckpt,
                device=args.device,
            )
            propagation_mask_source = "anydoor_object"
        except Exception as exc:
            print(f"AnyDoor object mask segmentation failed; using fitted reference mask fallback: {exc}")
            first_object_mask = fit_reference_mask_to_target_bbox(reference_mask, target_mask, output_size=(720, 480))
            propagation_mask_source = "fitted_reference_mask"

        first_object_mask_path = dirs["masks"] / "anydoor_object_first_mask.png"
        Image.fromarray((np.asarray(first_object_mask) > 0).astype(np.uint8) * 255).convert("L").save(first_object_mask_path)
        guide_object_mask = (np.asarray(first_object_mask) > 0).astype(np.uint8) * 255
        if args.reference_propagation_mask_source == "anydoor_object_sam2":
            propagation_masks = segment_video_from_initial_mask(
                tracking_frames,
                first_object_mask,
                args.sam2_cfg,
                args.sam2_ckpt,
                dilate_iter=args.sam2_dilate_iter,
            )
            propagation_mask_source = f"{propagation_mask_source}_sam2"
        else:
            propagation_masks = fit_object_mask_to_video_motion(first_object_mask, masks)
            propagation_mask_source = f"{propagation_mask_source}_fit_to_video_motion"
        Image.fromarray((propagation_masks[0].squeeze() * 255).astype(np.uint8)).save(dirs["masks"] / "propagation_first_mask.png")
        save_mask_overlay(frames, propagation_masks, dirs["overlays"] / "propagation_mask_overlay.mp4")

    metadata["reference_propagation_mask_source"] = propagation_mask_source
    reference_motion_guide = getattr(args, "reference_motion_guide", "none")
    metadata["reference_motion_guide"] = reference_motion_guide
    metadata["guide_edge_inner_erode"] = getattr(args, "guide_edge_inner_erode", 4)
    metadata["guide_edge_outer_dilate"] = getattr(args, "guide_edge_outer_dilate", 8)
    metadata["mask_bbox_smoothing"] = getattr(args, "mask_bbox_smoothing", "off")
    metadata["mask_bbox_smoothing_window"] = getattr(args, "mask_bbox_smoothing_window", 5)
    metadata["mask_bbox_max_scale_delta"] = getattr(args, "mask_bbox_max_scale_delta", 0.08)
    metadata["edit_mask_mode"] = getattr(args, "edit_mask_mode", "propagation")
    metadata["effective_edit_mask_mode"] = metadata["edit_mask_mode"]
    pil_frames, pil_masks = pil_frames_and_masks(frames, propagation_masks)

    guide_images = None
    guide_mode = "none"
    original_target_mask_arrays = [((mask.squeeze() > 0).astype(np.uint8) * 255) for mask in masks]
    propagation_mask_arrays = [((mask.squeeze() > 0).astype(np.uint8) * 255) for mask in propagation_masks]
    if metadata["edit_mask_mode"] == "union_target_object":
        generation_masks = build_union_edit_masks(original_target_mask_arrays, propagation_mask_arrays)
        generation_mask_arrays = [np.asarray(mask.convert("L"), dtype=np.uint8) for mask in generation_masks]
        Image.fromarray(generation_mask_arrays[0]).save(dirs["masks"] / "edit_first_mask.png")
        edit_overlay_masks = np.asarray([(arr > 0).astype(np.uint8)[:, :, None] for arr in generation_mask_arrays])
        save_mask_overlay(frames, edit_overlay_masks, dirs["overlays"] / "edit_mask_overlay.mp4")
    else:
        generation_masks = pil_masks
        generation_mask_arrays = propagation_mask_arrays
        Image.fromarray(generation_mask_arrays[0]).save(dirs["masks"] / "edit_first_mask.png")
    target_bboxes = build_target_bboxes_for_guides(propagation_mask_arrays, args, output_size=(720, 480))
    if getattr(args, "save_mask_bbox_stats", False):
        save_mask_bbox_diagnostics(dirs["root"] / "mask_bbox_stats.json", propagation_mask_arrays, target_bboxes)
    if reference_motion_guide in {"full_region", "edge_refine"}:
        guide_images = composite_object_guide_frames(
            pil_frames,
            replaced_first.resize((720, 480)),
            guide_object_mask,
            propagation_mask_arrays,
            target_bboxes=target_bboxes,
        )
        guide_mode = reference_motion_guide
        guide_dir = dirs["root"] / "guides"
        guide_dir.mkdir(parents=True, exist_ok=True)
        guide_images[0].save(guide_dir / "guide_frame_000.png")
        guide_images[min(1, len(guide_images) - 1)].save(guide_dir / "guide_frame_001.png")
        if reference_motion_guide == "edge_refine":
            generation_masks = build_edge_refine_masks(
                propagation_mask_arrays,
                inner_erode_iter=getattr(args, "guide_edge_inner_erode", 4),
                outer_dilate_iter=getattr(args, "guide_edge_outer_dilate", 8),
            )
            generation_masks[0].save(dirs["masks"] / "guide_edge_first_mask.png")
            metadata["effective_edit_mask_mode"] = "edge_refine"

    conditioning_mode = getattr(args, "conditioning_video_mode", "full_video")
    metadata["conditioning_video_mode"] = conditioning_mode
    conditioning_frames = [img.copy() for img in pil_frames]
    effective_conditioning_mode = conditioning_mode
    if conditioning_mode == "lama_cleaned_video":
        metadata["conditioning_video_lama_mask_mode"] = getattr(args, "conditioning_lama_mask_mode", getattr(args, "anydoor_lama_mask_mode", "rect"))
        metadata["conditioning_video_lama_mask_padding"] = getattr(args, "conditioning_lama_mask_padding", 48)
        metadata["conditioning_video_lama_mask_dilate"] = getattr(args, "conditioning_lama_mask_dilate", getattr(args, "anydoor_lama_mask_dilate", 31))
        conditioning_frames = run_lama_video_inpaint(
            images=conditioning_frames,
            masks=[mask.copy() for mask in pil_masks],
            lama_model=getattr(args, "lama_model", None),
            mask_mode=metadata["conditioning_video_lama_mask_mode"],
            mask_padding=metadata["conditioning_video_lama_mask_padding"],
            mask_dilate=metadata["conditioning_video_lama_mask_dilate"],
            device=args.device,
            output_dir=dirs["first_frames"],
        )
        effective_conditioning_mode = "full_video"
        (dirs["first_frames"] / "lama_conditioning_video_mode.txt").write_text("lama_cleaned_video\n")

    pipe, _ = load_model(
        model_path=args.model_path,
        inpainting_branch=args.inpainting_branch,
        img_inpainting_model="",
        id_adapter=args.id_adapter,
        device=args.device,
    )
    propagated = generate_frames(
        images=[img.copy() for img in conditioning_frames],
        masks=[mask.copy() for mask in generation_masks],
        pipe=pipe,
        pipe_img_inpainting=None,
        prompt=effective_prompt,
        image_inpainting_prompt="",
        seed=args.seed,
        cfg_scale=args.cfg_scale,
        dilate_size=args.dilate_size,
        first_frame_override=replaced_first,
        return_full_sequence=True,
        prev_clip_weight=args.reference_prev_clip_weight,
        id_pool_resample_learnable=True,
        guide_images=[img.copy() for img in guide_images] if guide_images is not None else None,
        guide_mode=guide_mode,
        guide_dilate_size=getattr(args, "guide_dilate_size", None),
        conditioning_video_mode=effective_conditioning_mode,
    )
    propagated_uint8 = finalize_propagation_sequence(propagated, fallback_first_frame=replaced_first.resize((720, 480)))
    propagated_first = Image.fromarray(propagated_uint8[0]).convert("RGB")
    propagated_first_path = dirs["first_frames"] / "propagated_first_frame.png"
    propagated_first.save(propagated_first_path)
    raw_path = save_video(propagated_uint8[1:], dirs["outputs"] / f"{output_prefix}_videopainter_48f.mp4")
    final_path = save_video(propagated_uint8, dirs["outputs"] / f"{output_prefix}_videopainter_49f_with_first.mp4")
    metadata["outputs"].update({
        "first_frame": str(propagated_first_path),
        "anydoor_first_frame": str(anydoor_path),
        "raw_48f": str(raw_path),
        "final_49f": str(final_path),
    })


def run(args):
    if args.dilate_size is None:
        args.dilate_size = 0 if args.mode in {"image_reference", "sketch_reference"} else 16

    dirs = ensure_run_dirs(args.output_dir)
    os.environ["GRADIO_TEMP_DIR"] = str(dirs["root"] / "tmp_gradio")
    vp_utils.GRADIO_TEMP_DIR = os.environ["GRADIO_TEMP_DIR"]
    (dirs["root"] / "tmp_gradio" / "inpaint").mkdir(parents=True, exist_ok=True)
    (dirs["root"] / "tmp_gradio" / "track").mkdir(parents=True, exist_ok=True)

    frames = load_video_frames(args.input_video)
    video_points, video_labels = parse_points_with_frame_size(args.video_points, width=frames.shape[2], height=frames.shape[1])
    save_run_inputs(args, dirs, frames, video_points, video_labels)
    Image.fromarray(frames[0]).save(dirs["first_frames"] / "source_first_frame.png")

    masks = segment_video(
        frames=frames,
        points=video_points,
        labels=video_labels,
        sam2_cfg=args.sam2_cfg,
        sam2_ckpt=args.sam2_ckpt,
        dilate_iter=args.sam2_dilate_iter,
    )
    Image.fromarray((masks[0].squeeze() * 255).astype(np.uint8)).save(dirs["masks"] / "video_target_first_mask.png")
    save_mask_overlay(frames, masks, dirs["overlays"] / "video_target_mask_overlay.mp4")

    pil_frames, pil_masks = pil_frames_and_masks(frames, masks)
    metadata = vars(args).copy()
    metadata["outputs"] = {}

    if args.mode == "text_prompt":
        pipe, flux_path = load_model(
            model_path=args.model_path,
            inpainting_branch=args.inpainting_branch,
            img_inpainting_model=args.img_inpainting_model,
            id_adapter=args.id_adapter,
            device=args.device,
        )
        outputs = generate_frames(
            images=[img.copy() for img in pil_frames],
            masks=[mask.copy() for mask in pil_masks],
            pipe=pipe,
            pipe_img_inpainting=flux_path,
            prompt=args.video_caption,
            image_inpainting_prompt=args.target_object_caption,
            seed=args.seed,
            cfg_scale=args.cfg_scale,
            dilate_size=args.dilate_size,
        )
        first_frame = Image.open(dirs["root"] / "tmp_gradio" / "inpaint" / "first_frame_inpainted.png").convert("RGB").resize((720, 480))
        first_frame_path = dirs["first_frames"] / "flux_first_frame.png"
        first_frame.save(first_frame_path)
        raw_path = save_video(outputs, dirs["outputs"] / "text_prompt_videopainter_48f.mp4")
        combined = np.concatenate([np.asarray(first_frame, dtype=np.uint8)[None, ...], np.clip(outputs * 255, 0, 255).astype(np.uint8)], axis=0)
        final_path = save_video(combined, dirs["outputs"] / "text_prompt_videopainter_49f_with_first.mp4")
        metadata["outputs"].update({"first_frame": str(first_frame_path), "raw_48f": str(raw_path), "final_49f": str(final_path)})
    else:
        if args.mode == "sketch_reference":
            target_shape_mask = (masks[0].squeeze() > 0).astype(np.uint8) * 255
            sketch_result = generate_reference_assets(
                sketch_image=args.sketch_image,
                label=args.label,
                attrs=args.attrs,
                output_dir=dirs["root"],
                cache_dir=args.cache_dir,
                sam2_cfg=args.sam2_cfg,
                sam2_ckpt=args.sam2_ckpt,
                device=args.device,
                candidate_count=args.candidate_count,
                seed=args.seed,
                target_mask_for_shape=target_shape_mask,
                shape_compatibility_weight=args.shape_compatibility_weight,
                reference_num_inference_steps=args.reference_num_inference_steps,
                reference_guidance_scale=args.reference_guidance_scale,
                reference_controlnet_scale=args.reference_controlnet_scale,
                shape_conditioned_scribble=args.shape_conditioned_scribble,
                sketch_mask_fit_strength=args.sketch_mask_fit_strength,
                mask_contour_weight=args.mask_contour_weight,
                frame_shaped_reference=args.frame_shaped_reference,
                frame_shaped_reference_object_scale=args.frame_shaped_reference_object_scale,
            )
            metadata["reference_meta"] = sketch_result["metadata"]
            reference_image = Image.open(sketch_result["reference_image_path"]).convert("RGB")
            reference_mask = Image.open(sketch_result["reference_mask_path"]).convert("L")
            shutil.copy2(sketch_result["reference_meta_path"], dirs["logs"] / "reference_meta.json")
            output_prefix = "image_reference"
        elif args.mode == "image_reference":
            reference_image, reference_mask, _ = load_reference_image_and_mask(args, dirs)
            reference_src = Path(args.reference_image)
            if reference_src.exists():
                shutil.copy2(reference_src, dirs["refs"] / reference_src.name)
            output_prefix = "image_reference"
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")

        run_reference_replacement(args, dirs, frames, masks, pil_frames, pil_masks, reference_image, reference_mask, metadata, output_prefix)

    metadata_path = dirs["root"] / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(json.dumps(metadata["outputs"], indent=2, ensure_ascii=False))
    print(f"METADATA={metadata_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Run VideoPainter replacement without Gradio.")
    parser.add_argument("--mode", choices=["text_prompt", "image_reference", "sketch_reference"], required=True)
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", default=str(ROOT / "runs" / "manual_run"))
    parser.add_argument("--video_points", required=True, help="JSON file/string, 'x,y,label;x,y,label', or 'center'. Labels: 1 positive, 0 negative.")
    parser.add_argument("--reference_image")
    parser.add_argument("--reference_points", help="Optional for image_reference when --reference_mask is not supplied.")
    parser.add_argument("--reference_mask")
    parser.add_argument("--sketch_image")
    parser.add_argument("--label")
    parser.add_argument("--attrs", default="")
    parser.add_argument("--candidate_count", type=int, default=2)
    parser.add_argument("--reference_num_inference_steps", type=int, default=40, help="SDXL ControlNet steps for sketch-to-reference generation.")
    parser.add_argument("--reference_guidance_scale", type=float, default=6.5, help="SDXL text guidance scale for sketch-to-reference generation.")
    parser.add_argument("--reference_controlnet_scale", type=float, default=0.7, help="ControlNet scribble conditioning scale; lower values such as 0.45-0.6 can reduce line-art artifacts.")
    parser.add_argument("--shape_compatibility_weight", type=float, default=0.25, help="Extra sketch candidate score weight for matching the clicked video target mask shape. Use 0 to disable.")
    parser.add_argument("--shape_conditioned_scribble", action="store_true", help="Add the video frame0 target mask contour to the sketch ControlNet input.")
    parser.add_argument("--sketch_mask_fit_strength", type=float, default=0.5, help="Reserved strength for fitting sketch control to frame0 mask shape.")
    parser.add_argument("--mask_contour_weight", type=float, default=0.6, help="Weight/enable value for frame0 mask contour in shape-conditioned scribble.")
    parser.add_argument("--frame_shaped_reference", action="store_true", help="Fit selected reference object into the original 720x480 frame0 target mask bbox.")
    parser.add_argument("--frame_shaped_reference_object_scale", type=float, default=0.82, help="Scale for the fitted sketch reference object inside the frame0 target bbox. The resulting smaller object mask is used by AnyDoor; CogVideoX can still use the original video target masks.")
    parser.add_argument(
        "--video_caption",
        required=True,
        help=(
            "Edited-video caption passed to VideoPainter. Prefer a description of the final video "
            "instead of an instruction such as 'replace X with Y'."
        ),
    )
    parser.add_argument("--target_object_caption", default="", help="Required for text_prompt; for image_reference/sketch_reference it is appended to the propagation prompt when provided.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_scale", type=float, default=6.0)
    parser.add_argument("--dilate_size", type=int, default=None)
    parser.add_argument("--sam2_dilate_iter", type=int, default=6)
    parser.add_argument("--anydoor_guidance_scale", type=float, default=5.0)
    parser.add_argument("--anydoor_pre_inpaint_mode", choices=["off", "opencv", "flux", "lama"], default="lama", help="How to remove the original target before AnyDoor. lama uses prompt-free LaMa object removal; flux uses FLUX.1 Fill.")
    parser.add_argument("--anydoor_background_prompt", default="unoccupied tabletop and laptop keyboard area, continuous desk and laptop surfaces matching the surrounding scene, same lighting, perspective, reflections, focus, and background texture", help="Prompt used when --anydoor_pre_inpaint_mode flux.")
    parser.add_argument("--anydoor_background_mask_dilate", type=int, default=0, help="Optional dilation kernel size before building the rectangular FLUX background inpaint mask.")
    parser.add_argument("--anydoor_background_guidance_scale", type=float, default=30.0, help="FLUX Fill guidance scale for AnyDoor background pre-inpainting.")
    parser.add_argument("--anydoor_background_num_inference_steps", type=int, default=50, help="FLUX Fill inference steps for AnyDoor background pre-inpainting.")
    parser.add_argument("--lama_model", default=str(ROOT / "ckpt" / "lama" / "big-lama.pt"), help="Local LaMa torchscript model used when --anydoor_pre_inpaint_mode lama.")
    parser.add_argument("--anydoor_lama_mask_mode", choices=["rect", "dilate", "exact"], default="rect", help="Mask shape for LaMa background cleanup before AnyDoor.")
    parser.add_argument("--anydoor_lama_mask_padding", type=int, default=24, help="Padding in pixels for rect LaMa cleanup mask.")
    parser.add_argument("--anydoor_lama_mask_dilate", type=int, default=31, help="Dilation kernel size for dilate LaMa cleanup mask.")
    parser.add_argument("--reference_prev_clip_weight", type=float, default=REFERENCE_PREV_CLIP_WEIGHT, help="Identity-preserving previous-clip weight for image_reference and sketch_reference propagation.")
    parser.add_argument("--reference_propagation_mask_source", choices=["anydoor_object", "anydoor_object_sam2", "video_target"], default="video_target", help="Mask source for image_reference/sketch_reference propagation. video_target keeps the original clicked video mask; anydoor_object re-segments the AnyDoor first frame and fits that object shape to the original video mask motion; anydoor_object_sam2 tracks the object mask with SAM2.")
    parser.add_argument("--reference_motion_guide", choices=["none", "full_region", "edge_refine"], default="none", help="Experimental guide mode for reference replacement. none keeps masked-video conditioning; full_region composites the reference object into conditioning frames; edge_refine composites the object and asks VideoPainter to refine boundary masks.")
    parser.add_argument("--conditioning_video_mode", choices=["lama_cleaned_video", "masked_video", "full_video"], default="lama_cleaned_video", help="Conditioning video mode for reference replacement. lama_cleaned_video removes the original target from every conditioning frame with LaMa while preserving the original masks; full_video keeps original pixels visible; masked_video blacks out edit regions.")
    parser.add_argument("--conditioning_lama_mask_mode", choices=["rect", "dilate", "exact"], default="rect", help="Mask shape for per-frame LaMa conditioning cleanup.")
    parser.add_argument("--conditioning_lama_mask_padding", type=int, default=48, help="Padding in pixels for rect per-frame LaMa cleanup mask. This does not change the CogVideoX edit masks.")
    parser.add_argument("--conditioning_lama_mask_dilate", type=int, default=31, help="Dilation kernel size for per-frame LaMa cleanup mask.")
    parser.add_argument(
        "--edit_mask_mode",
        choices=["propagation", "union_target_object"],
        default="propagation",
        help=(
            "Mask passed to CogVideoX for reference replacement. propagation keeps the current "
            "replacement propagation mask; union_target_object edits the union of the original "
            "clicked video target mask and the propagated replacement object mask."
        ),
    )
    parser.add_argument("--guide_edge_inner_erode", type=int, default=4)
    parser.add_argument("--guide_edge_outer_dilate", type=int, default=8)
    parser.add_argument("--guide_dilate_size", type=int, default=0, help="Mask dilation size used when reference_motion_guide=edge_refine.")
    parser.add_argument("--mask_bbox_smoothing", choices=["off", "median"], default="off", help="Temporal smoothing for per-frame target mask bboxes before composing reference guide frames.")
    parser.add_argument("--mask_bbox_smoothing_window", type=int, default=0, help="Median window for mask bbox smoothing.")
    parser.add_argument("--mask_bbox_max_scale_delta", type=float, default=0.08, help="Maximum per-frame relative width/height change after bbox smoothing.")
    parser.add_argument("--save_mask_bbox_stats", action="store_true", help="Save per-frame mask bbox diagnostics to mask_bbox_stats.json.")
    parser.add_argument("--model_path", default=str(ROOT / "ckpt" / "CogVideoX-5b-I2V"))
    parser.add_argument("--inpainting_branch", default=str(ROOT / "ckpt" / "VideoPainter" / "checkpoints" / "branch"))
    parser.add_argument("--id_adapter", default=str(ROOT / "ckpt" / "VideoPainterID" / "checkpoints"))
    parser.add_argument("--img_inpainting_model", default=str(ROOT / "ckpt" / "flux_inp"))
    parser.add_argument("--sam2_ckpt", default=str(ROOT / "ckpt" / "sam2_hiera_large.pt"))
    parser.add_argument("--sam2_cfg", default="sam2_hiera_l.yaml")
    parser.add_argument("--cache_dir", default=str(ROOT / "ckpt" / "sketch_ref"))
    parser.add_argument("--device", default="cuda:0")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "text_prompt" and not args.target_object_caption:
        parser.error("--target_object_caption is required for --mode text_prompt")
    if args.mode == "image_reference":
        if not args.reference_image:
            parser.error("--reference_image is required for --mode image_reference")
        if not args.reference_mask and not args.reference_points:
            parser.error("--reference_points or --reference_mask is required for --mode image_reference")
    if args.mode == "sketch_reference":
        if not args.sketch_image or not args.label:
            parser.error("--sketch_image and --label are required for --mode sketch_reference")
    run(args)


if __name__ == "__main__":
    main()
