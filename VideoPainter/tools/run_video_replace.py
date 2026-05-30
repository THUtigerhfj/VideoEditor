#!/usr/bin/env python3
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
from sam2.build_sam import build_sam2, build_sam2_video_predictor  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402
import utils as vp_utils  # noqa: E402
from utils import generate_frames, load_model  # noqa: E402


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
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
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

    prompt_payload = {
        "mode": args.mode,
        "video_caption": args.video_caption,
        "target_object_caption": args.target_object_caption,
        "note": (
            "video_caption is passed to VideoPainter as the edited-video description. "
            "For image_reference mode, AnyDoor creates the first frame and VideoPainter uses "
            "that first frame plus video_caption to propagate the replacement."
        ),
    }
    (dirs["prompts"] / "captions.json").write_text(json.dumps(prompt_payload, indent=2, ensure_ascii=False))
    (dirs["prompts"] / "video_caption.txt").write_text(str(args.video_caption) + "\n")
    if args.target_object_caption:
        (dirs["prompts"] / "target_object_caption.txt").write_text(str(args.target_object_caption) + "\n")

    point_payload = {
        "video_points_720x480": video_points.tolist(),
        "video_labels": video_labels.tolist(),
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


def pil_frames_and_masks(frames, masks):
    pil_frames = [Image.fromarray(frame).convert("RGB") for frame in frames]
    pil_masks = []
    for mask in masks:
        m = (mask.squeeze() > 0).astype(np.uint8) * 255
        pil_masks.append(Image.fromarray(np.stack([m, m, m], axis=-1)).convert("RGB"))
    return pil_frames, pil_masks


def run(args):
    dirs = ensure_run_dirs(args.output_dir)
    os.environ["GRADIO_TEMP_DIR"] = str(dirs["root"] / "tmp_gradio")
    vp_utils.GRADIO_TEMP_DIR = os.environ["GRADIO_TEMP_DIR"]
    (dirs["root"] / "tmp_gradio" / "inpaint").mkdir(parents=True, exist_ok=True)
    (dirs["root"] / "tmp_gradio" / "track").mkdir(parents=True, exist_ok=True)

    video_points, video_labels = parse_points(args.video_points)
    frames = load_video_frames(args.input_video)
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

    elif args.mode == "image_reference":
        ref_points, ref_labels = parse_points(args.reference_points)
        save_run_inputs(args, dirs, frames, video_points, video_labels, ref_points, ref_labels)
        reference_image, reference_mask, reference_mask_np = segment_reference_image(
            args.reference_image,
            ref_points,
            ref_labels,
            args.sam2_cfg,
            args.sam2_ckpt,
        )
        reference_image.save(dirs["refs"] / "reference_image.png")
        reference_src = Path(args.reference_image)
        if reference_src.exists():
            shutil.copy2(reference_src, dirs["refs"] / reference_src.name)
        reference_mask.save(dirs["masks"] / "reference_object_mask.png")

        target_first = Image.fromarray(frames[0]).convert("RGB")
        target_mask = Image.fromarray((masks[0].squeeze() * 255).astype(np.uint8)).convert("L")
        replaced_first = replace_first_frame_with_anydoor(
            target_image=target_first,
            target_mask=target_mask,
            reference_image=reference_image,
            reference_mask=reference_mask,
            guidance_scale=args.anydoor_guidance_scale,
        )
        anydoor_path = dirs["first_frames"] / "anydoor_first_frame.png"
        replaced_first.save(anydoor_path)
        torch.cuda.empty_cache()

        pipe, _ = load_model(
            model_path=args.model_path,
            inpainting_branch=args.inpainting_branch,
            img_inpainting_model="",
            id_adapter=args.id_adapter,
            device=args.device,
        )
        outputs = generate_frames(
            images=[img.copy() for img in pil_frames],
            masks=[mask.copy() for mask in pil_masks],
            pipe=pipe,
            pipe_img_inpainting=None,
            prompt=args.video_caption,
            image_inpainting_prompt="",
            seed=args.seed,
            cfg_scale=args.cfg_scale,
            dilate_size=args.dilate_size,
            first_frame_override=replaced_first,
        )
        raw_path = save_video(outputs, dirs["outputs"] / "image_reference_videopainter_48f.mp4")
        combined = np.concatenate([np.asarray(replaced_first.resize((720, 480)), dtype=np.uint8)[None, ...], np.clip(outputs * 255, 0, 255).astype(np.uint8)], axis=0)
        final_path = save_video(combined, dirs["outputs"] / "image_reference_videopainter_49f_with_first.mp4")
        metadata["outputs"].update({"first_frame": str(anydoor_path), "raw_48f": str(raw_path), "final_49f": str(final_path)})

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    metadata_path = dirs["root"] / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(json.dumps(metadata["outputs"], indent=2, ensure_ascii=False))
    print(f"METADATA={metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Run VideoPainter replacement without Gradio.")
    parser.add_argument("--mode", choices=["text_prompt", "image_reference"], required=True)
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", default=str(ROOT / "runs" / "manual_run"))
    parser.add_argument("--video_points", required=True, help="JSON file/string or 'x,y,label;x,y,label'. Labels: 1 positive, 0 negative.")
    parser.add_argument("--reference_image")
    parser.add_argument("--reference_points", help="Required for image_reference. JSON file/string or 'x,y,label;x,y,label'.")
    parser.add_argument(
        "--video_caption",
        required=True,
        help=(
            "Edited-video caption passed to VideoPainter. Prefer a description of the final video "
            "instead of an instruction such as 'replace X with Y'."
        ),
    )
    parser.add_argument("--target_object_caption", default="", help="Required for text_prompt; FLUX first-frame prompt.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_scale", type=float, default=6.0)
    parser.add_argument("--dilate_size", type=int, default=16)
    parser.add_argument("--sam2_dilate_iter", type=int, default=6)
    parser.add_argument("--anydoor_guidance_scale", type=float, default=5.0)
    parser.add_argument("--model_path", default=str(ROOT / "ckpt" / "CogVideoX-5b-I2V"))
    parser.add_argument("--inpainting_branch", default=str(ROOT / "ckpt" / "VideoPainter" / "checkpoints" / "branch"))
    parser.add_argument("--id_adapter", default=str(ROOT / "ckpt" / "VideoPainterID" / "checkpoints"))
    parser.add_argument("--img_inpainting_model", default=str(ROOT / "ckpt" / "flux_inp"))
    parser.add_argument("--sam2_ckpt", default=str(ROOT / "ckpt" / "sam2_hiera_large.pt"))
    parser.add_argument("--sam2_cfg", default="sam2_hiera_l.yaml")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.mode == "text_prompt" and not args.target_object_caption:
        parser.error("--target_object_caption is required for --mode text_prompt")
    if args.mode == "image_reference":
        if not args.reference_image or not args.reference_points:
            parser.error("--reference_image and --reference_points are required for --mode image_reference")
    run(args)


if __name__ == "__main__":
    main()
