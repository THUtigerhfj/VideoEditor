import os
import warnings
warnings.filterwarnings("ignore")

import argparse
import cv2
import json
import numpy as np
import scipy
import scipy.ndimage
import sys
import time
import torchvision
import torch
from collections import OrderedDict
from PIL import Image
from decord import VideoReader
from omegaconf import OmegaConf

import gradio as gr

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from frame_conditioning import build_edge_refine_masks, composite_object_guide_frames, finalize_propagation_sequence, mask_bbox, mask_bbox_trajectory, smooth_mask_bboxes
from reference_segmenter import clean_binary_mask, detect_object_box_with_grounding_dino, segment_box_with_sam2
from utils import REFERENCE_PREV_CLIP_WEIGHT, build_reference_propagation_prompt, load_model, generate_frames
import threading
from anydoor_bridge import replace_first_frame_with_anydoor
from sketch_reference_workflow import build_reference_preview, generate_reference_assets, load_reference_assets_for_ui

# Gradio temp directory (absolute path)
GRADIO_TEMP_DIR = os.path.abspath(os.environ.get("GRADIO_TEMP_DIR", "./tmp_gradio"))
os.makedirs(GRADIO_TEMP_DIR, exist_ok=True)
os.makedirs(os.path.join(GRADIO_TEMP_DIR, "track"), exist_ok=True)
os.makedirs(os.path.join(GRADIO_TEMP_DIR, "inpaint"), exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = GRADIO_TEMP_DIR
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SKETCH_REF_CACHE_DIR = os.path.abspath(os.path.join(APP_DIR, "..", "ckpt", "sketch_ref"))
os.makedirs(SKETCH_REF_CACHE_DIR, exist_ok=True)


def build_target_bboxes_for_guides_ui(target_masks, smoothing_mode="off", smoothing_window=5, max_scale_delta=0.08, output_size=(720, 480)):
    raw_bboxes = [mask_bbox(mask) for mask in target_masks]
    if smoothing_mode == "median":
        return smooth_mask_bboxes(
            raw_bboxes,
            image_size=output_size,
            window=int(smoothing_window),
            max_scale_delta=float(max_scale_delta),
        )
    return raw_bboxes


def save_mask_bbox_diagnostics_ui(path, target_masks, target_bboxes):
    payload = {
        "raw": mask_bbox_trajectory(target_masks),
        "used_bboxes": [None if box is None else [int(v) for v in box] for box in target_bboxes],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

# Increase Gradio timeouts for long-running SAM2 operations
os.environ.setdefault("GRADIO_TIMEOUT", "900")
os.environ.setdefault("GRADIO_CLIENT_TIMEOUT", "900")
os.environ.setdefault("GRADIO_VIDEO_CACHE_SIZE", "100")
os.environ.setdefault("GRADIO_SERVER_TIMEOUT", "900")

# Optional VLM for prompt enhancement
try:
    from openai import OpenAI
    vlm_model = OpenAI() if os.environ.get("OPENAI_API_KEY") else None
except ImportError:
    vlm_model = None

# =============================================================================
# Argument Parser
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="../ckpt/CogVideoX-5b-I2V",
                    help="Path to CogVideoX model")
parser.add_argument("--inpainting_branch", type=str, default="../ckpt/VideoPainter/checkpoints/branch",
                    help="Path to inpainting branch checkpoints")
parser.add_argument("--id_adapter", type=str, default="../ckpt/VideoPainterID/checkpoints",
                    help="Path to ID adapter checkpoints")
parser.add_argument("--img_inpainting_model", type=str, default="../ckpt/flux_inp",
                    help="Path to image inpainting model (Flux)")
parser.add_argument("--sam2_checkpoint", type=str, default="../ckpt/sam2_hiera_large.pt",
                    help="Path to SAM2 checkpoint")
parser.add_argument("--port", type=int, default=7860,
                    help="Port to run the Gradio app on")
args = parser.parse_args()

# =============================================================================
# Global state for latest results and processing status
# =============================================================================
latest_tracking_video = None
latest_inpaint_video = None
latest_sketch_reference = None

# Processing state for background jobs
processing_lock = threading.Lock()
processing_status = {
    "tracking": False,
    "inpainting": False,
    "sketch_reference": False,
    "tracking_message": "",
    "inpainting_message": "",
    "sketch_reference_message": "",
    "tracking_error": "",
    "inpainting_error": "",
    "sketch_reference_error": ""
}


# =============================================================================
# Load Models
# =============================================================================
print("Loading SAM2 video predictor...")
predictor = build_sam2_video_predictor(
    "sam2_hiera_l.yaml",
    args.sam2_checkpoint,
    apply_postprocessing=False
)
print("SAM2 video predictor loaded!")

print("Loading SAM2 image predictor...")
sam2_model = build_sam2("sam2_hiera_l.yaml", args.sam2_checkpoint, device="cuda")
image_predictor = SAM2ImagePredictor(sam2_model)
print("SAM2 image predictor loaded!")

print("Loading VideoPainter models...")
validation_pipeline, validation_pipeline_img = load_model(
    model_path=args.model_path,
    inpainting_branch=args.inpainting_branch,
    id_adapter=args.id_adapter,
    img_inpainting_model=args.img_inpainting_model
)
print("All models loaded!")


# =============================================================================
# Status Messages
# =============================================================================
class StatusMessage:
    INFO = "Info"
    WARNING = "Warning"
    ERROR = "Error"
    SUCCESS = "Success"


def update_status(previous_status, new_message, message_type=StatusMessage.INFO):
    """Update status for HighlightedText component."""
    return [(new_message, message_type)]


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


def segment_anydoor_object_mask_for_ui(image, label):
    label = str(label or "").strip()
    if not label:
        raise ValueError("Target region caption is empty")
    image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
    detection = detect_object_box_with_grounding_dino(image, label, cache_dir=SKETCH_REF_CACHE_DIR, device="cuda")
    if detection is None:
        raise RuntimeError(f"Could not detect replacement object in AnyDoor first frame: {label}")
    return segment_box_with_sam2(image, detection["box"], image_predictor)


def track_masks_from_initial_mask(frames, initial_mask, dilate_iter=6):
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


def fit_object_mask_to_video_motion(object_mask, video_masks):
    fitted = []
    for video_mask in video_masks:
        fitted_mask = fit_reference_mask_to_target_bbox(object_mask, video_mask, output_size=(720, 480))
        fitted.append((fitted_mask > 0).astype(np.uint8)[:, :, None])
    masks = np.asarray(fitted, dtype=np.uint8)
    if masks.max() == 0:
        raise RuntimeError("Object-shaped propagation mask sequence is empty")
    return masks

# =============================================================================
# Video Processing Functions
# =============================================================================
def get_frames_from_video(video_path, video_state):
    """Extract frames from uploaded video and initialize SAM2 state."""
    if video_path is None or video_path == "":
        return None, None, video_state, "", None, update_status([("", "")], "Please upload a video first.", StatusMessage.WARNING), video_state

    name = os.path.basename(video_path)
    vr = VideoReader(video_path)
    original_fps = vr.get_avg_fps()
    total_frames = len(vr)
    target_fps = 8
    target_frame_count = 49

    # Sample frames evenly
    num_sampled = min(total_frames, target_frame_count)
    frame_indices = np.linspace(0, total_frames - 1, num_sampled, dtype=int)
    frames = vr.get_batch(frame_indices).asnumpy()

    # Pad to exactly 49 frames if needed
    if len(frames) < target_frame_count:
        padding_frames = target_frame_count - len(frames)
        last_frame = frames[-1]
        padding = np.repeat(last_frame[np.newaxis, ...], padding_frames, axis=0)
        frames = np.concatenate([frames, padding], axis=0)

    # Resize to 480x720
    resized_frames = []
    for frame in frames:
        resized_frame = cv2.resize(frame, (720, 480))
        resized_frames.append(resized_frame)
    frames = np.array(resized_frames)

    # Initialize SAM2 inference state
    inference_state = predictor.init_state(
        images=frames,
        offload_video_to_cpu=True,
        async_loading_frames=True
    )

    # Update video state
    video_state = {
        "user_name": time.time(),
        "video_name": name,
        "origin_images": frames,
        "painted_images": frames.copy(),
        "masks": [np.zeros((frames[0].shape[0], frames[0].shape[1]), np.uint8) for _ in range(len(frames))],
        "logits": [None] * len(frames),
        "select_frame_number": 0,
        "fps": target_fps,
        "ann_obj_id": 0,
        "original_frame_count": num_sampled,
    }

    video_info = f"Video: {name} | FPS: {original_fps:.2f} | Frames: {total_frames} -> {len(frames)} | Size: 480x720"

    return (
        "",  # video_caption
        inference_state,
        video_state,
        video_info,
        frames[0],
        update_status([("", "")], f"Loaded {len(frames)} frames. Click to select region.", StatusMessage.SUCCESS)
    )


def sam_refine(inference_state, video_state, point_prompt, click_state, evt: gr.SelectData, previous_status):
    """Handle point clicks on video frame for SAM2 segmentation."""
    if video_state is None or video_state["origin_images"] is None:
        return None, video_state, [[], []], previous_status

    ann_frame_idx = video_state["select_frame_number"]
    label = 1 if point_prompt == "Positive" else 0

    click_state[0].append([evt.index[0], evt.index[1]])
    click_state[1].append(label)

    points = np.array(click_state[0])
    labels = np.array(click_state[1])

    height, width = video_state["origin_images"][0].shape[:2]
    org_image = video_state["origin_images"][ann_frame_idx]

    try:
        _, _, mask = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=0,
            points=points,
            labels=labels,
        )

        mask_ = mask.cpu().squeeze().detach().numpy()
        mask_[mask_ <= 0] = 0
        mask_[mask_ > 0] = 1
        mask_ = cv2.resize(mask_, (width, height))
        mask_ = mask_[:, :, None]
        mask_[mask_ > 0.5] = 1
        mask_[mask_ <= 0.5] = 0

        color = 63 * np.ones((height, width, 3)) * np.array([[[np.random.randint(5), np.random.randint(5), np.random.randint(5)]]])
        painted_image = np.uint8((1 - 0.5 * mask_) * org_image + 0.5 * mask_ * color)

        video_state["masks"][ann_frame_idx] = mask_
        video_state["painted_images"][ann_frame_idx] = painted_image

        return painted_image, video_state, click_state, \
               update_status(previous_status, f"Added {point_prompt} point. Click Tracking to propagate.", StatusMessage.SUCCESS)
    except Exception as e:
        import traceback
        print(f"SAM2 error: {traceback.format_exc()}")
        return None, video_state, [[], []], \
               update_status(previous_status, f"Error: {str(e)}", StatusMessage.ERROR)


def clear_click(inference_state, video_state, previous_status):
    """Clear all clicks and reset SAM2 state."""
    if inference_state is not None:
        predictor.reset_state(inference_state)
    if video_state is not None and video_state["origin_images"] is not None:
        template_frame = video_state["origin_images"][video_state["select_frame_number"]]
    else:
        template_frame = None
    return inference_state, template_frame, [[], []], "", \
           update_status(previous_status, "Cleared all points.", StatusMessage.INFO)


def track_video(inference_state, video_state, previous_status):
    """Propagate mask across all video frames - runs in background, returns immediately."""
    if video_state is None or video_state["origin_images"] is None:
        return None, video_state, None, update_status(previous_status, "Please upload a video first.", StatusMessage.ERROR)

    height, width = video_state["origin_images"][0].shape[:2]
    total_frames = len(video_state["origin_images"])

    # Check if already processing
    with processing_lock:
        if processing_status["tracking"]:
            return None, video_state, "⚠️ Already tracking a video. Please wait.", \
                   update_status(previous_status, "Already tracking. Please wait.", StatusMessage.WARNING)

        # Mark as processing
        processing_status["tracking"] = True
        processing_status["tracking_message"] = "Starting..."
        processing_status["tracking_error"] = ""

    def track_in_background():
        """Run SAM2 tracking in background thread."""
        global latest_tracking_video
        try:
            print(f"Starting SAM2 tracking for {total_frames} frames...")

            masks = []
            frame_count = 0

            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                mask = np.zeros([480, 720, 1])
                for i in range(len(out_mask_logits)):
                    out_mask = out_mask_logits[i].cpu().squeeze().detach().numpy()
                    out_mask[out_mask > 0] = 1
                    out_mask[out_mask <= 0] = 0
                    out_mask = out_mask[:, :, None]
                    mask += out_mask
                mask = cv2.resize(mask, (width, height))
                mask = mask[:, :, None]
                mask[mask > 0.5] = 1
                mask[mask < 1] = 0
                mask = scipy.ndimage.binary_dilation(mask, iterations=6)
                masks.append(mask)
                frame_count += 1

                # Update progress every 5 frames
                if frame_count % 5 == 0 or frame_count == total_frames:
                    progress_pct = int(100 * frame_count / total_frames)
                    with processing_lock:
                        processing_status["tracking_message"] = f"Tracking frame {frame_count}/{total_frames} ({progress_pct}%)"
                    print(f"Tracking progress: {frame_count}/{total_frames} frames ({progress_pct}%)")

            masks = np.array(masks)
            print(f"Tracking complete! Generating preview video...")

            with processing_lock:
                processing_status["tracking_message"] = "Generating preview video..."

            # Update painted images
            org_images = video_state["origin_images"]
            color = 255 * np.ones((1, org_images.shape[-3], org_images.shape[-2], 3)) * np.array([[[[0, 1, 1]]]])
            painted_images = np.uint8((1 - 0.5 * masks) * org_images + 0.5 * masks * color)

            # Update video_state (need to handle this carefully since it's shared)
            video_state["masks"] = masks
            video_state["painted_images"] = painted_images

            # Generate video preview
            video_output = generate_video_from_frames(
                painted_images,
                output_path=os.path.join(GRADIO_TEMP_DIR, "track", video_state['video_name']),
                fps=8
            )

            latest_tracking_video = video_output
            with processing_lock:
                processing_status["tracking_message"] = "Complete!"
                processing_status["tracking"] = False
            print(f"Tracking video saved: {video_output}")

        except Exception as e:
            import traceback
            print(f"Tracking error: {traceback.format_exc()}")
            with processing_lock:
                processing_status["tracking_error"] = str(e)
                processing_status["tracking"] = False

    # Start background tracking
    thread = threading.Thread(target=track_in_background, daemon=True)
    thread.start()

    # Return immediately - no yield, no blocking window
    return None, video_state, "🚀 Tracking started in background!\n\nWait 30-60 seconds, then click 'Load Latest Result' to view the tracked video.\n\n(Terminal will show progress updates)", \
           update_status(previous_status, "Tracking in background... Click 'Load Latest Result' when done.", StatusMessage.INFO)


def semantic_segment_reference(ref_image, point_prompt, click_state, evt: gr.SelectData):
    """Point-based segmentation of reference image."""
    if ref_image is None:
        return ref_image, click_state, None

    image = np.asarray(ref_image, dtype=np.uint8)
    image_predictor.set_image(image)

    label = 1 if point_prompt == "Positive" else 0
    click_state[0].append([evt.index[0], evt.index[1]])
    click_state[1].append(label)

    points = np.array(click_state[0])
    labels = np.array(click_state[1])

    masks, scores, _ = image_predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    mask = (masks[best_idx] > 0).astype(np.uint8)

    color = np.zeros_like(image)
    color[:, :, 1] = 255
    painted_image = np.uint8((1 - 0.45 * mask[:, :, None]) * image + 0.45 * mask[:, :, None] * color)

    return painted_image, click_state, mask


def generate_reference_from_sketch(sketch_image, video_state, label, attrs, candidate_count, reference_num_inference_steps, reference_guidance_scale, reference_controlnet_scale, seed_param, previous_status):
    """Start sketch-to-reference generation in a background thread.

    README notes that this Gradio version has a short request timeout, so heavy
    tasks must return immediately and be retrieved with explicit load buttons.
    """
    global latest_sketch_reference
    if sketch_image is None:
        return "", update_status(previous_status, "Please upload a sketch first.", StatusMessage.ERROR)

    clean_label = str(label).strip()
    if not clean_label:
        return "", update_status(previous_status, "Object label is required for sketch mode.", StatusMessage.ERROR)

    with processing_lock:
        if processing_status["sketch_reference"]:
            return "⚠️ Already generating a sketch reference. Please wait.", update_status(previous_status, "Sketch reference generation already running.", StatusMessage.WARNING)
        processing_status["sketch_reference"] = True
        processing_status["sketch_reference_message"] = "Starting..."
        processing_status["sketch_reference_error"] = ""
        latest_sketch_reference = None

    sketch_image_copy = np.asarray(sketch_image, dtype=np.uint8).copy()
    target_mask_for_shape = None
    if video_state is not None and video_state.get("masks") is not None and len(video_state["masks"]) > 0:
        target_mask_for_shape = (np.squeeze(video_state["masks"][0]) > 0).astype(np.uint8) * 255

    seed = int(seed_param) if int(seed_param) >= 0 else -1
    run_dir = os.path.join(GRADIO_TEMP_DIR, "sketch_ref", str(int(time.time() * 1000)))
    os.makedirs(run_dir, exist_ok=True)

    def generate_in_background():
        global latest_sketch_reference
        try:
            with processing_lock:
                processing_status["sketch_reference_message"] = "Loading SDXL scribble ControlNet from local cache..."
            print("Stage sketch-reference: loading/generating reference from sketch...")
            sketch_pil = Image.fromarray(sketch_image_copy).convert("RGB")
            with processing_lock:
                processing_status["sketch_reference_message"] = "Generating reference candidates..."
            result = generate_reference_assets(
                sketch_image=sketch_pil,
                label=clean_label,
                attrs=attrs,
                output_dir=run_dir,
                cache_dir=SKETCH_REF_CACHE_DIR,
                sam2_cfg="sam2_hiera_l.yaml",
                sam2_ckpt=args.sam2_checkpoint,
                device="cuda",
                candidate_count=int(candidate_count),
                seed=seed,
                reference_num_inference_steps=int(reference_num_inference_steps),
                reference_guidance_scale=float(reference_guidance_scale),
                reference_controlnet_scale=float(reference_controlnet_scale),
                allow_missing_mask=True,
                target_mask_for_shape=target_mask_for_shape,
                shape_compatibility_weight=0.25,
            )
            with processing_lock:
                processing_status["sketch_reference_message"] = "Loading generated reference assets..."
            ref_image, ref_mask = load_reference_assets_for_ui(
                result["reference_image_path"],
                result["reference_mask_path"],
            )
            preview = build_reference_preview(ref_image, ref_mask)
            latest_sketch_reference = {
                "ref_image": ref_image,
                "preview": preview,
                "ref_mask": ref_mask,
                "run_dir": run_dir,
                "metadata": result.get("metadata", {}),
            }
            with processing_lock:
                processing_status["sketch_reference_message"] = "Complete!"
                processing_status["sketch_reference"] = False
            print(f"Sketch reference saved under: {run_dir}")
        except Exception as exc:
            import traceback
            print(f"Sketch reference error: {traceback.format_exc()}")
            with processing_lock:
                processing_status["sketch_reference_error"] = str(exc)
                processing_status["sketch_reference"] = False

    thread = threading.Thread(target=generate_in_background, daemon=True)
    thread.start()
    return "🚀 Sketch reference generation started in background!\n\nWait, click 'Check Sketch Status', then 'Load Sketch Result' when complete.\n\n(Terminal will show model loading/generation progress.)", update_status(previous_status, "Sketch reference generation in background... Click 'Load Sketch Result' when done.", StatusMessage.INFO)

def inpaint_video_background(video_state, video_caption, target_region_caption, previous_status, seed_param, cfg_scale, dilate_size):
    """Inpaint video - runs in background, returns immediately."""
    if video_state is None or video_state["origin_images"] is None:
        return None, "", update_status(previous_status, "Please upload and track a video first.", StatusMessage.ERROR)
    if video_state["masks"] is None or len(video_state["masks"]) == 0:
        return None, "", update_status(previous_status, "Please track the video first.", StatusMessage.ERROR)

    seed = int(seed_param) if int(seed_param) >= 0 else np.random.randint(0, 2**32 - 1)

    # Check if already processing
    with processing_lock:
        if processing_status["inpainting"]:
            return None, "⚠️ Already processing a video. Please wait.", \
                   update_status(previous_status, "Already processing. Please wait.", StatusMessage.WARNING)

        # Mark as processing
        processing_status["inpainting"] = True
        processing_status["inpainting_message"] = "Starting..."
        processing_status["inpainting_error"] = ""

    def process_in_background():
        """Run the entire processing pipeline in background thread."""
        global latest_inpaint_video
        try:
            print("Preparing frames for inpainting...")
            frame_indices = list(range(0, len(video_state["origin_images"]), 1))
            validation_images = np.asarray(video_state["origin_images"])[frame_indices]
            validation_masks = np.asarray(video_state["masks"])[frame_indices]

            # Convert masks to 3-channel format
            validation_masks = [np.squeeze(mask) for mask in validation_masks]
            validation_masks = [(mask > 0).astype(np.uint8) * 255 for mask in validation_masks]
            validation_masks = [np.stack([m, m, m], axis=-1) for m in validation_masks]

            # Convert to PIL Images
            validation_images = [Image.fromarray(np.uint8(img)).convert('RGB') for img in validation_images]
            validation_masks = [Image.fromarray(np.uint8(mask)).convert('RGB') for mask in validation_masks]

            validation_images = [img.resize((720, 480)) for img in validation_images]
            validation_masks = [mask.resize((720, 480)) for mask in validation_masks]

            print(f"Video caption: {video_caption}")
            print(f"Target region caption: {target_region_caption}")

            with processing_lock:
                processing_status["inpainting_message"] = "Stage 1/3 & 2/3: Running FLUX + CogVideoX..."

            def progress_callback(step, total, message):
                """Receive progress updates."""
                with processing_lock:
                    processing_status["inpainting_message"] = f"Processing: {message} ({int(100*step/total)}%)"
                print(f"[Background Progress] {message} ({int(100*step/total)}%)")

            images = generate_frames(
                images=validation_images,
                masks=validation_masks,
                pipe=validation_pipeline,
                pipe_img_inpainting=validation_pipeline_img,
                prompt=str(video_caption),
                image_inpainting_prompt=str(target_region_caption),
                seed=seed,
                cfg_scale=float(cfg_scale),
                dilate_size=int(dilate_size),
                progress_callback=progress_callback,
            )

            print("Processing complete!")
            images = (images * 255).astype(np.uint8)

            # Trim output to original frame count while preserving the propagated first frame.
            original_count = video_state.get("original_frame_count", len(images))
            if len(images) > original_count:
                images = images[:original_count]

            Image.fromarray(images[0]).save(os.path.join(GRADIO_TEMP_DIR, "inpaint", "first_frame_propagated.png"))

            # Generate output video
            with processing_lock:
                processing_status["inpainting_message"] = "Generating output video..."
            print("Generating output video...")
            video_output = generate_video_from_frames(
                images,
                output_path=os.path.join(GRADIO_TEMP_DIR, "inpaint", video_state['video_name']),
                fps=8
            )

            latest_inpaint_video = video_output
            with processing_lock:
                processing_status["inpainting_message"] = "Complete!"
                processing_status["inpainting"] = False
            print(f"Inpainting video saved: {video_output}")

        except Exception as e:
            import traceback
            print(f"Processing error: {traceback.format_exc()}")
            with processing_lock:
                processing_status["inpainting_error"] = str(e)
                processing_status["inpainting"] = False

    # Start background processing
    thread = threading.Thread(target=process_in_background, daemon=True)
    thread.start()

    # Return immediately - no yield, no blocking window
    return None, "🚀 Processing started in background!\n\nWait 1-2 minutes, then click 'Load Latest Result' to view your video.\n\n(Terminal will show progress updates)", \
           update_status(previous_status, "Processing in background... Click 'Load Latest Result' when done.", StatusMessage.INFO)


def exact_replace_video_background(video_state, ref_image_input, ref_mask, video_caption, target_region_caption, previous_status, seed_param, cfg_scale, dilate_size, anydoor_guidance_scale, reference_motion_guide, guide_dilate_size, mask_bbox_smoothing, mask_bbox_smoothing_window, mask_bbox_max_scale_delta, save_mask_bbox_stats):
    """Exact object replacement - runs in background, returns immediately."""
    seed = int(seed_param) if int(seed_param) >= 0 else np.random.randint(0, 2**32 - 1)

    if video_state is None or video_state.get("origin_images") is None:
        return None, "", update_status(previous_status, "Please upload and track a video first.", StatusMessage.ERROR)
    if ref_image_input is None:
        return None, "", update_status(previous_status, "Please upload a reference image first.", StatusMessage.ERROR)
    if ref_mask is None or np.max(ref_mask) == 0:
        return None, "", update_status(previous_status, "Please segment the reference image first.", StatusMessage.ERROR)
    if video_state.get("masks") is None or len(video_state["masks"]) == 0:
        return None, "", update_status(previous_status, "Please track the video first.", StatusMessage.ERROR)

    # Check if already processing
    with processing_lock:
        if processing_status["inpainting"]:
            return None, "⚠️ Already processing a video. Please wait.", \
                   update_status(previous_status, "Already processing. Please wait.", StatusMessage.WARNING)

        # Mark as processing
        processing_status["inpainting"] = True
        processing_status["inpainting_message"] = "Starting..."
        processing_status["inpainting_error"] = ""

    def process_in_background():
        """Run the entire processing pipeline in background thread."""
        global latest_inpaint_video
        try:
            print("Preparing frames for replacement...")
            frame_indices = list(range(0, len(video_state["origin_images"]), 1))
            validation_images = np.asarray(video_state["origin_images"])[frame_indices]
            validation_masks = np.asarray(video_state["masks"])[frame_indices]

            # Convert masks to 3-channel format
            validation_masks = [np.squeeze(mask) for mask in validation_masks]
            validation_masks = [(mask > 0).astype(np.uint8) * 255 for mask in validation_masks]

            # Check if first frame mask has any content before proceeding
            first_mask_check = validation_masks[0]
            print(f"First frame mask check - shape: {first_mask_check.shape}, max: {first_mask_check.max()}, sum: {np.sum(first_mask_check)}")
            if first_mask_check.max() == 0:
                with processing_lock:
                    processing_status["inpainting_error"] = "First frame mask is empty"
                    processing_status["inpainting"] = False
                return

            # Create 3-channel version for AnyDoor target-mask use. It may be replaced before CogVideoX.
            validation_masks_3ch = [np.stack([m, m, m], axis=-1) for m in validation_masks]

            # AnyDoor generates the first frame with the reference object
            with processing_lock:
                processing_status["inpainting_message"] = "Stage 1/3: Running AnyDoor..."
            print("Stage 1/3: Running AnyDoor for first frame replacement...")

            tar_image = np.array(validation_images[0]).copy()
            tar_mask_arr = (validation_masks[0] > 128).astype(np.uint8) * 255
            ref_mask_arr = (np.array(ref_mask) > 0).astype(np.uint8) * 255
            reference_image = Image.fromarray(np.asarray(ref_image_input, dtype=np.uint8)).convert('RGB')

            print(f"Target mask: sum={np.sum(tar_mask_arr)}, Reference mask: sum={np.sum(ref_mask_arr)}")

            try:
                print("Running AnyDoor replacement...")
                validation_pipeline.to("cpu")
                torch.cuda.empty_cache()
                gen_image = replace_first_frame_with_anydoor(
                    target_image=Image.fromarray(tar_image).convert('RGB'),
                    target_mask=Image.fromarray(tar_mask_arr, mode='L'),
                    reference_image=reference_image,
                    reference_mask=Image.fromarray(ref_mask_arr, mode='L'),
                    guidance_scale=float(anydoor_guidance_scale),
                )
                gen_image_array = np.array(gen_image)
                validation_images[0] = gen_image_array
                print(f"Stage 1/3: AnyDoor first frame complete! Shape: {gen_image_array.shape}")
            except Exception as exc:
                import traceback
                print(f"AnyDoor Error: {traceback.format_exc()}")
                with processing_lock:
                    processing_status["inpainting_error"] = f"AnyDoor error: {exc}"
                    processing_status["inpainting"] = False
                return
            finally:
                validation_pipeline.to("cuda")
                torch.cuda.empty_cache()

            os.makedirs(os.path.join(GRADIO_TEMP_DIR, "inpaint"), exist_ok=True)
            Image.fromarray(validation_images[0]).save(os.path.join(GRADIO_TEMP_DIR, "inpaint", "first_frame_anydoor.png"))

            try:
                object_mask = segment_anydoor_object_mask_for_ui(Image.fromarray(validation_images[0]).convert('RGB'), target_region_caption)
                propagation_mask_source = "anydoor_object"
            except Exception as exc:
                print(f"AnyDoor object mask segmentation failed; using fitted reference mask fallback: {exc}")
                object_mask = fit_reference_mask_to_target_bbox(ref_mask_arr, tar_mask_arr, output_size=(720, 480))
                propagation_mask_source = "fitted_reference_mask"

            Image.fromarray((np.asarray(object_mask) > 0).astype(np.uint8) * 255).convert('L').save(
                os.path.join(GRADIO_TEMP_DIR, "inpaint", "anydoor_object_first_mask.png")
            )
            tracked_masks = fit_object_mask_to_video_motion(object_mask, validation_masks)
            propagation_mask_source = f"{propagation_mask_source}_fit_to_video_motion"
            Image.fromarray((tracked_masks[0].squeeze() * 255).astype(np.uint8)).save(
                os.path.join(GRADIO_TEMP_DIR, "inpaint", "propagation_first_mask.png")
            )
            validation_masks_3ch = [np.stack([m.squeeze() * 255, m.squeeze() * 255, m.squeeze() * 255], axis=-1).astype(np.uint8) for m in tracked_masks]
            print(f"Using {propagation_mask_source} mask for CogVideoX propagation")

            # Now convert all frames to PIL
            validation_images = [Image.fromarray(np.uint8(img)).convert('RGB') for img in validation_images]
            validation_masks_pil = [Image.fromarray(np.uint8(mask)).convert('RGB') for mask in validation_masks_3ch]

            validation_images = [img.resize((720, 480)) for img in validation_images]
            validation_masks_pil = [mask.resize((720, 480)) for mask in validation_masks_pil]

            print(f"Using {propagation_mask_source} first frame mask for propagation")
            first_mask = np.array(validation_masks_pil[0])
            print(f"First mask after resize - pixel sum: {np.sum(first_mask)}")

            reference_motion_guide_value = reference_motion_guide or "none"
            guide_images = None
            guide_mode = "none"
            generation_masks_pil = validation_masks_pil
            anydoor_first_frame_pil = validation_images[0]
            tracked_mask_arrays = [(mask.squeeze() * 255).astype(np.uint8) for mask in tracked_masks]
            target_bboxes = build_target_bboxes_for_guides_ui(
                tracked_mask_arrays,
                smoothing_mode=mask_bbox_smoothing or "off",
                smoothing_window=int(mask_bbox_smoothing_window),
                max_scale_delta=float(mask_bbox_max_scale_delta),
                output_size=(720, 480),
            )
            if save_mask_bbox_stats:
                save_mask_bbox_diagnostics_ui(os.path.join(GRADIO_TEMP_DIR, "inpaint", "mask_bbox_stats.json"), tracked_mask_arrays, target_bboxes)
            if reference_motion_guide_value in {"full_region", "edge_refine"}:
                guide_images = composite_object_guide_frames(
                    validation_images,
                    anydoor_first_frame_pil.resize((720, 480)),
                    (np.asarray(object_mask) > 0).astype(np.uint8) * 255,
                    tracked_mask_arrays,
                    target_bboxes=target_bboxes,
                )
                guide_mode = reference_motion_guide_value
                guide_images[0].save(os.path.join(GRADIO_TEMP_DIR, "inpaint", "guide_frame_000.png"))
                guide_images[min(1, len(guide_images) - 1)].save(os.path.join(GRADIO_TEMP_DIR, "inpaint", "guide_frame_001.png"))
                if reference_motion_guide_value == "edge_refine":
                    generation_masks_pil = build_edge_refine_masks(tracked_mask_arrays)
                    generation_masks_pil[0].save(os.path.join(GRADIO_TEMP_DIR, "inpaint", "guide_edge_first_mask.png"))
                print(f"Using reference motion guide mode: {guide_mode}")

            effective_video_caption = build_reference_propagation_prompt(video_caption, target_region_caption)
            print(f"Video caption: {video_caption}")
            print(f"Target region caption: {target_region_caption}")
            print(f"Effective video caption: {effective_video_caption}")

            # VideoPainter propagation
            with processing_lock:
                processing_status["inpainting_message"] = "Stage 2/3: Running CogVideoX propagation (1-2 min)..."
            print("Stage 2/3: Propagating with CogVideoX...")

            def progress_callback(step, total, message):
                """Receive progress updates."""
                with processing_lock:
                    processing_status["inpainting_message"] = f"Stage 2/3: {message} ({int(100*step/total)}%)"
                print(f"[Background Progress] {processing_status['inpainting_message']}")

            images = generate_frames(
                images=validation_images,
                masks=generation_masks_pil,
                pipe=validation_pipeline,
                pipe_img_inpainting=validation_pipeline_img,
                prompt=str(effective_video_caption),
                image_inpainting_prompt="",
                seed=seed,
                cfg_scale=float(cfg_scale),
                dilate_size=int(dilate_size),
                first_frame_override=anydoor_first_frame_pil,
                progress_callback=progress_callback,
                return_full_sequence=True,
                prev_clip_weight=REFERENCE_PREV_CLIP_WEIGHT,
                id_pool_resample_learnable=True,
                guide_images=guide_images,
                guide_mode=guide_mode,
                guide_dilate_size=int(guide_dilate_size),
            )

            print("Stage 2/3: CogVideoX propagation complete!")
            images = finalize_propagation_sequence(images, fallback_first_frame=anydoor_first_frame_pil.resize((720, 480)))
            # Trim output to original frame count
            original_count = video_state.get("original_frame_count", len(images))
            if len(images) > original_count:
                images = images[:original_count]

            print(f"Generated frames shape: {images.shape}")

            # Generate output video
            with processing_lock:
                processing_status["inpainting_message"] = "Stage 3/3: Generating output video..."
            print("Stage 3/3: Generating output video...")
            video_output = generate_video_from_frames(
                images,
                output_path=os.path.join(GRADIO_TEMP_DIR, "inpaint", f"exact_replace_{video_state['video_name']}"),
                fps=8
            )

            if os.path.exists(video_output):
                file_size = os.path.getsize(video_output)
                print(f"Video file exists, size: {file_size/1024/1024:.2f} MB")
            else:
                print(f"WARNING: Video file does not exist at {video_output}")

            latest_inpaint_video = video_output
            with processing_lock:
                processing_status["inpainting_message"] = "Complete!"
                processing_status["inpainting"] = False
            print(f"Exact replacement video saved: {video_output}")

        except Exception as e:
            import traceback
            print(f"Processing error: {traceback.format_exc()}")
            with processing_lock:
                processing_status["inpainting_error"] = str(e)
                processing_status["inpainting"] = False

    # Start background processing
    thread = threading.Thread(target=process_in_background, daemon=True)
    thread.start()

    # Return immediately - no yield, no blocking window
    return None, "🚀 Processing started in background!\n\nWait 1-2 minutes, then click 'Load Latest Result' to view your video.\n\n(Terminal will show progress updates)", \
           update_status(previous_status, "Processing in background... Click 'Load Latest Result' when done.", StatusMessage.INFO)


# =============================================================================
# Load Latest Result Functions
# =============================================================================
def load_latest_tracking():
    """Load the latest tracking video if it exists."""
    global latest_tracking_video
    if latest_tracking_video and os.path.exists(latest_tracking_video):
        return latest_tracking_video, "✓ Video loaded successfully!", update_status([("", "")], "Loaded latest tracking result.", StatusMessage.SUCCESS)
    return None, None, update_status([("", "")], "No tracking result available. Try tracking first.", StatusMessage.WARNING)


def load_latest_inpaint():
    """Load the latest inpainting video if it exists."""
    global latest_inpaint_video
    if latest_inpaint_video and os.path.exists(latest_inpaint_video):
        return latest_inpaint_video, "✓ Video loaded successfully!", update_status([("", "")], "Loaded latest inpainting result.", StatusMessage.SUCCESS)
    return None, None, update_status([("", "")], "No inpainting result available. Try inpainting first.", StatusMessage.WARNING)


def load_latest_sketch_reference():
    """Load the latest sketch-generated reference image/mask if it exists."""
    global latest_sketch_reference
    if latest_sketch_reference:
        ref_mask = latest_sketch_reference.get("ref_mask")
        if ref_mask is None:
            return (
                latest_sketch_reference["ref_image"],
                latest_sketch_reference["preview"],
                [[], []],
                None,
                "✓ Reference image loaded. Auto mask failed; click the reference image to segment it manually.",
                update_status([("", "")], "Loaded sketch reference image; manual mask refinement needed.", StatusMessage.WARNING),
            )
        return (
            latest_sketch_reference["ref_image"],
            latest_sketch_reference["preview"],
            [[], []],
            ref_mask,
            "✓ Sketch reference image and mask loaded successfully!",
            update_status([("", "")], "Loaded latest sketch reference result.", StatusMessage.SUCCESS),
        )
    return None, None, [[], []], None, "", update_status([("", "")], "No sketch reference result available. Generate one first.", StatusMessage.WARNING)


def check_sketch_reference_status():
    """Check only sketch-reference generation status."""
    with processing_lock:
        if processing_status["sketch_reference"]:
            msg = processing_status["sketch_reference_message"]
            return f"⏳ Sketch reference generation in progress...\n\n{msg}\n\nWait, then click 'Load Sketch Result'.", update_status([("", "")], f"Sketch reference: {msg}", StatusMessage.INFO)
        if processing_status["sketch_reference_error"]:
            error = processing_status["sketch_reference_error"]
            processing_status["sketch_reference_error"] = ""
            return f"❌ Sketch reference generation failed: {error}", update_status([("", "")], f"Error: {error}", StatusMessage.ERROR)
    if latest_sketch_reference:
        return "✓ Sketch reference complete. Click 'Load Sketch Result'.", update_status([("", "")], "Sketch reference complete.", StatusMessage.SUCCESS)
    return "", update_status([("", "")], "No sketch reference generation running.", StatusMessage.INFO)


def check_processing_status():
    """Check the current processing status and return a message."""
    with processing_lock:
        # Check tracking status first
        if processing_status["tracking"]:
            msg = processing_status["tracking_message"]
            return None, f"⏳ Tracking in progress...\n\n{msg}\n\nWait, then click 'Load Latest Result'.", \
                   update_status([("", "")], f"Tracking: {msg}", StatusMessage.INFO)
        elif processing_status["tracking_error"]:
            error = processing_status["tracking_error"]
            # Clear error after reading
            processing_status["tracking_error"] = ""
            return None, f"❌ Tracking failed: {error}", \
                   update_status([("", "")], f"Error: {error}", StatusMessage.ERROR)
        # Check sketch-reference status
        elif processing_status["sketch_reference"]:
            msg = processing_status["sketch_reference_message"]
            return None, f"⏳ Sketch reference generation in progress...\n\n{msg}\n\nWait, then click 'Load Sketch Result'.", \
                   update_status([("", "")], f"Sketch reference: {msg}", StatusMessage.INFO)
        elif processing_status["sketch_reference_error"]:
            error = processing_status["sketch_reference_error"]
            processing_status["sketch_reference_error"] = ""
            return None, f"❌ Sketch reference generation failed: {error}", \
                   update_status([("", "")], f"Error: {error}", StatusMessage.ERROR)
        # Check inpainting status
        elif processing_status["inpainting"]:
            msg = processing_status["inpainting_message"]
            return None, f"⏳ Processing in progress...\n\n{msg}\n\nWait, then click 'Load Latest Result'.", \
                   update_status([("", "")], f"Processing: {msg}", StatusMessage.INFO)
        elif processing_status["inpainting_error"]:
            error = processing_status["inpainting_error"]
            # Clear error after reading
            processing_status["inpainting_error"] = ""
            return None, f"❌ Processing failed: {error}", \
                   update_status([("", "")], f"Error: {error}", StatusMessage.ERROR)
        else:
            return None, "", update_status([("", "")], "No processing running.", StatusMessage.INFO)


# =============================================================================
# VLM Prompt Enhancement (Optional)
# =============================================================================
def enhance_video_prompt(video_caption):
    """Enhance video caption using VLM (optional)."""
    if vlm_model is None:
        return video_caption
    try:
        sys_prompt = """You are part of a team of bots that creates videos. You work with an assistant bot that will draw anything you say in square brackets.
Your task is to take short prompts and make them extremely detailed and descriptive.
Video descriptions must have similar length to examples below. Extra words will be ignored."""

        response = vlm_model.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f'Create an imaginative video descriptive caption for: "{video_caption}"'},
            ],
            model="gpt-4o",
            temperature=0.01,
            max_tokens=200,
        )
        if response.choices:
            return response.choices[0].message.content
    except Exception as e:
        print(f"VLM enhancement error: {e}")
    return video_caption


def enhance_target_prompt(target_region_caption, video_state):
    """Generate target region caption using VLM (optional)."""
    if vlm_model is None or video_state is None or video_state.get("masks") is None:
        return target_region_caption

    try:
        # Create masked image
        validation_masks = video_state["masks"]
        masked_image = np.where(np.array(validation_masks[0]) > 0, np.array(video_state["origin_images"][0]), 0)
        masked_image = Image.fromarray(masked_image.astype(np.uint8))

        import base64
        from io import BytesIO
        buffered = BytesIO()
        masked_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        system_prompt = """You are an expert in visual scene understanding. Generate a concise description of the masked region."""

        response = vlm_model.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the masked region in under 20 words:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=50,
        )
        if response.choices:
            result = response.choices[0].message.content
            print(f"Generated target caption: {result}")
            return result
    except Exception as e:
        print(f"VLM target caption error: {e}")
    return target_region_caption


# =============================================================================
# Video Generation Helper
# =============================================================================
def generate_video_from_frames(frames, output_path, fps=8):
    """Generate MP4 video from frames."""
    frames = torch.from_numpy(np.asarray(frames)).to(torch.uint8)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    # Use absolute path for writing and return
    abs_output_path = os.path.abspath(output_path)
    torchvision.io.write_video(abs_output_path, frames, fps=fps, video_codec="libx264")

    # Verify the video was written
    if os.path.exists(abs_output_path):
        print(f"Video saved: {abs_output_path}")
    else:
        print(f"WARNING: Video not saved at {abs_output_path}")

    return abs_output_path


# =============================================================================
# Main Gradio Interface
# =============================================================================
def main():
    with gr.Blocks(title="VideoPainter") as iface:
        gr.HTML("""
        <div style="text-align: center;">
            <h1 style="color: #333;">🖌️ VideoPainter</h1>
            <h3 style="color: #666;">Video Inpainting and Editing with Plug-and-Play Context Control</h3>
            <p style="font-weight: bold">
                <a href="https://yxbian23.github.io/project/video-painter/">🌍 Project Page</a> |
                <a href="https://arxiv.org/abs/2503.05639">📃 ArXiv</a> |
                <a href="https://github.com/TencentARC/VideoPainter">🧑‍💻 GitHub</a>
            </p>
        </div>
        """)

        with gr.Row():
            # Left Column - Video Input
            with gr.Column(scale=1):
                gr.Markdown("### 📹 Input Video")
                video_input = gr.Video(label="Upload Video")

                gr.Markdown("### 🖱️ Select Region")
                template_frame = gr.Image(label="Click to select region", interactive=True, type="numpy")

                with gr.Row():
                    point_prompt = gr.Radio(
                        ["Positive", "Negative"],
                        value="Positive",
                        label="Point Type",
                        scale=1
                    )
                    clear_button = gr.Button("Clear Points", scale=1)

                gr.Markdown("### 🎬 Tracking")
                tracking_btn = gr.Button("Track (Propagate Mask)", variant="secondary")
                with gr.Row():
                    check_tracking_status_btn = gr.Button("🔄 Check Status", size="sm")
                    load_tracking_btn = gr.Button("🔄 Load Result", size="sm")
                tracking_progress = gr.Textbox(label="Tracking Progress", interactive=False, lines=3)
                tracked_video_output = gr.Video(label="Tracked Mask Preview")

            # Middle Column - Reference Image
            with gr.Column(scale=1):
                with gr.Accordion("Sketch to Reference", open=False):
                    sketch_image_input = gr.Image(label="Upload Sketch", type="numpy")
                    sketch_label_input = gr.Textbox(label="Object Label", placeholder="Required, e.g. apple")
                    sketch_attrs_input = gr.Textbox(label="Optional Attributes", placeholder="e.g. red, glossy")
                    sketch_candidate_count = gr.Slider(label="Candidate Count", minimum=1, maximum=8, step=1, value=2)
                    reference_num_inference_steps = gr.Slider(label="Reference Steps", minimum=20, maximum=80, step=1, value=40)
                    reference_guidance_scale = gr.Slider(label="Reference Guidance Scale", minimum=3.0, maximum=12.0, step=0.1, value=6.5)
                    reference_controlnet_scale = gr.Slider(label="Reference ControlNet Scale", minimum=0.2, maximum=1.2, step=0.05, value=0.7)
                    sketch_generate_btn = gr.Button("Generate Reference From Sketch", variant="secondary")
                    with gr.Row():
                        sketch_check_status_btn = gr.Button("🔄 Check Sketch Status", size="sm")
                        sketch_load_result_btn = gr.Button("🔄 Load Sketch Result", size="sm")
                    sketch_progress = gr.Textbox(label="Sketch Reference Progress", interactive=False, lines=3)

                gr.Markdown("### 🖼️ Reference Image (for Exact Replacement)")
                ref_image_input = gr.Image(label="Upload Reference Image", type="numpy")
                ref_template = gr.Image(label="Segmented Reference", interactive=False, type="numpy")

                with gr.Row():
                    ref_point_prompt = gr.Radio(
                        ["Positive", "Negative"],
                        value="Positive",
                        label="Point Type",
                        scale=1
                    )
                    ref_clear_button = gr.Button("Clear", scale=1)

                anydoor_guidance_scale = gr.Slider(
                    label="AnyDoor Guidance Scale",
                    minimum=1,
                    maximum=15,
                    step=0.5,
                    value=5.0
                )

                reference_motion_guide = gr.Radio(
                    choices=["none", "full_region", "edge_refine"],
                    value="none",
                    label="Reference Motion Guide",
                    info="Experimental. Use edge_refine for cross-shape objects after checking guide frames.",
                )
                guide_dilate_size = gr.Slider(label="Guide Dilate Size", minimum=0, maximum=32, step=1, value=4)
                mask_bbox_smoothing = gr.Radio(choices=["off", "median"], value="off", label="Mask Bbox Smoothing")
                mask_bbox_smoothing_window = gr.Slider(label="Mask Bbox Smoothing Window", minimum=1, maximum=15, step=2, value=5)
                mask_bbox_max_scale_delta = gr.Slider(label="Mask Bbox Max Scale Delta", minimum=0.0, maximum=0.3, step=0.01, value=0.08)
                save_mask_bbox_stats = gr.Checkbox(label="Save Mask Bbox Stats", value=False)

            # Right Column - Settings & Output
            with gr.Column(scale=1):
                gr.Markdown("### ✍️ Captions")
                video_caption = gr.Textbox(
                    label="Video Caption",
                    placeholder="Describe the edited video...",
                    lines=3
                )
                enhance_video_btn = gr.Button("✨ Enhance Video Caption (Optional)", size="sm")

                target_region_caption = gr.Textbox(
                    label="Target Region Caption",
                    placeholder="Describe the target object...",
                    lines=2
                )
                enhance_target_btn = gr.Button("✨ Generate Target Caption (Optional)", size="sm")

                gr.Markdown("### ⚙️ Settings")
                with gr.Accordion("Advanced Settings", open=False):
                    seed_param = gr.Number(label="Seed (-1 for random)", value=42, minimum=-1)
                    cfg_scale = gr.Slider(label="CFG Scale", value=6.0, minimum=1.0, maximum=10.0, step=0.1)
                    dilate_size = gr.Slider(label="Dilate Size", value=16, minimum=0, maximum=32, step=1)

                video_info = gr.Textbox(label="Video Info", interactive=False)

                gr.Markdown("### 🎥 Generate")
                inpaint_btn = gr.Button("Inpaint Video", variant="primary", size="lg")
                exact_replace_btn = gr.Button("Exact Object Replacement", variant="primary", size="lg")
                check_status_btn = gr.Button("🔄 Check Processing Status", size="sm")
                load_inpaint_btn = gr.Button("🔄 Load Latest Result", size="sm")
                inpaint_progress = gr.Textbox(label="Processing Progress", interactive=False, lines=3)

                video_output = gr.Video(label="Output Video")

                run_status = gr.HighlightedText(
                    value=[("", "")],
                    label="Status",
                    color_map={
                        "Success": "green",
                        "Error": "red",
                        "Warning": "orange",
                        "Info": "blue"
                    }
                )

        # State variables
        inference_state = gr.State(None)
        video_state = gr.State(None)
        click_state = gr.State([[], []])
        ref_click_state = gr.State([[], []])
        ref_mask_state = gr.State(None)

        # Event handlers
        video_input.change(
            fn=get_frames_from_video,
            inputs=[video_input, video_state],
            outputs=[
                video_caption,
                inference_state,
                video_state,
                video_info,
                template_frame,
                run_status
            ]
        )

        template_frame.select(
            fn=sam_refine,
            inputs=[inference_state, video_state, point_prompt, click_state, run_status],
            outputs=[template_frame, video_state, click_state, run_status],
            show_progress="full",
        )

        clear_button.click(
            fn=clear_click,
            inputs=[inference_state, video_state, run_status],
            outputs=[inference_state, template_frame, click_state, tracking_progress, run_status]
        )

        tracking_btn.click(
            fn=track_video,
            inputs=[inference_state, video_state, run_status],
            outputs=[tracked_video_output, video_state, tracking_progress, run_status],
        )

        load_tracking_btn.click(
            fn=load_latest_tracking,
            inputs=[],
            outputs=[tracked_video_output, tracking_progress, run_status],
        )

        check_tracking_status_btn.click(
            fn=check_processing_status,
            inputs=[],
            outputs=[tracked_video_output, tracking_progress, run_status],
        )

        sketch_generate_btn.click(
            fn=generate_reference_from_sketch,
            inputs=[sketch_image_input, video_state, sketch_label_input, sketch_attrs_input, sketch_candidate_count, reference_num_inference_steps, reference_guidance_scale, reference_controlnet_scale, seed_param, run_status],
            outputs=[sketch_progress, run_status],
        )

        sketch_check_status_btn.click(
            fn=check_sketch_reference_status,
            inputs=[],
            outputs=[sketch_progress, run_status],
        )

        sketch_load_result_btn.click(
            fn=load_latest_sketch_reference,
            inputs=[],
            outputs=[ref_image_input, ref_template, ref_click_state, ref_mask_state, sketch_progress, run_status],
        )

        ref_image_input.select(
            fn=semantic_segment_reference,
            inputs=[ref_image_input, ref_point_prompt, ref_click_state],
            outputs=[ref_template, ref_click_state, ref_mask_state]
        )

        ref_clear_button.click(
            fn=lambda ref: (ref, [[], []], None),
            inputs=[ref_image_input],
            outputs=[ref_template, ref_click_state, ref_mask_state]
        )

        inpaint_btn.click(
            fn=inpaint_video_background,
            inputs=[video_state, video_caption, target_region_caption, run_status, seed_param, cfg_scale, dilate_size],
            outputs=[video_output, inpaint_progress, run_status],
        )

        exact_replace_btn.click(
            fn=exact_replace_video_background,
            inputs=[video_state, ref_image_input, ref_mask_state, video_caption, target_region_caption, run_status, seed_param, cfg_scale, dilate_size, anydoor_guidance_scale, reference_motion_guide, guide_dilate_size, mask_bbox_smoothing, mask_bbox_smoothing_window, mask_bbox_max_scale_delta, save_mask_bbox_stats],
            outputs=[video_output, inpaint_progress, run_status],
        )

        check_status_btn.click(
            fn=check_processing_status,
            inputs=[],
            outputs=[video_output, inpaint_progress, run_status],
        )

        load_inpaint_btn.click(
            fn=load_latest_inpaint,
            inputs=[],
            outputs=[video_output, inpaint_progress, run_status],
        )

        enhance_video_btn.click(
            fn=enhance_video_prompt,
            inputs=[video_caption],
            outputs=[video_caption]
        )

        enhance_target_btn.click(
            fn=enhance_target_prompt,
            inputs=[target_region_caption, video_state],
            outputs=[target_region_caption]
        )

    # Launch
    iface.queue(max_size=10)
    iface.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=False,
        allowed_paths=[GRADIO_TEMP_DIR]
    )


if __name__ == "__main__":
    main()
