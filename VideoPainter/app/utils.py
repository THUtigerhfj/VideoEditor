import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
GRADIO_TEMP_DIR = os.environ.get("GRADIO_TEMP_DIR", os.path.join(APP_DIR, "tmp_gradio"))
os.environ["GRADIO_TEMP_DIR"] = GRADIO_TEMP_DIR
import warnings
warnings.filterwarnings("ignore")
import argparse
from typing import Literal
import json
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogvideoXBranchModel,
    CogVideoXTransformer3DModel,
    CogVideoXI2VDualInpaintPipeline,
    CogVideoXI2VDualInpaintAnyLPipeline,
    FluxFillPipeline
)
import cv2
from openai import OpenAI
from diffusers.utils import export_to_video, load_image, load_video
from PIL import Image
from safetensors import safe_open
from frame_conditioning import build_first_frame_ground_truth_masks, build_lama_background_inpaint_mask, build_masked_video_frames, finalize_propagation_sequence
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

# The VideoPainterID reference pipeline uses a non-zero previous-clip weight to
# preserve object identity after first-frame replacement.
REFERENCE_PREV_CLIP_WEIGHT = 0.5

def load_model(
    model_path,
    inpainting_branch,
    img_inpainting_model,
    id_adapter,
    device="cuda:0",
    dtype=torch.bfloat16
):
    cpu_offload = os.environ.get("VIDEOPAINTER_CPU_OFFLOAD", "").lower()
    model_device = "cpu" if cpu_offload else device

    branch = CogvideoXBranchModel.from_pretrained(inpainting_branch, torch_dtype=dtype).to(model_device, dtype=dtype)
    
    # load the transformer
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=dtype,
        id_pool_resample_learnable=True,
    ).to(model_device, dtype=dtype)

    pipe = CogVideoXI2VDualInpaintAnyLPipeline.from_pretrained(
        model_path,
        branch=branch,
        transformer=transformer,
        torch_dtype=dtype,
    )

    pipe.load_lora_weights(
        id_adapter, 
        weight_name="pytorch_lora_weights.safetensors", 
        adapter_name="test_1",
        target_modules=["transformer"]
        )
    # pipe.fuse_lora(lora_scale=1 / lora_rank)

    list_adapters_component_wise = pipe.get_list_adapters()
    print(f"list_adapters_component_wise: {list_adapters_component_wise}")

    pipe.text_encoder.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.branch.requires_grad_(False)

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to(device)
    if cpu_offload == "sequential":
        pipe.enable_sequential_cpu_offload(gpu_id=0)
    elif cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=0)
    else:
        pipe.to(device)
    # Keep the default app inference path aligned with the original demo:
    # no VAE slicing/tiling unless the caller changes the model explicitly.

    pipe_img_inpainting = None
    if img_inpainting_model:
        model_index = os.path.join(img_inpainting_model, "model_index.json")
        if not os.path.exists(model_index):
            raise FileNotFoundError(
                f"Image inpainting model not found or incomplete: {img_inpainting_model}. "
                "Leave --img_inpainting_model empty for exact object replacement, "
                "or provide a local FLUX.1-Fill-dev directory for text-prompted editing."
            )
        pipe_img_inpainting = img_inpainting_model
    return pipe, pipe_img_inpainting


def build_reference_propagation_prompt(video_caption, target_object_caption):
    video_caption = str(video_caption or "").strip()
    target_object_caption = str(target_object_caption or "").strip()
    if not target_object_caption:
        return video_caption
    if not video_caption:
        return f"Focus on keeping the replacement object as {target_object_caption}."
    return f"{video_caption} Focus on keeping the replacement object as {target_object_caption}."


def run_flux_fill_inpaint(
    image,
    mask_image,
    pipe_img_inpainting,
    prompt,
    seed=42,
    guidance_scale=30.0,
    num_inference_steps=50,
    max_sequence_length=512,
    device="cuda",
    video_pipe=None,
):
    """Run FLUX.1 Fill on one image/mask pair and return an RGB PIL image."""
    if pipe_img_inpainting is None:
        raise ValueError("FLUX background inpainting requires --img_inpainting_model")

    image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
    mask_image = mask_image if isinstance(mask_image, Image.Image) else Image.fromarray(np.asarray(mask_image).astype(np.uint8)).convert("L")

    flux_pipe = None
    cpu_offload = os.environ.get("VIDEOPAINTER_CPU_OFFLOAD", "").lower()
    moved_video_pipe = isinstance(pipe_img_inpainting, str) and video_pipe is not None and not cpu_offload
    if moved_video_pipe:
        video_pipe.to("cpu")
        torch.cuda.empty_cache()

    if isinstance(pipe_img_inpainting, str):
        flux_pipe = FluxFillPipeline.from_pretrained(pipe_img_inpainting, torch_dtype=torch.bfloat16)
        if cpu_offload:
            flux_pipe.enable_model_cpu_offload(gpu_id=0)
        else:
            flux_pipe.to(device)
    else:
        flux_pipe = pipe_img_inpainting.to(device)

    try:
        result = flux_pipe(
            prompt=str(prompt or "continuous surrounding background, same lighting, perspective, reflections, focus, and texture"),
            image=image,
            mask_image=mask_image,
            height=image.size[1],
            width=image.size[0],
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(num_inference_steps),
            max_sequence_length=int(max_sequence_length),
            generator=torch.Generator("cpu").manual_seed(int(seed)),
        ).images[0]
    finally:
        if flux_pipe is not None:
            if hasattr(flux_pipe, "maybe_free_model_hooks"):
                flux_pipe.maybe_free_model_hooks()
            flux_pipe.to("cpu")
        if isinstance(pipe_img_inpainting, str) and flux_pipe is not None:
            del flux_pipe
        torch.cuda.empty_cache()
        if moved_video_pipe:
            video_pipe.to(device)

    return result.convert("RGB")


def run_lama_inpaint(
    image,
    mask_image,
    lama_model,
    device="cuda",
    video_pipe=None,
):
    """Run LaMa object-removal inpainting on one image/mask pair."""
    if not lama_model:
        raise ValueError("LaMa background inpainting requires --lama_model")
    if not os.path.exists(str(lama_model)):
        raise FileNotFoundError(f"LaMa model not found: {lama_model}")

    image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
    mask_image = mask_image if isinstance(mask_image, Image.Image) else Image.fromarray(np.asarray(mask_image).astype(np.uint8)).convert("L")

    cpu_offload = os.environ.get("VIDEOPAINTER_CPU_OFFLOAD", "").lower()
    moved_video_pipe = video_pipe is not None and not cpu_offload
    if moved_video_pipe:
        video_pipe.to("cpu")
        torch.cuda.empty_cache()

    old_lama_model = os.environ.get("LAMA_MODEL")
    os.environ["LAMA_MODEL"] = str(lama_model)
    try:
        from simple_lama_inpainting import SimpleLama

        lama = SimpleLama(device=torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"))
        result = lama(image, mask_image)
    finally:
        if old_lama_model is None:
            os.environ.pop("LAMA_MODEL", None)
        else:
            os.environ["LAMA_MODEL"] = old_lama_model
        if "lama" in locals():
            lama.model.to("cpu")
            del lama
        torch.cuda.empty_cache()
        if moved_video_pipe:
            video_pipe.to(device)

    return result.convert("RGB")


def run_lama_video_inpaint(
    images,
    masks,
    lama_model,
    mask_mode="rect",
    mask_padding=24,
    mask_dilate=31,
    device="cuda",
    progress_callback=None,
    output_dir=None,
):
    """Run LaMa cleanup on a sequence while preserving the original masks."""
    if not lama_model:
        raise ValueError("LaMa video cleanup requires --lama_model")
    if not os.path.exists(str(lama_model)):
        raise FileNotFoundError(f"LaMa model not found: {lama_model}")
    if len(images) != len(masks):
        raise ValueError(f"images/masks length mismatch: {len(images)} vs {len(masks)}")

    old_lama_model = os.environ.get("LAMA_MODEL")
    os.environ["LAMA_MODEL"] = str(lama_model)
    try:
        from simple_lama_inpainting import SimpleLama

        lama = SimpleLama(device=torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"))
        cleaned_images = []
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        for idx, (image, mask) in enumerate(zip(images, masks)):
            image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
            cleanup_mask = build_lama_background_inpaint_mask(
                mask,
                mode=mask_mode,
                padding=mask_padding,
                dilate_size=mask_dilate,
            )
            cleanup_mask_pil = Image.fromarray(cleanup_mask).convert("L")
            cleaned = lama(image.convert("RGB"), cleanup_mask_pil).convert("RGB")
            cleaned_images.append(cleaned)
            if output_dir and idx in {0, 1, len(images) - 1}:
                cleaned.save(os.path.join(output_dir, f"lama_conditioning_frame_{idx:03d}.png"))
                cleanup_mask_pil.save(os.path.join(output_dir, f"lama_conditioning_mask_{idx:03d}.png"))
            if progress_callback:
                progress_callback(idx + 1, len(images), f"LaMa cleaned conditioning frame {idx + 1}/{len(images)}")
    finally:
        if old_lama_model is None:
            os.environ.pop("LAMA_MODEL", None)
        else:
            os.environ["LAMA_MODEL"] = old_lama_model
        if "lama" in locals():
            lama.model.to("cpu")
            del lama
        torch.cuda.empty_cache()

    return cleaned_images


def generate_frames(
        images,
        masks,
        pipe,
        pipe_img_inpainting,
        prompt,
        image_inpainting_prompt,
        seed=42,
        cfg_scale=6.0,
        dilate_size=16,
        first_frame_override=None,
        progress_callback=None,
        return_full_sequence=False,
        prev_clip_weight=0.0,
        id_pool_resample_learnable=None,
        guide_images=None,
        guide_mode="none",
        guide_dilate_size=None,
        conditioning_video_mode="full_video",
    ):
    """
    Generate inpainted video frames.

    Args:
        progress_callback: Optional callable that takes (step, total_steps, message) parameters
                          for progress updates during processing.
    """
    num_frames = 49
    total_steps = 100  # For progress reporting
    current_step = 0

    def report_progress(step, total, message):
        """Report progress via callback or print."""
        print(f"Progress: {step}/{total} - {message}")
        if progress_callback:
            progress_callback(step, total, message)

    if len(images) != num_frames:
        raise ValueError(f"Original VideoPainter demo inference expects 49 frames, got {len(images)}")

    # save the first frame
    images[0].save(f"{GRADIO_TEMP_DIR}/inpaint/first_frame.png")
    masks[0].save(f"{GRADIO_TEMP_DIR}/inpaint/first_mask.png")
    masks[-1].save(f"{GRADIO_TEMP_DIR}/inpaint/last_mask.png")
    # for i in range(len(masks)):
    #     masks[i].save(f"{GRADIO_TEMP_DIR}/inpaint/mask_{i:03d}.png")

    effective_dilate_size = dilate_size
    if guide_images is not None and guide_mode != "none" and guide_dilate_size is not None:
        effective_dilate_size = guide_dilate_size

    report_progress(current_step, total_steps, f"Dilating masks ({effective_dilate_size}px)...")
    current_step += 5

    print(f"Dilating the mask with size {effective_dilate_size}...")
    for i in range(len(masks)):
        kernel_size = max(1, int(effective_dilate_size))
        mask = cv2.dilate(np.array(masks[i]), np.ones((kernel_size, kernel_size)))
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask)
        masks[i] = mask

    masks[0].save(f"{GRADIO_TEMP_DIR}/inpaint/first_mask_dilate.png")
    masks[-1].save(f"{GRADIO_TEMP_DIR}/inpaint/last_mask_dilate.png")

    report_progress(current_step, total_steps, "Mask dilation complete")
    current_step += 5

    if first_frame_override is None:
        if pipe_img_inpainting is None:
            raise ValueError(
                "Text-prompted inpainting requires --img_inpainting_model. "
                "Exact object replacement supplies an edited first frame from AnyDoor and does not need FLUX."
            )
        print(f"Image inpainting prompt: {image_inpainting_prompt}")

        report_progress(current_step, total_steps, "Loading FLUX model for first frame inpainting...")
        current_step += 5

        report_progress(current_step, total_steps, "Running FLUX inpainting (50 steps)...")
        current_step += 20

        image_inpainting = run_flux_fill_inpaint(
            image=images[0],
            mask_image=masks[0],
            pipe_img_inpainting=pipe_img_inpainting,
            prompt=image_inpainting_prompt,
            seed=seed,
            guidance_scale=30,
            num_inference_steps=50,
            max_sequence_length=512,
            device="cuda",
            video_pipe=pipe,
        )
        images[0] = image_inpainting
        print(f"Image inpainting done! {np.array(images[0]).shape}")
        report_progress(current_step, total_steps, "FLUX inpainting complete")
        current_step += 5
    else:
        report_progress(current_step, total_steps, "Using AnyDoor-edited first frame")
        current_step += 5
        images[0] = first_frame_override.resize(images[0].size).convert("RGB")
        print(f"Using externally edited first frame: {np.array(images[0]).shape}")

    if id_pool_resample_learnable is None:
        id_pool_resample_learnable = first_frame_override is not None

    conditioning_masks = build_first_frame_ground_truth_masks(masks, mask_background=False) if first_frame_override is not None else masks
    conditioning_images = images
    if guide_images is not None and guide_mode != "none":
        if len(guide_images) != len(images):
            raise ValueError(f"guide_images must have {len(images)} frames, got {len(guide_images)}")
        conditioning_images = [img.resize(images[0].size).convert("RGB") for img in guide_images]
        conditioning_images[0] = images[0].copy()
    if conditioning_video_mode not in {"masked_video", "full_video"}:
        raise ValueError(f"Unsupported conditioning_video_mode: {conditioning_video_mode}")
    if guide_images is not None and guide_mode != "none":
        # Guide frames already carry the replacement object's color/identity.
        # Keep those pixels visible; masks are still passed separately to the video model.
        masked_video = [img.copy() for img in conditioning_images]
    elif conditioning_video_mode == "full_video":
        # Debug/compatibility mode matching the initial reference pipeline: keep original
        # video pixels visible and let the separate mask tensor define edit regions.
        masked_video = [img.copy() for img in conditioning_images]
    else:
        masked_video = build_masked_video_frames(conditioning_images, conditioning_masks, mask_background=False)
    masked_video[0] = images[0].copy()

    # save the first frame (only if FLUX inpainting was used)
    if first_frame_override is None:
        images[0].save(f"{GRADIO_TEMP_DIR}/inpaint/first_frame_inpainted.png")

    report_progress(current_step, total_steps, "Starting CogVideoX propagation (50 steps)...")
    current_step += 5

    # Define callback for CogVideoX pipeline progress
    def callback_on_step_end(pipe, step, timestep, callback_kwargs):
        """Callback function called at each denoising step."""
        progress_pct = 50 + int((step / 50) * 45)  # Steps 5-95 represent the 50 diffusion steps
        report_progress(progress_pct, total_steps, f"CogVideoX step {step}/50")
        return callback_kwargs

    inpaint_outputs = pipe(
        prompt=prompt,
        image=masked_video[0],
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=49,
        use_dynamic_cfg=True,
        guidance_scale=cfg_scale,
        generator=torch.Generator().manual_seed(seed),
        video=masked_video,
        masks=conditioning_masks,
        strength=1.0,
        replace_gt=True,
        mask_add=True,
        stride=int(49 - 0), # int(frames - down_sample_fps), frames,
        prev_clip_weight=prev_clip_weight,
        id_pool_resample_learnable=id_pool_resample_learnable,
        output_type="np",
        callback_on_step_end=callback_on_step_end,
    ).frames[0]
    outputs = inpaint_outputs if return_full_sequence else inpaint_outputs[1:]

    report_progress(total_steps, total_steps, "CogVideoX propagation complete")
    print(f"Video inpainting done! {np.array(outputs).shape}, {np.array(outputs).min()}, {np.array(outputs).max()}")
    torch.cuda.empty_cache()
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--inpainting_branch", type=str, default="")
    parser.add_argument("--img_inpainting_model", type=str, default="../")
    args = parser.parse_args()


    validation_pipeline = load_model(
        model_path=args.model_path,
        inpainting_branch=args.inpainting_branch,
        img_inpainting_model=args.img_inpainting_model
    )
