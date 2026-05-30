import importlib
import os
import sys
from contextlib import contextmanager

import numpy as np
import torch
from PIL import Image


DEFAULT_ANYDOOR_ROOT = "/root/autodl-tmp/AnyDoor"


@contextmanager
def _working_directory(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _load_anydoor_module(anydoor_root=DEFAULT_ANYDOOR_ROOT):
    if not os.path.isdir(anydoor_root):
        raise FileNotFoundError(f"AnyDoor repo not found: {anydoor_root}")

    inference_config = os.path.join(anydoor_root, "configs", "inference.yaml")
    if not os.path.exists(inference_config):
        raise FileNotFoundError(f"AnyDoor inference config not found: {inference_config}")

    if anydoor_root not in sys.path:
        sys.path.insert(0, anydoor_root)

    with _working_directory(anydoor_root):
        return importlib.import_module("run_inference")


def _pil_to_rgb_array(image):
    if image is None:
        raise ValueError("Image is required")
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _mask_to_binary_array(mask, size=None):
    if mask is None:
        raise ValueError("Mask is required")
    if not isinstance(mask, Image.Image):
        mask = Image.fromarray(np.asarray(mask))
    if size is not None:
        mask = mask.resize(size)
    mask_arr = np.asarray(mask.convert("L"))
    return (mask_arr > 127).astype(np.uint8)


def replace_first_frame_with_anydoor(
    target_image,
    target_mask,
    reference_image,
    reference_mask,
    guidance_scale=5.0,
    anydoor_root=DEFAULT_ANYDOOR_ROOT,
):
    anydoor = _load_anydoor_module(anydoor_root)
    if hasattr(anydoor, "model"):
        anydoor.model = anydoor.model.cuda()

    tar_image = _pil_to_rgb_array(target_image)
    ref_image = _pil_to_rgb_array(reference_image)
    tar_mask = _mask_to_binary_array(target_mask, size=(tar_image.shape[1], tar_image.shape[0]))
    ref_mask = _mask_to_binary_array(reference_mask, size=(ref_image.shape[1], ref_image.shape[0]))

    if tar_mask.max() == 0:
        raise ValueError("Target video mask is empty")
    if ref_mask.max() == 0:
        raise ValueError("Reference image mask is empty")

    try:
        generated = anydoor.inference_single_image(
            ref_image=ref_image,
            ref_mask=ref_mask,
            tar_image=tar_image.copy(),
            tar_mask=tar_mask,
            guidance_scale=float(guidance_scale),
        )
    finally:
        if hasattr(anydoor, "model"):
            anydoor.model = anydoor.model.cpu()
        torch.cuda.empty_cache()

    return Image.fromarray(np.asarray(generated).clip(0, 255).astype(np.uint8)).convert("RGB")
