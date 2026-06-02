from __future__ import annotations

import cv2
import scipy.ndimage
import numpy as np
from PIL import Image


def _to_rgb_array(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.asarray(Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB"), dtype=np.uint8)


def _to_mask_array(mask) -> np.ndarray:
    if isinstance(mask, Image.Image):
        arr = np.asarray(mask.convert("L"), dtype=np.uint8)
    else:
        arr = np.asarray(mask, dtype=np.uint8)
        if arr.ndim == 3:
            arr = arr[..., 0]
    return arr




def mask_bbox(mask):
    arr = (_to_mask_array(mask) > 0).astype(np.uint8)
    ys, xs = np.where(arr > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def mask_geometry(mask):
    arr = (_to_mask_array(mask) > 0).astype(np.uint8)
    bbox = mask_bbox(arr)
    if bbox is None:
        return {
            "bbox": None,
            "width": 0,
            "height": 0,
            "aspect": 0.0,
            "area_ratio": 0.0,
        }
    x0, y0, x1, y1 = bbox
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    return {
        "bbox": bbox,
        "width": width,
        "height": height,
        "aspect": float(width / max(height, 1)),
        "area_ratio": float(arr.sum() / arr.size),
    }


def _scaled_center_crop_mask(mask, scale=0.82):
    target = (_to_mask_array(mask) > 0).astype(np.uint8)
    bbox = mask_bbox(target)
    if bbox is None:
        return target.astype(np.uint8) * 255
    x0, y0, x1, y1 = bbox
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    scale = min(1.0, max(0.05, float(scale)))
    crop_w = max(1, int(round(width * scale)))
    crop_h = max(1, int(round(height * scale)))
    cx = (x0 + x1 + 1) / 2.0
    cy = (y0 + y1 + 1) / 2.0
    cx0 = max(0, int(round(cx - crop_w / 2.0)))
    cy0 = max(0, int(round(cy - crop_h / 2.0)))
    cx1 = min(target.shape[1], cx0 + crop_w)
    cy1 = min(target.shape[0], cy0 + crop_h)
    crop = np.zeros_like(target, dtype=np.uint8)
    crop[cy0:cy1, cx0:cx1] = 1
    return (np.logical_and(target > 0, crop > 0).astype(np.uint8) * 255)


def select_anydoor_target_mask(
    reference_mask,
    video_target_mask,
    output_size=(720, 480),
    min_overlap_ratio=0.25,
    min_containment_ratio=0.98,
    fallback_scale=0.82,
    preserve_full_frame_reference=False,
):
    """Choose a full-frame AnyDoor placement mask in video coordinates.

    Frame-shaped sketch references can provide a smaller object mask. Use it
    only when it is already contained by the tracked video target. Otherwise,
    build a smaller placement mask from the tracked video target itself so the
    AnyDoor mask cannot drift outside the original edit region.
    """
    target = (_to_mask_array(video_target_mask) > 0).astype(np.uint8)
    ref = (_to_mask_array(reference_mask) > 0).astype(np.uint8)
    expected_shape = (int(output_size[1]), int(output_size[0]))
    if target.shape != expected_shape:
        raise ValueError(f"video_target_mask must have shape {expected_shape}, got {target.shape}")
    target_area = max(int(target.sum()), 1)
    if ref.shape == expected_shape and ref.any():
        overlap = np.logical_and(ref > 0, target > 0).sum()
        ref_area = max(int(ref.sum()), 1)
        containment = overlap / ref_area
        if preserve_full_frame_reference and containment >= float(min_containment_ratio):
            return ref.astype(np.uint8) * 255
        if containment >= float(min_containment_ratio) and ref_area < target_area * 0.95:
            return ref.astype(np.uint8) * 255
        if containment >= float(min_overlap_ratio) or target.any():
            return _scaled_center_crop_mask(target, scale=fallback_scale)
    if target.any():
        return _scaled_center_crop_mask(target, scale=fallback_scale)
    return target.astype(np.uint8) * 255


def select_anydoor_target_mask_for_strategy(
    reference_mask,
    video_target_mask,
    reference_strategy,
    output_size=(720, 480),
):
    """Select the AnyDoor target mask without changing LaMa-clean behavior.

    mask_twist is the strict same-shape path: AnyDoor and CogVideoX must see the
    same frame-0 target region. LaMa-clean keeps the smaller-object placement
    behavior implemented by select_anydoor_target_mask().
    """
    target = (_to_mask_array(video_target_mask) > 0).astype(np.uint8)
    expected_shape = (int(output_size[1]), int(output_size[0]))
    if target.shape != expected_shape:
        raise ValueError(f"video_target_mask must have shape {expected_shape}, got {target.shape}")
    if reference_strategy == "mask_twist":
        return target.astype(np.uint8) * 255
    return select_anydoor_target_mask(reference_mask, video_target_mask, output_size=output_size)


def prepare_anydoor_target_image(
    target_image,
    video_target_mask,
    anydoor_target_mask,
    inpaint_radius=7,
    erase_dilate_size=21,
    erase_top_scale=0.45,
    erase_side_scale=0.35,
    erase_bottom_scale=0.08,
):
    """Remove the original target object before placing a smaller AnyDoor object.

    When the AnyDoor placement mask is smaller than the clicked video mask, the
    unedited target crop still contains the old large object. AnyDoor can then
    preserve or regenerate that large object. Pre-inpainting the original target
    mask gives AnyDoor a cleaner background around the smaller placement area.
    """
    image = _to_rgb_array(target_image).copy()
    video_mask = (_to_mask_array(video_target_mask) > 0).astype(np.uint8)
    placement_mask = (_to_mask_array(anydoor_target_mask) > 0).astype(np.uint8)
    if video_mask.shape != placement_mask.shape or video_mask.shape != image.shape[:2]:
        return image
    video_area = int(video_mask.sum())
    placement_area = int(placement_mask.sum())
    if video_area == 0 or placement_area == 0 or placement_area >= video_area * 0.95:
        return image
    inpaint_mask = (video_mask * 255).astype(np.uint8)
    kernel_size = max(1, int(erase_dilate_size))
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        inpaint_mask = cv2.dilate(inpaint_mask, kernel, iterations=1)
    bbox = mask_bbox(inpaint_mask)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        width = x1 - x0 + 1
        height = y1 - y0 + 1
        pad = max(0, kernel_size // 2)
        x0 = max(0, x0 - pad - int(round(width * float(erase_side_scale))))
        y0 = max(0, y0 - pad - int(round(height * float(erase_top_scale))))
        x1 = min(image.shape[1] - 1, x1 + pad + int(round(width * float(erase_side_scale))))
        y1 = min(image.shape[0] - 1, y1 + pad + int(round(height * float(erase_bottom_scale))))
        inpaint_mask[y0 : y1 + 1, x0 : x1 + 1] = 255
    return cv2.inpaint(image, inpaint_mask, int(inpaint_radius), cv2.INPAINT_TELEA)


def build_anydoor_background_inpaint_mask(
    video_target_mask,
    preserve_mask=None,
    dilate_size=0,
):
    """Build the FLUX mask for cleaning the AnyDoor target frame.

    The default is the tracked frame-0 video target mask. Do not expand it:
    FLUX.1 Fill should see the original object region and infer the missing
    background from surrounding pixels. `preserve_mask` is kept only as an
    explicit experimental option for ring cleanup and is not used by defaults.
    """
    mask = (_to_mask_array(video_target_mask) > 0).astype(np.uint8) * 255
    if mask.max() == 0:
        return mask
    kernel_size = max(0, int(dilate_size))
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    if preserve_mask is not None:
        preserve = _to_mask_array(preserve_mask)
        if preserve.shape != mask.shape:
            preserve = cv2.resize(preserve, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        background_ring = mask.copy()
        background_ring[preserve > 0] = 0
        return background_ring.astype(np.uint8)
    return mask.astype(np.uint8)


def build_lama_background_inpaint_mask(video_target_mask, mode="rect", padding=24, dilate_size=31):
    """Build a stronger object-removal mask for LaMa background cleanup.

    LaMa is prompt-free, so an expanded rectangular or dilated mask is often
    better for removing residual object rims and transparent reflections.
    """
    mask = (_to_mask_array(video_target_mask) > 0).astype(np.uint8) * 255
    if mask.max() == 0:
        return mask
    mode = str(mode or "rect").lower()
    if mode == "exact":
        return mask.astype(np.uint8)
    if mode == "dilate":
        kernel_size = max(1, int(dilate_size))
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        return mask.astype(np.uint8)
    if mode != "rect":
        raise ValueError(f"Unsupported LaMa background mask mode: {mode}")
    bbox = mask_bbox(mask)
    if bbox is None:
        return mask.astype(np.uint8)
    x0, y0, x1, y1 = bbox
    pad = max(0, int(padding))
    rect = np.zeros_like(mask, dtype=np.uint8)
    rect[
        max(0, y0 - pad) : min(mask.shape[0], y1 + pad + 1),
        max(0, x0 - pad) : min(mask.shape[1], x1 + pad + 1),
    ] = 255
    return rect


def _bbox_to_center_size(bbox):
    x0, y0, x1, y1 = bbox
    return {
        "center_x": (x0 + x1) / 2.0,
        "center_y": (y0 + y1) / 2.0,
        "width": x1 - x0 + 1,
        "height": y1 - y0 + 1,
    }


def _center_size_to_bbox(center_x, center_y, width, height, image_size):
    image_w, image_h = image_size
    width = max(1, int(round(width)))
    height = max(1, int(round(height)))
    width = min(width, image_w)
    height = min(height, image_h)
    x0 = int(round(center_x - (width - 1) / 2.0))
    y0 = int(round(center_y - (height - 1) / 2.0))
    x0 = max(0, min(image_w - width, x0))
    y0 = max(0, min(image_h - height, y0))
    return (x0, y0, x0 + width - 1, y0 + height - 1)


def mask_bbox_trajectory(masks):
    stats = []
    for idx, mask in enumerate(masks):
        arr = (_to_mask_array(mask) > 0).astype(np.uint8)
        bbox = mask_bbox(arr)
        if bbox is None:
            stats.append({
                "frame": idx,
                "bbox": None,
                "width": 0,
                "height": 0,
                "center_x": None,
                "center_y": None,
                "area": 0,
            })
            continue
        geom = _bbox_to_center_size(bbox)
        stats.append({
            "frame": idx,
            "bbox": [int(v) for v in bbox],
            "width": int(geom["width"]),
            "height": int(geom["height"]),
            "center_x": float(geom["center_x"]),
            "center_y": float(geom["center_y"]),
            "area": int(arr.sum()),
        })
    return stats


def smooth_mask_bboxes(bboxes, image_size, window=5, max_scale_delta=0.08):
    if window <= 1:
        return [None if box is None else tuple(int(v) for v in box) for box in bboxes]
    valid = [box for box in bboxes if box is not None]
    if not valid:
        return [None for _ in bboxes]

    half = max(1, int(window) // 2)
    centers_x = []
    centers_y = []
    widths = []
    heights = []
    last = valid[0]
    for box in bboxes:
        if box is None:
            box = last
        last = box
        geom = _bbox_to_center_size(box)
        centers_x.append(geom["center_x"])
        centers_y.append(geom["center_y"])
        widths.append(geom["width"])
        heights.append(geom["height"])

    smoothed = []
    prev_w = widths[0]
    prev_h = heights[0]
    for idx in range(len(bboxes)):
        lo = max(0, idx - half)
        hi = min(len(bboxes), idx + half + 1)
        cx = float(np.median(centers_x[lo:hi]))
        cy = float(np.median(centers_y[lo:hi]))
        width = float(np.median(widths[lo:hi]))
        height = float(np.median(heights[lo:hi]))
        if idx > 0 and max_scale_delta > 0:
            width = min(max(width, prev_w * (1.0 - max_scale_delta)), prev_w * (1.0 + max_scale_delta))
            height = min(max(height, prev_h * (1.0 - max_scale_delta)), prev_h * (1.0 + max_scale_delta))
        bbox = _center_size_to_bbox(cx, cy, width, height, image_size)
        smoothed.append(bbox)
        prev_w = bbox[2] - bbox[0] + 1
        prev_h = bbox[3] - bbox[1] + 1
    return smoothed


def fit_mask_to_bbox(mask, target_bbox, output_size=(720, 480)):
    arr = (_to_mask_array(mask) > 0).astype(np.uint8) * 255
    source_bbox = mask_bbox(arr)
    if source_bbox is None:
        raise ValueError("Cannot fit an empty mask")
    sx0, sy0, sx1, sy1 = source_bbox
    tx0, ty0, tx1, ty1 = target_bbox
    crop = arr[sy0 : sy1 + 1, sx0 : sx1 + 1]
    target_w = tx1 - tx0 + 1
    target_h = ty1 - ty0 + 1
    scale = min(target_w / max(crop.shape[1], 1), target_h / max(crop.shape[0], 1))
    new_w = max(1, int(round(crop.shape[1] * scale)))
    new_h = max(1, int(round(crop.shape[0] * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((output_size[1], output_size[0]), dtype=np.uint8)
    cx = int(round((tx0 + tx1) / 2.0))
    cy = int(round((ty0 + ty1) / 2.0))
    x0 = max(0, min(output_size[0] - new_w, cx - new_w // 2))
    y0 = max(0, min(output_size[1] - new_h, cy - new_h // 2))
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas




def _crop_to_mask(image, mask, padding=0):
    image_arr = _to_rgb_array(image)
    mask_arr = (_to_mask_array(mask) > 0).astype(np.uint8) * 255
    bbox = mask_bbox(mask_arr)
    if bbox is None:
        raise ValueError("Cannot crop an empty object mask")
    x0, y0, x1, y1 = bbox
    pad = max(0, int(padding))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(image_arr.shape[1] - 1, x1 + pad)
    y1 = min(image_arr.shape[0] - 1, y1 + pad)
    return image_arr[y0 : y1 + 1, x0 : x1 + 1], mask_arr[y0 : y1 + 1, x0 : x1 + 1]


def composite_object_guide_frames(frames, object_image, object_mask, target_masks, target_bboxes=None, feather_radius=4):
    object_crop, object_alpha = _crop_to_mask(object_image, object_mask, padding=feather_radius)
    guides = []
    if target_bboxes is not None and len(target_bboxes) != len(target_masks):
        raise ValueError("target_bboxes must have the same length as target_masks")
    for idx, (frame, target_mask) in enumerate(zip(frames, target_masks)):
        frame_arr = _to_rgb_array(frame).copy()
        target = (_to_mask_array(target_mask) > 0).astype(np.uint8) * 255
        target_bbox = target_bboxes[idx] if target_bboxes is not None else mask_bbox(target)
        if target_bbox is None:
            guides.append(Image.fromarray(frame_arr).convert("RGB"))
            continue
        tx0, ty0, tx1, ty1 = target_bbox
        target_w = tx1 - tx0 + 1
        target_h = ty1 - ty0 + 1
        scale = min(target_w / max(object_crop.shape[1], 1), target_h / max(object_crop.shape[0], 1))
        new_w = max(1, int(round(object_crop.shape[1] * scale)))
        new_h = max(1, int(round(object_crop.shape[0] * scale)))
        resized_rgb = cv2.resize(object_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        resized_alpha = cv2.resize(object_alpha, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        cx = int(round((tx0 + tx1) / 2.0))
        cy = int(round((ty0 + ty1) / 2.0))
        x0 = max(0, min(frame_arr.shape[1] - new_w, cx - new_w // 2))
        y0 = max(0, min(frame_arr.shape[0] - new_h, cy - new_h // 2))
        if feather_radius and feather_radius > 0:
            ksize = max(3, int(feather_radius) * 2 + 1)
            alpha_f = cv2.GaussianBlur((resized_alpha > 0).astype(np.float32), (ksize, ksize), 0)
            alpha_max = float(alpha_f.max())
            if alpha_max > 0:
                alpha_f = alpha_f / alpha_max
            alpha = alpha_f[:, :, None]
        else:
            alpha = (resized_alpha > 0).astype(np.float32)[:, :, None]
        patch = frame_arr[y0 : y0 + new_h, x0 : x0 + new_w]
        patch[:] = np.clip(alpha * resized_rgb.astype(np.float32) + (1.0 - alpha) * patch.astype(np.float32), 0, 255).astype(np.uint8)
        guides.append(Image.fromarray(frame_arr).convert("RGB"))
    return guides


def build_union_edit_masks(target_masks, object_masks):
    if len(target_masks) != len(object_masks):
        raise ValueError("target_masks and object_masks must have the same length")

    union_masks = []
    for target_mask, object_mask in zip(target_masks, object_masks):
        target = _to_mask_array(target_mask) > 0
        obj = _to_mask_array(object_mask) > 0
        union = np.logical_or(target, obj).astype(np.uint8) * 255
        union_rgb = np.stack([union, union, union], axis=-1)
        union_masks.append(Image.fromarray(union_rgb).convert("RGB"))
    return union_masks

def build_edge_refine_masks(target_masks, inner_erode_iter=4, outer_dilate_iter=8):
    edge_masks = []
    for mask in target_masks:
        arr = (_to_mask_array(mask) > 0)
        outer = scipy.ndimage.binary_dilation(arr, iterations=outer_dilate_iter)
        inner = scipy.ndimage.binary_erosion(arr, iterations=inner_erode_iter)
        edge = np.logical_and(outer, np.logical_not(inner)).astype(np.uint8) * 255
        edge_masks.append(Image.fromarray(edge).convert("RGB"))
    return edge_masks


def build_masked_video_frames(images, masks, mask_background: bool = False):
    if len(images) != len(masks):
        raise ValueError("images and masks must have the same length")

    masked_frames = []
    for image, mask in zip(images, masks):
        image_arr = _to_rgb_array(image)
        mask_arr = _to_mask_array(mask)
        foreground = mask_arr >= 128
        if mask_background:
            masked_arr = np.where(foreground[:, :, None], image_arr, 0)
        else:
            masked_arr = np.where(foreground[:, :, None], 0, image_arr)
        masked_frames.append(Image.fromarray(masked_arr.astype(np.uint8)).convert("RGB"))
    return masked_frames


def build_first_frame_ground_truth_masks(masks, mask_background: bool = False):
    if not masks:
        return []

    processed = []
    for mask in masks:
        mask_arr = _to_mask_array(mask)
        processed.append(Image.fromarray(mask_arr).convert("RGB"))
    return processed


def finalize_propagation_sequence(propagated_frames, fallback_first_frame=None):
    frames = np.asarray(propagated_frames)
    if frames.ndim != 4:
        raise ValueError("propagated_frames must be a 4D frame tensor")
    if frames.dtype != np.uint8:
        frames = np.clip(frames * 255, 0, 255).astype(np.uint8)

    if frames.shape[0] == 49:
        return frames

    if frames.shape[0] == 48:
        if fallback_first_frame is None:
            raise ValueError("fallback_first_frame is required for 48-frame propagated tails")
        first = _to_rgb_array(fallback_first_frame)
        return np.concatenate([first[None, ...], frames], axis=0)

    raise ValueError(f"Unexpected propagated frame count: {frames.shape[0]}")
