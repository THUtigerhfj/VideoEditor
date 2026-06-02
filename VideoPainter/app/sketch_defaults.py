REFERENCE_STRATEGY_LAMA = "lama_background"
REFERENCE_STRATEGY_TWIST = "mask_twist"

REFERENCE_STRATEGY_LABELS = {
    REFERENCE_STRATEGY_LAMA: "LaMa background cleanup (default)",
    REFERENCE_STRATEGY_TWIST: "Twist/fit to original video mask",
}

REFERENCE_STRATEGIES = {
    REFERENCE_STRATEGY_LAMA: {
        "shape_conditioned_scribble": True,
        "frame_shaped_reference": True,
        "frame_shaped_reference_object_scale": 0.82,
        "anydoor_pre_inpaint_mode": "lama",
        "reference_propagation_mask_source": "video_target",
        "edit_mask_mode": "propagation",
        "reference_motion_guide": "none",
        "conditioning_video_mode": "lama_cleaned_video",
        "conditioning_lama_mask_padding": 48,
        "dilate_size": 0,
        "guide_dilate_size": 0,
    },
    REFERENCE_STRATEGY_TWIST: {
        "shape_conditioned_scribble": True,
        "frame_shaped_reference": True,
        "frame_shaped_reference_object_scale": 1.0,
        "anydoor_pre_inpaint_mode": "off",
        "reference_propagation_mask_source": "video_target",
        "edit_mask_mode": "propagation",
        "reference_motion_guide": "none",
        "conditioning_video_mode": "full_video",
        "conditioning_lama_mask_padding": 48,
        "dilate_size": 0,
        "guide_dilate_size": 0,
    },
}


def normalize_reference_strategy(value):
    return value if value in REFERENCE_STRATEGIES else REFERENCE_STRATEGY_LAMA


def reference_strategy_settings(value):
    strategy = normalize_reference_strategy(value)
    settings = dict(REFERENCE_STRATEGIES[strategy])
    settings["reference_strategy"] = strategy
    settings["reference_strategy_label"] = REFERENCE_STRATEGY_LABELS[strategy]
    return settings


SKETCH_MODE_DEFAULTS = {
    "reference_strategy": REFERENCE_STRATEGY_LAMA,
    "shape_conditioned_scribble": True,
    "frame_shaped_reference": True,
    "frame_shaped_reference_object_scale": 0.82,
    "sketch_mask_fit_strength": 0.5,
    "mask_contour_weight": 0.6,
    "reference_controlnet_scale": 0.5,
    "reference_num_inference_steps": 30,
    "reference_guidance_scale": 6.5,
    "candidate_count": 3,
    "reference_propagation_mask_source": "video_target",
    "edit_mask_mode": "propagation",
    "reference_motion_guide": "none",
    "anydoor_pre_inpaint_mode": "lama",
    "anydoor_background_prompt": "unoccupied tabletop and laptop keyboard area, continuous desk and laptop surfaces matching the surrounding scene, same lighting, perspective, reflections, focus, and background texture",
    "anydoor_background_mask_dilate": 0,
    "anydoor_background_guidance_scale": 30.0,
    "anydoor_background_num_inference_steps": 50,
    "anydoor_lama_mask_mode": "rect",
    "anydoor_lama_mask_padding": 24,
    "anydoor_lama_mask_dilate": 31,
    "dilate_size": 0,
    "guide_dilate_size": 0,
    "conditioning_video_mode": "lama_cleaned_video",
    "conditioning_lama_mask_padding": 48,
    "anydoor_guidance_scale": 5.0,
}
