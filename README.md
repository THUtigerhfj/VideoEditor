# VideoEditor

## Current Status

Run the gradio by
```bash
cd VideoPainter/app
python app.py
```

Noticies:
* Our gradio is of old version, it forces a 5 seconds' timeout. Setting the timeout manually does not work and upgrading gradio will cause conflicts with other dependencies.
* We choose to let user retrieve the processed video by themselves to avoid displaying "error" on the frontend.
* We don't use `yield` to prevent stucking at a function running for a long time.

TODO:
* Support sketching. This is straightforward by treating the output of sketching as the input (reference image) of the current implementation.
* [Optional] Use flux.2 for better quality.

### Gradio sketch reference workflow

Because the bundled Gradio version has a short frontend timeout, sketch-to-reference generation also runs in the background, like tracking and inpainting.

Use it as follows:

1. Upload a sketch and fill Object Label / Optional Attributes.
2. Click `Generate Reference From Sketch`.
3. Wait and click `Check Sketch Status` to inspect progress.
4. Click `Load Sketch Result` after it completes. This fills the Reference Image, Segmented Reference preview, and reference mask state.

### Reference guide and edit mask modes

`reference_motion_guide` controls whether replacement-object pixels are visible in CogVideoX conditioning frames:

- `none`: no rough-pasted guide; the masked region is hidden from conditioning frames.
- `full_region`: rough-pastes the AnyDoor/reference object into every conditioning frame and lets CogVideoX regenerate the selected edit mask.
- `edge_refine`: rough-pastes the object but edits only a boundary ring. This is experimental and may amplify jitter if the pasted guide bbox is unstable.

`edit_mask_mode` controls the mask passed to CogVideoX:

- `propagation`: current behavior. CogVideoX edits the propagated replacement-object mask.
- `union_target_object`: edits the union of the original clicked video target mask and the propagated replacement-object mask. This gives the model room to regenerate both the new object and the boundary/background transition.

Recommended first trial for black borders:

```bash
--edit_mask_mode union_target_object \
--reference_motion_guide none \
--dilate_size 8
```

If identity is weak, try:

```bash
--edit_mask_mode union_target_object \
--reference_motion_guide full_region \
--mask_bbox_smoothing median \
--guide_dilate_size 4
```

### Frame0 mask-shaped sketch reference

For same-category replacements such as bottle-to-bottle, sketch reference generation can optionally use the original frame0 mask as shape guidance.

Recommended first trial:

```bash
--shape_conditioned_scribble \
--reference_controlnet_scale 0.5
```

If you want the reference mask to exactly match the original target mask:

```bash
--shape_conditioned_scribble \
--frame_shaped_reference \
--reference_propagation_mask_source video_target \
--edit_mask_mode propagation
```

In Gradio, use the matching controls: enable `Use Frame0 Mask Shape For Sketch Reference` and `Frame-Shaped Reference Mask`, then set `Reference Propagation Mask Source` to `video_target` and `Edit Mask Mode` to `propagation`. Reference replacement now uses full-video conditioning by default, so CogVideoX sees the original frame pixels while masks still define the editable region; the old blacked-out masked-video mode remains available in CLI as `--conditioning_video_mode masked_video` for debugging.

This is best for same-shape replacements. Do not use it to force cross-shape replacements such as apple into a bottle silhouette unless that is the intended visual result.

