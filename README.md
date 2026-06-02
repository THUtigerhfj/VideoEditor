# VideoEditor

VideoEditor is a local VideoPainter/AnyDoor-based video object editing workspace. 

## Current Status

The repository currently supports three editing paths:

- **Prompt mode**: use a text prompt and FLUX first-frame inpainting before VideoPainter propagation.
- **Image reference mode**: use a user-provided reference image/mask, AnyDoor first-frame replacement, and VideoPainter/CogVideoX propagation.
- **Sketch mode**: generate a realistic reference image from a user sketch, optionally fit it to the source video frame-0 target mask, then reuse the image-reference replacement pipeline.

## Run The Existing Gradio Demo

```bash
conda activate vp_310
cd root/autodl-tmp/VideoEditor/VideoPainter/app
python app.py
```

Notes:

- The bundled Gradio version has a short frontend timeout. Long-running tracking, sketch-reference generation, and inpainting are run as background jobs with explicit status/load buttons.
- The Gradio app intentionally avoids `yield` in long-running handlers because it can get stuck with the current dependency set.
- Use `Check Status` and `Load Result` buttons after starting tracking, sketch reference generation, or inpainting.

## Run The React Studio UI

The React UI is a separate frontend over `VideoPainter/app/api_server.py`. It provides a darker video-editor style interface with direct sketch drawing in the browser.

Build the frontend:

```bash
cd /root/autodl-tmp/VideoEditor/frontend
npm install
npm run build
```

Start the API and single-port built UI:

```bash
conda activate vp_310
cd /root/autodl-tmp/VideoEditor
python VideoPainter/app/api_server.py --port 7862
```

For frontend-only development without loading model weights:

```bash
cd /root/autodl-tmp/VideoEditor
python VideoPainter/app/api_server.py --demo --port 7862
```

Then in another shell:

```bash
cd /root/autodl-tmp/VideoEditor/frontend
npm run dev
```

The Vite dev server talks to the API on `http://127.0.0.1:7862` by default. Override with `VITE_API_BASE` if needed.

## Sketch Reference Workflow

Sketch input is always treated as a sketch, not as a photo. The pipeline is:

1. Convert the sketch into an SDXL scribble ControlNet condition.
2. Generate realistic reference candidates.
3. Segment candidates with GroundingDINO + SAM2, with BiRefNet fallback when enabled.
4. Rank candidates by segmentation/object quality and optional source-mask shape compatibility.
5. Save `reference_image.png`, `reference_mask.png`, candidate artifacts, and metadata.
6. Feed the selected reference into AnyDoor for frame-0 replacement.
7. Feed the AnyDoor first frame, masks, and prompts into VideoPainter/CogVideoX.

Gradio usage:

1. Upload a video and click the target object on frame 0.
2. Track the target mask.
3. Open `Sketch to Reference`.
4. Upload or draw a sketch image, fill `Object Label` and optional attributes.
5. Click `Generate Reference From Sketch`.
6. Click `Check Sketch Status` until complete.
7. Click `Load Sketch Result`.
8. Run `Exact Object Replacement`.

React Studio usage:

1. Upload a video.
2. Click the target object.
3. Track the target mask.
4. Select `Sketch` mode.
5. Draw directly on the canvas.
6. Enter the target object and material/style prompts.
7. Click `Generate`.

## Recommended Sketch Settings

For same-shape or same-category replacements, such as bottle-to-bottle or can-to-can, use the frame-0 mask-shaped reference path:

```bash
--shape_conditioned_scribble \
--frame_shaped_reference \
--reference_propagation_mask_source video_target \
--edit_mask_mode propagation \
--conditioning_video_mode full_video \
--reference_controlnet_scale 0.5 \
--dilate_size 8
```

This makes the final generated `reference_mask.png` match the original frame-0 target mask. It is useful when you want the replacement to preserve the source object silhouette. It should not be used to force a cross-shape object, such as an apple, into a bottle-shaped mask unless that silhouette is intended.

Important parameter notes:

- `shape_conditioned_scribble`: adds the source frame-0 target mask contour into the sketch ControlNet condition.
- `frame_shaped_reference`: places the selected reference object into the original 720x480 frame-0 target mask and uses that mask as `reference_mask`.
- `reference_propagation_mask_source=video_target`: keeps the original clicked video mask sequence for propagation.
- `conditioning_video_mode=full_video`: keeps original video pixels visible in conditioning frames while masks still define the edit region. This avoids the black masked conditioning artifacts seen with `masked_video`.
- `sketch_mask_fit_strength`: currently reserved and recorded in metadata; it is not yet a true continuous warp-strength control.

## Reference Guide And Edit Mask Modes

`reference_motion_guide` controls whether rough reference-object pixels are visible in VideoPainter conditioning frames:

- `none`: no rough-pasted object guide.
- `full_region`: rough-pastes the AnyDoor/reference object into every conditioning frame and lets VideoPainter regenerate the selected edit mask.
- `edge_refine`: rough-pastes the object but edits only a boundary ring. This is experimental and can amplify jitter if the guide bbox is unstable.

`edit_mask_mode` controls the mask passed to VideoPainter:

- `propagation`: edits the propagated replacement-object mask.
- `union_target_object`: edits the union of the original clicked target mask and the propagated replacement-object mask. This gives the model room to regenerate both the object and some boundary/background transition.

## CLI Entry Points

Run sketch-to-reference only:

```bash
conda activate vp_310
cd /root/autodl-tmp/VideoEditor
python VideoPainter/tools/run_sketch_to_reference.py \
  --sketch_image path/to/sketch.png \
  --label "bottle" \
  --attrs "colorful glittering" \
  --output_dir runs/sketch_ref_test
```

Run full replacement without Gradio:

```bash
conda activate vp_310
cd /root/autodl-tmp/VideoEditor
python VideoPainter/tools/run_video_replace.py \
  --mode sketch_reference \
  --input_video path/to/input.mp4 \
  --video_points coordinate of the object to be replaced \
  --sketch_image path/to/sketch.png \
  --label "bottle" \
  --video_caption "a colorful glittering bottle on a table" \
  --target_object_caption "a colorful glittering bottle" \
  --shape_conditioned_scribble \
  --frame_shaped_reference \
  --reference_propagation_mask_source video_target \
  --edit_mask_mode propagation \
  --conditioning_video_mode full_video \
  --output_dir runs/sketch_replace_bottle
```