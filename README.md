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
cd /root/autodl-tmp/VideoEditor/VideoPainter/app
python app.py
```

Notes:

- The bundled Gradio version has a short frontend timeout. Long-running tracking, sketch-reference generation, and inpainting are run as background jobs with explicit status/load buttons.
- The Gradio app intentionally avoids `yield` in long-running handlers because it can get stuck with the current dependency set.
- Use `Check Status` and `Load Result` buttons after starting tracking, sketch reference generation, or inpainting.
- Sketch-to-reference model loaders use local Hugging Face cache first and only try the Hub if the required model is not available locally. The default cache directory is `/root/autodl-tmp/VideoEditor/VideoPainter/ckpt/sketch_ref`.
- Reference/sketch replacement now defaults to prompt-free LaMa cleanup before AnyDoor. The local LaMa model path is `/root/autodl-tmp/VideoEditor/VideoPainter/ckpt/lama/big-lama.pt`.
- Gradio exposes `Reference Strategy` with two presets: `lama_background` and `mask_twist`.
- Reference/sketch exact replacement saves the edited first frame separately and returns a tail video without frame 0. This matches the current assumption that CogVideoX does not regenerate frame 0; the comparable first-frame artifact is saved as `exact_replace_first_frame.png`.

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

Open the UI at:

```text
http://127.0.0.1:7862
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

## Reference Strategy Presets

React Studio and Gradio both expose the same two reference/sketch replacement presets.

Prompt mode has its own defaults and does not use these reference strategy presets. Its `dilate_size` default is `16`, matching the initial prompt-inpainting demo behavior.

### `lama_background` (default)

Use this for the current sketch/reference demo path.

- Sketch reference is shape-conditioned and fitted into the original frame-0 target bbox at scale `0.82`.
- LaMa removes the original object from frame 0 before AnyDoor.
- AnyDoor uses the smaller reference-object mask as its target mask, so the replacement object can be smaller than the original source mask.
- CogVideoX still receives the original clicked video mask sequence.
- Later conditioning frames are cleaned with LaMa using `conditioning_video_mode=lama_cleaned_video` and `conditioning_lama_mask_padding=48`.
- `dilate_size=0` and `guide_dilate_size=0`, so the CogVideoX edit mask is not expanded by default.

### `mask_twist`

Use this when you want the generated reference to match the original target mask size/shape as closely as possible.

- Sketch reference is shape-conditioned and fitted to the original frame-0 target bbox at scale `1.0`.
- AnyDoor pre-inpaint is disabled by default.
- AnyDoor receives the original video frame-0 target mask as its target mask.
- CogVideoX uses `conditioning_video_mode=full_video`.
- The original video target mask sequence is kept for propagation and editing.
- This is closer to the older “twist/fit into original video mask” behavior. It can be more stable when the replacement sketch has nearly the same silhouette as the original object, but it gives less room for background completion inside the mask.
- Important limitation: AnyDoor is generative replacement, not strict alpha-mask paste. Even when the reference mask, AnyDoor target mask, and CogVideoX frame-0 mask are identical, AnyDoor may still alter the object silhouette or orientation inside the target bbox.

## Sketch Reference Workflow

Sketch input is always treated as a sketch, not as a photo. The pipeline is:

1. Convert the sketch into an SDXL scribble ControlNet condition.
2. Generate realistic reference candidates.
3. Segment candidates with GroundingDINO + SAM2, with BiRefNet fallback when enabled.
4. Rank candidates by segmentation/object quality and optional source-mask shape compatibility.
5. Save `reference_image.png`, `reference_mask.png`, candidate artifacts, and metadata.
6. Feed the selected reference into AnyDoor for frame-0 replacement.
7. Feed the AnyDoor first frame, masks, and prompts into VideoPainter/CogVideoX.
8. Save the edited first frame separately and return the generated tail video without frame 0 for UI/Gradio exact replacement.

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

For most sketch/reference runs, start with the default `lama_background` preset:

```bash
--shape_conditioned_scribble \
--frame_shaped_reference \
--frame_shaped_reference_object_scale 0.82 \
--reference_propagation_mask_source video_target \
--edit_mask_mode propagation \
--conditioning_video_mode lama_cleaned_video \
--conditioning_lama_mask_padding 48 \
--reference_controlnet_scale 0.5 \
--anydoor_pre_inpaint_mode lama \
--anydoor_lama_mask_mode rect \
--anydoor_lama_mask_padding 24 \
--dilate_size 0
```

This creates a frame-sized `reference_image.png` but keeps the automatically segmented `reference_mask.png` on the smaller fitted object. AnyDoor uses that smaller object mask as its frame-0 `target_mask`, so the first replacement object stays small. VideoPainter/CogVideoX should still use `reference_propagation_mask_source=video_target`, so frame 0 and all later edit masks remain the original tracked video masks. For sketch/reference mode the default `dilate_size` is `0`; this avoids expanding the original edit mask and prevents boundary blur from dilation.

Recommended target caption style:

```text
a smaller colorful glittering bottle centered in the original area, with the surrounding table/background naturally filled, no black outline
```

This is useful when you want FLUX-like behavior from sketch/reference mode: a new object inside the original edit region plus natural background completion around it. It is still best for same-shape or same-category replacements. Forcing a cross-shape object, such as an apple, into a bottle-shaped source mask remains unstable.

Important parameter notes:

- `shape_conditioned_scribble`: adds the source frame-0 target mask contour into the sketch ControlNet condition.
- `frame_shaped_reference`: places the selected reference object into the original 720x480 frame-0 target bbox.
- `frame_shaped_reference_object_scale`: controls how much of that bbox the reference object occupies. Default `0.82` leaves room for background completion inside the original edit mask.
- `reference_propagation_mask_source=video_target`: keeps the original clicked video mask sequence for propagation.
- `conditioning_video_mode=lama_cleaned_video`: removes the old object from later conditioning frames with LaMa while keeping the original CogVideoX edit masks unchanged.
- `sketch_mask_fit_strength`: currently reserved and recorded in metadata; it is not yet a true continuous warp-strength control.

## AnyDoor Pre-Inpaint Backends

Reference and sketch modes first clean the source frame-0 target area, then let AnyDoor place the smaller replacement object. The current default is:

```bash
pip install --no-deps simple-lama-inpainting
--anydoor_pre_inpaint_mode lama \
--lama_model /root/autodl-tmp/VideoEditor/VideoPainter/ckpt/lama/big-lama.pt \
--anydoor_lama_mask_mode rect \
--anydoor_lama_mask_padding 24
```

Why LaMa is default: FLUX.1 Fill is strong for text-guided image editing, but in this bottle/cup sample it repeatedly hallucinated a new cup/bottle when asked to remove the original object from a cup-shaped mask. LaMa is prompt-free object-removal inpainting, so it is less likely to create a second semantic object before AnyDoor runs.

Available modes:

- `lama`: prompt-free object removal. Default for reference/sketch replacement.
- `flux`: FLUX.1 Fill cleanup. Kept for comparison and scenes where text-guided background synthesis works better.
- `opencv`: fast classical inpainting fallback.
- `off`: no pre-cleanup; AnyDoor receives the original frame.

Debug outputs:

- `first_frames/lama_background_first_frame.png`
- `masks/lama_background_mask.png`
- `first_frames/anydoor_target_input.png`
- `first_frames/anydoor_first_frame.png`

Known limitation: LaMa can leave a mild smudge or blur inside the removed area. In the current bottle sample, `rect` mode with `padding=24` removed the duplicate old object and produced an AnyDoor first frame with only one replacement object, but the background is not perfectly clean.

## Exact Replacement Outputs And Debug Files

UI/Gradio exact replacement currently trims frame 0 from the returned result video because frame 0 comes from the external first-frame edit stage rather than CogVideoX generation. Inspect frame 0 and masks through saved debug files:

- `VideoPainter/app/tmp_gradio/inpaint/first_frame_anydoor.png`
- `VideoPainter/app/tmp_gradio/inpaint/exact_replace_first_frame.png`
- `VideoPainter/app/tmp_gradio/inpaint/exact_replace_tail_<video_name>`
- `VideoPainter/app/tmp_gradio/inpaint/anydoor_target_input.png`
- `VideoPainter/app/tmp_gradio/inpaint/anydoor_target_mask.png`
- `VideoPainter/app/tmp_gradio/inpaint/cogvideox_video_target_first_mask.png`
- `VideoPainter/app/tmp_gradio/inpaint/propagation_first_mask.png`

The CLI path still writes run-directory outputs, including `outputs/image_reference_videopainter_48f.mp4` and `outputs/image_reference_videopainter_49f_with_first.mp4` when that mode is used.

## AnyDoor Mask Limitation

The current AnyDoor bridge passes masks correctly, but AnyDoor's `run_inference.py` path uses the target mask mainly to derive the target bbox and conditioning crop. It also uses a rectangular collage region inside that crop. It is therefore not guaranteed to preserve an exact mask-shaped silhouette. If exact pixel-level mask conformity is required, a later implementation must add a post-AnyDoor warp/composite step or switch to a stricter shape-control path.

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
  --video_caption "a smaller colorful glittering bottle on a table, with the surrounding background naturally filled" \
  --target_object_caption "a smaller colorful glittering bottle centered in the original area, with the surrounding table/background naturally filled, no black outline" \
  --shape_conditioned_scribble \
  --frame_shaped_reference \
  --frame_shaped_reference_object_scale 0.82 \
  --reference_propagation_mask_source video_target \
  --edit_mask_mode propagation \
  --conditioning_video_mode lama_cleaned_video \
  --conditioning_lama_mask_padding 48 \
  --anydoor_pre_inpaint_mode lama \
  --anydoor_lama_mask_mode rect \
  --anydoor_lama_mask_padding 24 \
  --dilate_size 0 \
  --output_dir runs/sketch_replace_bottle
```
