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