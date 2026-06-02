import argparse
import base64
import importlib
import os
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import sys
import threading
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw
import numpy as np

from sketch_defaults import SKETCH_MODE_DEFAULTS, reference_strategy_settings

VIDEO_PAINTER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT_ROOT = VIDEO_PAINTER_ROOT / "ckpt"


class TargetPointRequest(BaseModel):
    x: float
    y: float
    label: str = Field(default="Positive", pattern="^(Positive|Negative)$")


class PromptGenerateRequest(BaseModel):
    video_caption: str = ""
    target_region_caption: str = ""
    seed: int = 42
    cfg_scale: float = 6.0
    dilate_size: int = 8


class ImageGenerateRequest(PromptGenerateRequest):
    reference_strategy: str = SKETCH_MODE_DEFAULTS["reference_strategy"]
    dilate_size: int = 0
    anydoor_guidance_scale: float = 5.0
    anydoor_pre_inpaint_mode: str = "lama"
    anydoor_background_prompt: str = "unoccupied tabletop and laptop keyboard area, continuous desk and laptop surfaces matching the surrounding scene, same lighting, perspective, reflections, focus, and background texture"
    anydoor_background_mask_dilate: int = 0
    anydoor_background_guidance_scale: float = 30.0
    anydoor_background_num_inference_steps: int = 50
    lama_model: str = str(DEFAULT_CKPT_ROOT / "lama" / "big-lama.pt")
    anydoor_lama_mask_mode: str = "rect"
    anydoor_lama_mask_padding: int = 24
    anydoor_lama_mask_dilate: int = 31
    reference_propagation_mask_source: str = "video_target"
    reference_motion_guide: str = "none"
    edit_mask_mode: str = "propagation"
    guide_dilate_size: int = 0
    mask_bbox_smoothing: str = "off"
    mask_bbox_smoothing_window: int = 0
    mask_bbox_max_scale_delta: float = 0.08
    save_mask_bbox_stats: bool = False


class SketchGenerateRequest(BaseModel):
    sketch_image: str
    label: str
    attrs: str = ""
    video_caption: str = ""
    target_region_caption: str = ""
    seed: int = 42
    reference_strategy: str = SKETCH_MODE_DEFAULTS["reference_strategy"]


@dataclass
class EditSession:
    id: str
    root: Path
    inference_state: Any = None
    video_state: Any = None
    click_state: List[List[Any]] = field(default_factory=lambda: [[], []])
    ref_click_state: List[List[Any]] = field(default_factory=lambda: [[], []])
    ref_image: Any = None
    ref_mask: Any = None
    video_info: Dict[str, Any] = field(default_factory=dict)
    latest_output: Optional[Path] = None


@dataclass
class Job:
    id: str
    session_id: str
    type: str
    state: str = "queued"
    stage: str = "queued"
    progress: int = 0
    message: str = ""
    result: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.sessions: Dict[str, EditSession] = {}
        self.jobs: Dict[str, Job] = {}
        self.lock = threading.Lock()

    def create_session(self) -> EditSession:
        session_id = uuid4().hex
        session = EditSession(id=session_id, root=self.root / session_id)
        session.root.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> EditSession:
        session = self.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return session

    def create_job(self, session_id: str, job_type: str) -> Job:
        job = Job(id=uuid4().hex, session_id=session_id, type=job_type)
        with self.lock:
            self.jobs[job.id] = job
        return job

    def get_job(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job


class SelectEvent:
    def __init__(self, x: float, y: float):
        self.index = (int(round(x)), int(round(y)))


def artifact_url(session_id: str, path: Path) -> str:
    return f"/api/artifacts/{session_id}/{path.name}"


def save_image(path: Path, image: Any) -> Path:
    array = np.asarray(image, dtype=np.uint8)
    Image.fromarray(array).save(path)
    return path


def decode_data_url_image(data_url: str) -> np.ndarray:
    payload = data_url.split(",", 1)[1] if "," in data_url else data_url
    raw = base64.b64decode(payload)
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"sketch-{uuid4().hex}.png"
    tmp.write_bytes(raw)
    try:
        return np.asarray(Image.open(tmp).convert("RGB"), dtype=np.uint8)
    finally:
        tmp.unlink(missing_ok=True)


def write_demo_video(path: Path) -> Path:
    width, height = 720, 480
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (width, height))
    for idx in range(16):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (16, 20, 30)
        cv2.circle(frame, (120 + idx * 24, 240), 46, (20, 210, 235), -1)
        writer.write(frame)
    writer.release()
    return path


class DemoBridge:
    def upload_video(self, session: EditSession, upload_path: Path) -> Dict[str, Any]:
        frame = read_first_frame(upload_path)
        if frame is None:
            frame = Image.new("RGB", (720, 480), (12, 16, 26))
            draw = ImageDraw.Draw(frame)
            draw.rectangle((260, 130, 460, 350), outline=(42, 227, 255), width=4)
            draw.text((24, 24), "Frame 0 target preview", fill=(230, 245, 255))
        frame_path = session.root / "frame_000.png"
        frame.save(frame_path)
        session.video_state = {"demo": True, "video_name": upload_path.name, "masks": []}
        session.video_info = {"name": upload_path.name, "fps": 8, "frames": 49, "size": "720x480"}
        return {"video_info": session.video_info, "frame_path": frame_path}

    def refine_target(self, session: EditSession, point: TargetPointRequest) -> Path:
        frame = Image.new("RGB", (720, 480), (12, 16, 26))
        draw = ImageDraw.Draw(frame, "RGBA")
        draw.ellipse((point.x - 70, point.y - 70, point.x + 70, point.y + 70), fill=(42, 227, 255, 100))
        draw.ellipse((point.x - 5, point.y - 5, point.x + 5, point.y + 5), fill=(255, 255, 255, 255))
        overlay_path = session.root / "target_overlay.png"
        frame.save(overlay_path)
        session.click_state[0].append([point.x, point.y])
        session.click_state[1].append(1 if point.label == "Positive" else 0)
        session.video_state["masks"] = ["demo-mask"]
        return overlay_path

    def clear_target(self, session: EditSession) -> Path:
        session.click_state = [[], []]
        return session.root / "frame_000.png"

    def finish_demo_job(self, session: EditSession, job: Job, job_type: str, defaults: Optional[Dict[str, Any]] = None) -> None:
        job.state = "running"
        job.stage = "Rendering demo result"
        job.progress = 50
        output = write_demo_video(session.root / f"{job_type}_result.mp4")
        session.latest_output = output
        job.state = "succeeded"
        job.stage = "Complete"
        job.progress = 100
        job.message = "Demo job complete"
        job.result = {"video_url": artifact_url(session.id, output)}
        if defaults:
            job.result["defaults"] = defaults

    def segment_reference(self, session: EditSession, point: TargetPointRequest) -> Path:
        if session.ref_image is None:
            raise HTTPException(status_code=409, detail="Upload a reference image first")
        image = Image.fromarray(np.asarray(session.ref_image, dtype=np.uint8)).convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        draw.ellipse((point.x - 60, point.y - 60, point.x + 60, point.y + 60), fill=(80, 255, 120, 95))
        session.ref_mask = np.ones((image.height, image.width), dtype=np.uint8)
        path = session.root / "reference_segmented.png"
        image.save(path)
        return path


def read_first_frame(video_path: Path) -> Optional[Image.Image]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (720, 480), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(frame).convert("RGB")


class LegacyBridge:
    def __init__(self, legacy: Any):
        self.legacy = legacy

    def upload_video(self, session: EditSession, upload_path: Path) -> Dict[str, Any]:
        video_caption, inference_state, video_state, video_info, frame, _status = self.legacy.get_frames_from_video(str(upload_path), None)
        session.inference_state = inference_state
        session.video_state = video_state
        session.video_info = {"name": upload_path.name, "summary": video_info, "caption": video_caption}
        frame_path = save_image(session.root / "frame_000.png", frame)
        return {"video_info": session.video_info, "frame_path": frame_path}

    def refine_target(self, session: EditSession, point: TargetPointRequest) -> Path:
        frame, video_state, click_state, _status = self.legacy.sam_refine(
            session.inference_state,
            session.video_state,
            point.label,
            session.click_state,
            SelectEvent(point.x, point.y),
            [("", "")],
        )
        session.video_state = video_state
        session.click_state = click_state
        return save_image(session.root / "target_overlay.png", frame)

    def clear_target(self, session: EditSession) -> Path:
        inference_state, frame, click_state, _progress, _status = self.legacy.clear_click(
            session.inference_state,
            session.video_state,
            [("", "")],
        )
        session.inference_state = inference_state
        session.click_state = click_state
        return save_image(session.root / "frame_000.png", frame)

    def segment_reference(self, session: EditSession, point: TargetPointRequest) -> Path:
        preview, ref_click_state, ref_mask = self.legacy.semantic_segment_reference(
            session.ref_image,
            point.label,
            session.ref_click_state,
            SelectEvent(point.x, point.y),
        )
        session.ref_click_state = ref_click_state
        session.ref_mask = ref_mask
        return save_image(session.root / "reference_segmented.png", preview)

    def start_tracking(self, session: EditSession) -> None:
        _video, video_state, _message, _status = self.legacy.track_video(session.inference_state, session.video_state, [("", "")])
        session.video_state = video_state

    def start_prompt_generation(self, session: EditSession, request: PromptGenerateRequest) -> None:
        self.legacy.inpaint_video_background(
            session.video_state,
            request.video_caption,
            request.target_region_caption,
            [("", "")],
            request.seed,
            request.cfg_scale,
            request.dilate_size,
        )

    def start_image_generation(self, session: EditSession, request: ImageGenerateRequest) -> None:
        strategy = reference_strategy_settings(request.reference_strategy)
        self.legacy.exact_replace_video_background(
            session.video_state,
            session.ref_image,
            session.ref_mask,
            request.video_caption,
            request.target_region_caption,
            [("", "")],
            request.seed,
            request.cfg_scale,
            strategy["dilate_size"],
            request.anydoor_guidance_scale,
            strategy["reference_strategy"],
            strategy["reference_propagation_mask_source"],
            strategy["reference_motion_guide"],
            strategy["edit_mask_mode"],
            strategy["guide_dilate_size"],
            request.mask_bbox_smoothing,
            request.mask_bbox_smoothing_window,
            request.mask_bbox_max_scale_delta,
            request.save_mask_bbox_stats,
            strategy["anydoor_pre_inpaint_mode"],
            request.anydoor_background_prompt,
            request.anydoor_background_mask_dilate,
            request.anydoor_background_guidance_scale,
            request.anydoor_background_num_inference_steps,
            request.lama_model,
            request.anydoor_lama_mask_mode,
            request.anydoor_lama_mask_padding,
            request.anydoor_lama_mask_dilate,
        )

    def run_sketch_generation(self, session: EditSession, request: SketchGenerateRequest, job: Job) -> None:
        strategy = reference_strategy_settings(request.reference_strategy)
        sketch = decode_data_url_image(request.sketch_image)
        job.stage = "Generating reference candidates"
        job.progress = 10
        self.legacy.generate_reference_from_sketch(
            sketch,
            session.video_state,
            request.label,
            request.attrs,
            strategy["reference_strategy"],
            SKETCH_MODE_DEFAULTS["candidate_count"],
            SKETCH_MODE_DEFAULTS["reference_num_inference_steps"],
            SKETCH_MODE_DEFAULTS["reference_guidance_scale"],
            SKETCH_MODE_DEFAULTS["reference_controlnet_scale"],
            strategy["shape_conditioned_scribble"],
            SKETCH_MODE_DEFAULTS["sketch_mask_fit_strength"],
            SKETCH_MODE_DEFAULTS["mask_contour_weight"],
            strategy["frame_shaped_reference"],
            strategy["frame_shaped_reference_object_scale"],
            request.seed,
            [("", "")],
        )
        wait_for_legacy_flag(self.legacy, "sketch_reference", job, "Generating reference")
        ref_image, _preview, ref_click_state, ref_mask, _progress, _status = self.legacy.load_latest_sketch_reference()
        session.ref_image = ref_image
        session.ref_mask = ref_mask
        session.ref_click_state = ref_click_state
        if ref_image is None or ref_mask is None:
            raise RuntimeError("Sketch reference did not produce a usable reference image and mask")
        job.stage = "Running AnyDoor and CogVideoX"
        job.progress = 55
        self.legacy.exact_replace_video_background(
            session.video_state,
            session.ref_image,
            session.ref_mask,
            request.video_caption,
            request.target_region_caption,
            [("", "")],
            request.seed,
            6.0,
            strategy["dilate_size"],
            SKETCH_MODE_DEFAULTS["anydoor_guidance_scale"],
            strategy["reference_strategy"],
            strategy["reference_propagation_mask_source"],
            strategy["reference_motion_guide"],
            strategy["edit_mask_mode"],
            strategy["guide_dilate_size"],
            "off",
            0,
            0.08,
            False,
            strategy["anydoor_pre_inpaint_mode"],
            SKETCH_MODE_DEFAULTS["anydoor_background_prompt"],
            SKETCH_MODE_DEFAULTS["anydoor_background_mask_dilate"],
            SKETCH_MODE_DEFAULTS["anydoor_background_guidance_scale"],
            SKETCH_MODE_DEFAULTS["anydoor_background_num_inference_steps"],
            str(DEFAULT_CKPT_ROOT / "lama" / "big-lama.pt"),
            SKETCH_MODE_DEFAULTS["anydoor_lama_mask_mode"],
            SKETCH_MODE_DEFAULTS["anydoor_lama_mask_padding"],
            SKETCH_MODE_DEFAULTS["anydoor_lama_mask_dilate"],
        )
        wait_for_legacy_flag(self.legacy, "inpainting", job, "Generating video")


def wait_for_legacy_flag(legacy: Any, flag: str, job: Job, label: str) -> None:
    message_key = f"{flag}_message"
    error_key = f"{flag}_error"
    while True:
        with legacy.processing_lock:
            running = legacy.processing_status[flag]
            message = legacy.processing_status.get(message_key, "")
            error = legacy.processing_status.get(error_key, "")
        if error:
            raise RuntimeError(error)
        if not running:
            return
        job.state = "running"
        job.stage = label
        job.message = message
        time.sleep(1.0)


def copy_result_to_session(session: EditSession, source_path: str, name: str) -> Path:
    source = Path(source_path)
    target = session.root / name
    if source.exists():
        shutil.copy2(source, target)
        return target
    raise RuntimeError(f"Result file does not exist: {source}")


def job_payload(job: Job) -> Dict[str, Any]:
    return {
        "job_id": job.id,
        "session_id": job.session_id,
        "type": job.type,
        "state": job.state,
        "stage": job.stage,
        "progress": job.progress,
        "message": job.message,
        "result": job.result,
        "error": job.error,
    }


def session_payload(session: EditSession) -> Dict[str, Any]:
    return {
        "session_id": session.id,
        "video_info": session.video_info,
        "points": [
            {"x": float(point[0]), "y": float(point[1]), "label": "Positive" if label == 1 else "Negative"}
            for point, label in zip(session.click_state[0], session.click_state[1])
        ],
        "has_tracked_mask": has_nonempty_mask(session.video_state),
        "has_reference": session.ref_image is not None and session.ref_mask is not None,
        "latest_output_url": artifact_url(session.id, session.latest_output) if session.latest_output else None,
    }


def has_nonempty_mask(video_state: Any) -> bool:
    if not video_state or not video_state.get("masks"):
        return False
    try:
        return any(np.asarray(mask).max() > 0 for mask in video_state["masks"])
    except Exception:
        return bool(video_state.get("masks"))


def require_video(session: EditSession) -> None:
    if session.video_state is None:
        raise HTTPException(status_code=409, detail="Upload a video before generating.")


def require_reference(session: EditSession) -> None:
    if session.ref_image is None or session.ref_mask is None:
        raise HTTPException(status_code=409, detail="Upload and segment a reference image before generating.")


def load_legacy_app(args: argparse.Namespace) -> Any:
    original_argv = sys.argv[:]
    sys.argv = [
        "app.py",
        "--model_path",
        args.model_path,
        "--inpainting_branch",
        args.inpainting_branch,
        "--id_adapter",
        args.id_adapter,
        "--img_inpainting_model",
        args.img_inpainting_model,
        "--sam2_checkpoint",
        args.sam2_checkpoint,
        "--port",
        str(args.gradio_port),
    ]
    try:
        return importlib.import_module("app")
    finally:
        sys.argv = original_argv


def default_frontend_dist() -> Path:
    return Path(__file__).resolve().parents[2] / "frontend" / "dist"


def create_app(
    demo_mode: bool = False,
    artifact_root: Optional[Path] = None,
    legacy_module: Any = None,
    frontend_dist: Optional[Path] = None,
) -> FastAPI:
    root = Path(artifact_root or os.environ.get("VIDEO_PAINTER_API_TMP", Path(__file__).resolve().parent / "tmp_api")).resolve()
    store = Store(root)
    bridge = DemoBridge() if demo_mode else LegacyBridge(legacy_module)

    app = FastAPI(title="VideoPainter API")
    app.state.store = store
    app.state.bridge = bridge
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {"ok": True, "demo_mode": demo_mode}

    @app.post("/api/sessions")
    def create_session() -> Dict[str, Any]:
        session = store.create_session()
        return session_payload(session)

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> Dict[str, Any]:
        return session_payload(store.get_session(session_id))

    @app.post("/api/sessions/{session_id}/video")
    async def upload_video(session_id: str, file: UploadFile = File(...)) -> Dict[str, Any]:
        session = store.get_session(session_id)
        suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
        upload_path = session.root / f"input{suffix}"
        upload_path.write_bytes(await file.read())
        result = bridge.upload_video(session, upload_path)
        return {
            "session_id": session.id,
            "video_info": result["video_info"],
            "frame_url": artifact_url(session.id, result["frame_path"]),
        }

    @app.post("/api/sessions/{session_id}/target-points")
    def add_target_point(session_id: str, point: TargetPointRequest) -> Dict[str, Any]:
        session = store.get_session(session_id)
        require_video(session)
        overlay = bridge.refine_target(session, point)
        return {"overlay_url": artifact_url(session.id, overlay), "points": session_payload(session)["points"]}

    @app.delete("/api/sessions/{session_id}/target-points")
    def clear_target_points(session_id: str) -> Dict[str, Any]:
        session = store.get_session(session_id)
        require_video(session)
        frame = bridge.clear_target(session)
        return {"frame_url": artifact_url(session.id, frame), "points": []}

    @app.post("/api/sessions/{session_id}/track")
    def start_tracking(session_id: str) -> Dict[str, Any]:
        session = store.get_session(session_id)
        require_video(session)
        job = store.create_job(session.id, "tracking")
        if demo_mode:
            bridge.finish_demo_job(session, job, "tracking")
            return job_payload(job)

        def run() -> None:
            try:
                job.state = "running"
                job.stage = "Tracking target"
                bridge.start_tracking(session)
                wait_for_legacy_flag(bridge.legacy, "tracking", job, "Tracking target")
                output = copy_result_to_session(session, bridge.legacy.latest_tracking_video, "tracked_preview.mp4")
                job.state = "succeeded"
                job.stage = "Complete"
                job.progress = 100
                job.result = {"video_url": artifact_url(session.id, output)}
            except Exception as exc:
                job.state = "failed"
                job.error = str(exc)

        threading.Thread(target=run, daemon=True).start()
        return job_payload(job)

    @app.post("/api/sessions/{session_id}/reference-image")
    async def upload_reference_image(session_id: str, file: UploadFile = File(...)) -> Dict[str, Any]:
        session = store.get_session(session_id)
        image_path = session.root / "reference.png"
        image_path.write_bytes(await file.read())
        session.ref_image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        session.ref_click_state = [[], []]
        session.ref_mask = None
        return {"reference_url": artifact_url(session.id, image_path)}

    @app.post("/api/sessions/{session_id}/reference-points")
    def add_reference_point(session_id: str, point: TargetPointRequest) -> Dict[str, Any]:
        session = store.get_session(session_id)
        preview = bridge.segment_reference(session, point)
        return {"preview_url": artifact_url(session.id, preview), "has_reference_mask": session.ref_mask is not None}

    @app.post("/api/sessions/{session_id}/generate/prompt")
    def generate_prompt(session_id: str, request: PromptGenerateRequest) -> Dict[str, Any]:
        session = store.get_session(session_id)
        require_video(session)
        job = store.create_job(session.id, "prompt")
        if demo_mode:
            bridge.finish_demo_job(session, job, "prompt")
            return job_payload(job)

        def run() -> None:
            try:
                job.state = "running"
                job.stage = "Generating prompt edit"
                bridge.start_prompt_generation(session, request)
                wait_for_legacy_flag(bridge.legacy, "inpainting", job, "Generating prompt edit")
                output = copy_result_to_session(session, bridge.legacy.latest_inpaint_video, "prompt_result.mp4")
                session.latest_output = output
                job.state = "succeeded"
                job.stage = "Complete"
                job.progress = 100
                job.result = {"video_url": artifact_url(session.id, output)}
            except Exception as exc:
                job.state = "failed"
                job.error = str(exc)

        threading.Thread(target=run, daemon=True).start()
        return job_payload(job)

    @app.post("/api/sessions/{session_id}/generate/image")
    def generate_image(session_id: str, request: ImageGenerateRequest) -> Dict[str, Any]:
        session = store.get_session(session_id)
        require_video(session)
        require_reference(session)
        job = store.create_job(session.id, "image")
        if demo_mode:
            bridge.finish_demo_job(session, job, "image")
            return job_payload(job)

        def run() -> None:
            try:
                job.state = "running"
                job.stage = "Generating reference edit"
                bridge.start_image_generation(session, request)
                wait_for_legacy_flag(bridge.legacy, "inpainting", job, "Generating reference edit")
                output = copy_result_to_session(session, bridge.legacy.latest_inpaint_video, "image_result.mp4")
                session.latest_output = output
                job.state = "succeeded"
                job.stage = "Complete"
                job.progress = 100
                job.result = {"video_url": artifact_url(session.id, output)}
            except Exception as exc:
                job.state = "failed"
                job.error = str(exc)

        threading.Thread(target=run, daemon=True).start()
        return job_payload(job)

    @app.post("/api/sessions/{session_id}/generate/sketch")
    def generate_sketch(session_id: str, request: SketchGenerateRequest) -> Dict[str, Any]:
        session = store.get_session(session_id)
        require_video(session)
        job = store.create_job(session.id, "sketch")
        if demo_mode:
            defaults = {**SKETCH_MODE_DEFAULTS, **reference_strategy_settings(request.reference_strategy)}
            bridge.finish_demo_job(session, job, "sketch", defaults=defaults)
            return job_payload(job)

        def run() -> None:
            try:
                job.state = "running"
                bridge.run_sketch_generation(session, request, job)
                output = copy_result_to_session(session, bridge.legacy.latest_inpaint_video, "sketch_result.mp4")
                session.latest_output = output
                job.state = "succeeded"
                job.stage = "Complete"
                job.progress = 100
                defaults = {**SKETCH_MODE_DEFAULTS, **reference_strategy_settings(request.reference_strategy)}
                job.result = {"video_url": artifact_url(session.id, output), "defaults": defaults}
            except Exception as exc:
                job.state = "failed"
                job.error = str(exc)

        threading.Thread(target=run, daemon=True).start()
        return job_payload(job)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> Dict[str, Any]:
        return job_payload(store.get_job(job_id))

    @app.get("/api/artifacts/{session_id}/{filename}")
    def get_artifact(session_id: str, filename: str) -> FileResponse:
        session = store.get_session(session_id)
        path = session.root / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail="Artifact not found")
        return FileResponse(path)

    dist = Path(frontend_dist or default_frontend_dist()).resolve()
    index = dist / "index.html"
    assets = dist / "assets"
    if index.exists() and assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")

        @app.get("/", include_in_schema=False)
        def get_frontend_index() -> FileResponse:
            return FileResponse(index, headers={"Cache-Control": "no-store"})

        @app.get("/{path:path}", include_in_schema=False)
        def get_frontend_fallback(path: str) -> FileResponse:
            if path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not Found")
            asset_alias = assets / path
            if asset_alias.exists() and asset_alias.is_file():
                return FileResponse(asset_alias, headers={"Cache-Control": "no-cache"})
            if Path(path).suffix in {".js", ".css", ".map", ".png", ".jpg", ".jpeg", ".svg", ".ico", ".webp"}:
                raise HTTPException(status_code=404, detail="Static asset not found")
            return FileResponse(index, headers={"Cache-Control": "no-store"})

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--demo", action="store_true", help="Start without loading model weights; useful for frontend development.")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_CKPT_ROOT / "CogVideoX-5b-I2V"))
    parser.add_argument("--inpainting_branch", type=str, default=str(DEFAULT_CKPT_ROOT / "VideoPainter" / "checkpoints" / "branch"))
    parser.add_argument("--id_adapter", type=str, default=str(DEFAULT_CKPT_ROOT / "VideoPainterID" / "checkpoints"))
    parser.add_argument("--img_inpainting_model", type=str, default=str(DEFAULT_CKPT_ROOT / "flux_inp"))
    parser.add_argument("--sam2_checkpoint", type=str, default=str(DEFAULT_CKPT_ROOT / "sam2_hiera_large.pt"))
    parser.add_argument("--gradio_port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    legacy = None if args.demo else load_legacy_app(args)
    app = create_app(demo_mode=args.demo, legacy_module=legacy)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
