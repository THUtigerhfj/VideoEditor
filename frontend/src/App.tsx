import {
  Brush,
  CircleDot,
  Eraser,
  Image as ImageIcon,
  Layers,
  Loader2,
  PanelRightOpen,
  Play,
  RotateCcw,
  Sparkles,
  Square,
  Trash2,
  Undo2,
  Upload,
  Video,
  Wand2,
  X,
} from "lucide-react";
import { PointerEvent, ReactNode, forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import { api, apiBase, assetUrl, JobPayload, Mode, PointLabel, SessionPayload } from "./api/client";

type Notice = { tone: "info" | "success" | "error"; text: string };
type ReferenceStrategy = "lama_background" | "mask_twist";

const modes: Array<{ id: Mode; label: string; icon: typeof Sparkles }> = [
  { id: "prompt", label: "Prompt", icon: Sparkles },
  { id: "image", label: "Image", icon: ImageIcon },
  { id: "sketch", label: "Sketch", icon: Brush },
];

const referenceStrategies: Array<{ id: ReferenceStrategy; label: string; description: string }> = [
  {
    id: "lama_background",
    label: "LaMa clean",
    description: "Small reference object, LaMa background cleanup, original video masks.",
  },
  {
    id: "mask_twist",
    label: "Mask twist",
    description: "Fit/twist reference to the original frame-0 mask size and keep full-video conditioning.",
  },
];

function replacementSettingsForStrategy(referenceStrategy: ReferenceStrategy) {
  if (referenceStrategy === "mask_twist") {
    return {
      reference_strategy: referenceStrategy,
      anydoor_pre_inpaint_mode: "off",
      reference_propagation_mask_source: "video_target",
      reference_motion_guide: "none",
      edit_mask_mode: "propagation",
      guide_dilate_size: 0,
      dilate_size: 0,
    };
  }
  return {
    reference_strategy: referenceStrategy,
    anydoor_pre_inpaint_mode: "lama",
    reference_propagation_mask_source: "video_target",
    reference_motion_guide: "none",
    edit_mask_mode: "propagation",
    guide_dilate_size: 0,
    dilate_size: 0,
  };
}

type ImageSize = { width: number; height: number };

function freshAssetUrl(path?: unknown): string {
  const url = assetUrl(path);
  if (!url) return "";
  return `${url}${url.includes("?") ? "&" : "?"}v=${Date.now()}`;
}

function naturalImageSize(container: HTMLElement, fallback: ImageSize): ImageSize {
  const image = container.querySelector("img");
  return {
    width: image?.naturalWidth || fallback.width,
    height: image?.naturalHeight || fallback.height,
  };
}

function mapClientToCoveredImagePoint(clientX: number, clientY: number, rect: DOMRect, imageSize: ImageSize) {
  const scale = Math.max(rect.width / imageSize.width, rect.height / imageSize.height);
  const renderedWidth = imageSize.width * scale;
  const renderedHeight = imageSize.height * scale;
  const offsetX = (rect.width - renderedWidth) / 2;
  const offsetY = (rect.height - renderedHeight) / 2;
  const x = (clientX - rect.left - offsetX) / scale;
  const y = (clientY - rect.top - offsetY) / scale;
  return {
    x: Math.max(0, Math.min(imageSize.width - 1, x)),
    y: Math.max(0, Math.min(imageSize.height - 1, y)),
  };
}

function App() {
  const [session, setSession] = useState<SessionPayload | null>(null);
  const [mode, setMode] = useState<Mode>("sketch");
  const [pointLabel, setPointLabel] = useState<PointLabel>("Positive");
  const [referencePointLabel, setReferencePointLabel] = useState<PointLabel>("Positive");
  const [frameUrl, setFrameUrl] = useState("");
  const [targetOverlayUrl, setTargetOverlayUrl] = useState("");
  const [referenceUrl, setReferenceUrl] = useState("");
  const [referencePreviewUrl, setReferencePreviewUrl] = useState("");
  const [videoCaption, setVideoCaption] = useState("A cinematic product shot with stable lighting and natural motion.");
  const [targetCaption, setTargetCaption] = useState("a clean metallic can");
  const [attrs, setAttrs] = useState("metallic, realistic, studio lighting");
  const [seed, setSeed] = useState(42);
  const [cfgScale, setCfgScale] = useState(6);
  const [dilateSize, setDilateSize] = useState(0);
  const [referenceStrategy, setReferenceStrategy] = useState<ReferenceStrategy>("lama_background");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [activeJob, setActiveJob] = useState<JobPayload | null>(null);
  const [resultUrl, setResultUrl] = useState("");
  const [notice, setNotice] = useState<Notice>({ tone: "info", text: `API ${apiBase}` });
  const [isBusy, setIsBusy] = useState(false);
  const sketchRef = useRef<SketchCanvasHandle>(null);

  useEffect(() => {
    api
      .createSession()
      .then((payload) => {
        setSession(payload);
        setNotice({ tone: "success", text: "Session ready" });
      })
      .catch((error) => setNotice({ tone: "error", text: error.message }));
  }, []);

  useEffect(() => {
    if (!activeJob || activeJob.state === "succeeded" || activeJob.state === "failed") return;
    const timer = window.setInterval(async () => {
      try {
        const next = await api.getJob(activeJob.job_id);
        setActiveJob(next);
        if (next.state === "succeeded") {
          const videoUrl = freshAssetUrl(next.result.video_url);
          setResultUrl(videoUrl);
          setIsBusy(false);
          setNotice({ tone: "success", text: `${next.type} complete` });
        }
        if (next.state === "failed") {
          setIsBusy(false);
          setNotice({ tone: "error", text: next.error || "Job failed" });
        }
      } catch (error) {
        setIsBusy(false);
        setNotice({ tone: "error", text: (error as Error).message });
      }
    }, 1200);
    return () => window.clearInterval(timer);
  }, [activeJob]);

  const hasVideo = Boolean(frameUrl);
  const currentFrame = targetOverlayUrl || frameUrl;

  const refreshSession = useCallback(async () => {
    if (!session) return;
    setSession(await api.getSession(session.session_id));
  }, [session]);

  const uploadVideo = async (file?: File) => {
    if (!session || !file) return;
    try {
      setNotice({ tone: "info", text: "Loading video" });
      const payload = await api.uploadVideo(session.session_id, file);
      setFrameUrl(freshAssetUrl(payload.frame_url));
      setTargetOverlayUrl("");
      setResultUrl("");
      await refreshSession();
      setNotice({ tone: "success", text: "Video loaded" });
    } catch (error) {
      setNotice({ tone: "error", text: `Video upload failed: ${(error as Error).message}` });
    }
  };

  const addTargetPoint = async (clientX: number, clientY: number, container: HTMLElement) => {
    if (!session || !hasVideo) return;
    const point = mapClientToCoveredImagePoint(clientX, clientY, container.getBoundingClientRect(), naturalImageSize(container, { width: 720, height: 480 }));
    const payload = await api.addTargetPoint(session.session_id, point.x, point.y, pointLabel);
    setTargetOverlayUrl(freshAssetUrl(payload.overlay_url));
    await refreshSession();
  };

  const clearTarget = async () => {
    if (!session || !hasVideo) return;
    const payload = await api.clearTargetPoints(session.session_id);
    setTargetOverlayUrl("");
    setFrameUrl(freshAssetUrl(payload.frame_url));
    await refreshSession();
  };

  const startJob = async (runner: () => Promise<JobPayload>) => {
    setIsBusy(true);
    setNotice({ tone: "info", text: "Job started" });
    try {
      const job = await runner();
      setActiveJob(job);
      if (job.state === "succeeded") {
        setResultUrl(freshAssetUrl(job.result.video_url));
        setIsBusy(false);
        setNotice({ tone: "success", text: `${job.type} complete` });
      }
    } catch (error) {
      setIsBusy(false);
      setNotice({ tone: "error", text: (error as Error).message });
    }
  };

  const startTrack = () => {
    if (!session) return;
    startJob(() => api.startTrack(session.session_id));
  };

  const generate = () => {
    if (!session) return;
    const replacementSettings = replacementSettingsForStrategy(referenceStrategy);
    if (mode === "prompt") {
      startJob(() =>
        api.generatePrompt(session.session_id, {
          video_caption: videoCaption,
          target_region_caption: targetCaption,
          seed,
          cfg_scale: cfgScale,
          dilate_size: dilateSize,
        }),
      );
      return;
    }
    if (mode === "image") {
      startJob(() =>
        api.generateImage(session.session_id, {
          video_caption: videoCaption,
          target_region_caption: targetCaption,
          seed,
          cfg_scale: cfgScale,
          dilate_size: replacementSettings.dilate_size,
          anydoor_guidance_scale: 5,
          anydoor_pre_inpaint_mode: replacementSettings.anydoor_pre_inpaint_mode,
          anydoor_background_prompt:
            "unoccupied tabletop and laptop keyboard area, continuous desk and laptop surfaces matching the surrounding scene, same lighting, perspective, reflections, focus, and background texture",
          anydoor_background_mask_dilate: 0,
          anydoor_background_guidance_scale: 30,
          anydoor_background_num_inference_steps: 50,
          anydoor_lama_mask_mode: "rect",
          anydoor_lama_mask_padding: 24,
          anydoor_lama_mask_dilate: 31,
          reference_propagation_mask_source: replacementSettings.reference_propagation_mask_source,
          reference_motion_guide: replacementSettings.reference_motion_guide,
          edit_mask_mode: replacementSettings.edit_mask_mode,
          guide_dilate_size: replacementSettings.guide_dilate_size,
          reference_strategy: replacementSettings.reference_strategy,
          mask_bbox_smoothing: "off",
          mask_bbox_smoothing_window: 0,
          mask_bbox_max_scale_delta: 0.08,
          save_mask_bbox_stats: false,
        }),
      );
      return;
    }
    const sketch = sketchRef.current?.exportPng();
    if (!sketch) {
      setNotice({ tone: "error", text: "Sketch canvas is empty" });
      return;
    }
    startJob(() =>
      api.generateSketch(session.session_id, {
        sketch_image: sketch,
        label: targetCaption || "object",
        attrs,
        video_caption: videoCaption,
        target_region_caption: targetCaption,
        seed,
        reference_strategy: referenceStrategy,
      }),
    );
  };

  const progress = activeJob ? activeJob.progress : 0;
  const steps = useMemo(
    () => [
      { label: "Upload", done: hasVideo },
      { label: "Target", done: Boolean(session?.points.length) },
      { label: "Track", done: Boolean(session?.has_tracked_mask || activeJob?.type === "tracking") },
      { label: "Create", done: mode !== "sketch" || Boolean(sketchRef.current) },
      { label: "Generate", done: Boolean(activeJob) },
      { label: "Result", done: Boolean(resultUrl) },
    ],
    [activeJob, hasVideo, mode, resultUrl, session],
  );

  return (
    <main className="studio">
      <aside className="mode-rail" aria-label="Mode selector">
        <div className="brand-mark">
          <Wand2 size={24} />
        </div>
        {modes.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              className={`mode-button ${mode === item.id ? "active" : ""}`}
              onClick={() => setMode(item.id)}
              title={`${item.label} mode`}
            >
              <Icon size={21} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">VideoPainter Studio</p>
            <h1>Object-aware video editing</h1>
          </div>
          <StatusRail steps={steps} />
          <div className={`notice ${notice.tone}`}>{notice.text}</div>
        </header>

        <section className="stage-grid">
          <Panel title="Target" icon={<Video size={18} />} accent="cyan">
            <div className="upload-line">
              <label className="file-button">
                <Upload size={17} />
                <span>Video</span>
                <input
                  type="file"
                  accept="video/*"
                  onChange={(event) => {
                    void uploadVideo(event.target.files?.[0]);
                    event.currentTarget.value = "";
                  }}
                />
              </label>
              <button className="ghost-button" onClick={clearTarget} disabled={!hasVideo}>
                <RotateCcw size={16} />
                Clear
              </button>
            </div>
            <button className="target-frame" disabled={!hasVideo} onClick={(event) => addTargetPoint(event.clientX, event.clientY, event.currentTarget)}>
              {currentFrame ? <img src={currentFrame} alt="Target frame" /> : <EmptyMedia icon={<Video />} label="Upload video" />}
              <span className={`point-pill ${pointLabel.toLowerCase()}`}>{pointLabel}</span>
            </button>
            <div className="segmented">
              <button className={pointLabel === "Positive" ? "selected" : ""} onClick={() => setPointLabel("Positive")}>
                <CircleDot size={15} />
                Positive
              </button>
              <button className={pointLabel === "Negative" ? "selected" : ""} onClick={() => setPointLabel("Negative")}>
                <X size={15} />
                Negative
              </button>
            </div>
            <button className="primary-action secondary" onClick={startTrack} disabled={!hasVideo || isBusy}>
              {isBusy && activeJob?.type === "tracking" ? <Loader2 className="spin" size={18} /> : <Layers size={18} />}
              Track target
            </button>
          </Panel>

          <Panel title={modeLabel(mode)} icon={modeIcon(mode)} accent="violet">
            <ModeContent
              mode={mode}
              sessionId={session?.session_id}
              referenceUrl={referenceUrl}
              referencePreviewUrl={referencePreviewUrl}
              referencePointLabel={referencePointLabel}
              setReferencePointLabel={setReferencePointLabel}
              setReferenceUrl={setReferenceUrl}
              setReferencePreviewUrl={setReferencePreviewUrl}
              videoCaption={videoCaption}
              setVideoCaption={setVideoCaption}
              targetCaption={targetCaption}
              setTargetCaption={setTargetCaption}
              attrs={attrs}
              setAttrs={setAttrs}
              referenceStrategy={referenceStrategy}
              setReferenceStrategy={setReferenceStrategy}
              sketchRef={sketchRef}
            />
          </Panel>

          <Panel title="Result" icon={<Play size={18} />} accent="green">
            <div className="result-view">
              {resultUrl ? <video src={resultUrl} controls /> : <EmptyMedia icon={<Play />} label="No result yet" />}
            </div>
            <JobMeter job={activeJob} progress={progress} />
            <button className="primary-action" onClick={generate} disabled={!session || isBusy}>
              {isBusy ? <Loader2 className="spin" size={18} /> : <Sparkles size={18} />}
              Generate
            </button>
          </Panel>
        </section>
      </section>

      <button className="drawer-toggle" onClick={() => setAdvancedOpen(true)} title="Advanced settings">
        <PanelRightOpen size={21} />
      </button>
      <AdvancedDrawer
        open={advancedOpen}
        onClose={() => setAdvancedOpen(false)}
        seed={seed}
        setSeed={setSeed}
        cfgScale={cfgScale}
        setCfgScale={setCfgScale}
        dilateSize={dilateSize}
        setDilateSize={setDilateSize}
        referenceStrategy={referenceStrategy}
        job={activeJob}
        session={session}
      />
    </main>
  );
}

function modeLabel(mode: Mode) {
  if (mode === "prompt") return "Prompt Mode";
  if (mode === "image") return "Image Mode";
  return "Sketch Mode";
}

function modeIcon(mode: Mode) {
  if (mode === "prompt") return <Sparkles size={18} />;
  if (mode === "image") return <ImageIcon size={18} />;
  return <Brush size={18} />;
}

function Panel({ title, icon, accent, children }: { title: string; icon: ReactNode; accent: string; children: ReactNode }) {
  return (
    <section className={`panel ${accent}`}>
      <header className="panel-header">
        <span>{icon}</span>
        <h2>{title}</h2>
      </header>
      {children}
    </section>
  );
}

function EmptyMedia({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <div className="empty-media">
      {icon}
      <span>{label}</span>
    </div>
  );
}

function StatusRail({ steps }: { steps: Array<{ label: string; done: boolean }> }) {
  return (
    <div className="status-rail">
      {steps.map((step) => (
        <div key={step.label} className={`step ${step.done ? "done" : ""}`}>
          <span />
          {step.label}
        </div>
      ))}
    </div>
  );
}

function JobMeter({ job, progress }: { job: JobPayload | null; progress: number }) {
  return (
    <div className="job-meter">
      <div className="meter-label">
        <span>{job ? job.stage : "Idle"}</span>
        <span>{job ? job.state : "ready"}</span>
      </div>
      <div className="meter-track">
        <div style={{ width: `${Math.max(0, Math.min(100, progress))}%` }} />
      </div>
      {job?.message ? <p>{job.message}</p> : null}
      {job?.error ? <p className="error-text">{job.error}</p> : null}
    </div>
  );
}

interface ModeContentProps {
  mode: Mode;
  sessionId?: string;
  referenceUrl: string;
  referencePreviewUrl: string;
  referencePointLabel: PointLabel;
  setReferencePointLabel: (label: PointLabel) => void;
  setReferenceUrl: (url: string) => void;
  setReferencePreviewUrl: (url: string) => void;
  videoCaption: string;
  setVideoCaption: (value: string) => void;
  targetCaption: string;
  setTargetCaption: (value: string) => void;
  attrs: string;
  setAttrs: (value: string) => void;
  referenceStrategy: ReferenceStrategy;
  setReferenceStrategy: (value: ReferenceStrategy) => void;
  sketchRef: React.RefObject<SketchCanvasHandle | null>;
}

function ModeContent(props: ModeContentProps) {
  const strategyControl = props.mode === "prompt" ? null : (
    <div className="strategy-card">
      <span>Reference strategy</span>
      <div className="segmented strategy-toggle">
        {referenceStrategies.map((strategy) => (
          <button key={strategy.id} className={props.referenceStrategy === strategy.id ? "selected" : ""} onClick={() => props.setReferenceStrategy(strategy.id)}>
            {strategy.label}
          </button>
        ))}
      </div>
      <p>{referenceStrategies.find((strategy) => strategy.id === props.referenceStrategy)?.description}</p>
    </div>
  );
  const commonPrompts = (
    <div className="prompt-stack">
      {strategyControl}
      <label>
        <span>Video prompt</span>
        <textarea value={props.videoCaption} onChange={(event) => props.setVideoCaption(event.target.value)} rows={4} />
      </label>
      <label>
        <span>Target object</span>
        <input value={props.targetCaption} onChange={(event) => props.setTargetCaption(event.target.value)} />
      </label>
    </div>
  );

  if (props.mode === "prompt") {
    return <div className="mode-surface">{commonPrompts}</div>;
  }

  if (props.mode === "image") {
    const uploadReference = async (file?: File) => {
      if (!props.sessionId || !file) return;
      try {
        const payload = await api.uploadReference(props.sessionId, file);
        props.setReferenceUrl(freshAssetUrl(payload.reference_url));
        props.setReferencePreviewUrl("");
      } catch (error) {
        console.error("Reference upload failed", error);
      }
    };
    const addReferencePoint = async (clientX: number, clientY: number, container: HTMLElement) => {
      if (!props.sessionId || !props.referenceUrl) return;
      const { x, y } = mapClientToCoveredImagePoint(clientX, clientY, container.getBoundingClientRect(), naturalImageSize(container, { width: 720, height: 480 }));
      const payload = await api.addReferencePoint(props.sessionId, x, y, props.referencePointLabel);
      props.setReferencePreviewUrl(freshAssetUrl(payload.preview_url));
    };
    return (
      <div className="mode-surface split">
        <div>
          <div className="upload-line">
            <label className="file-button">
              <Upload size={17} />
              Reference
              <input
                type="file"
                accept="image/*"
                onChange={(event) => {
                  void uploadReference(event.target.files?.[0]);
                  event.currentTarget.value = "";
                }}
              />
            </label>
          </div>
          <button className="reference-frame" disabled={!props.referenceUrl} onClick={(event) => addReferencePoint(event.clientX, event.clientY, event.currentTarget)}>
            {props.referencePreviewUrl || props.referenceUrl ? (
              <img src={props.referencePreviewUrl || props.referenceUrl} alt="Reference" />
            ) : (
              <EmptyMedia icon={<ImageIcon />} label="Reference" />
            )}
          </button>
          <div className="segmented compact">
            <button className={props.referencePointLabel === "Positive" ? "selected" : ""} onClick={() => props.setReferencePointLabel("Positive")}>
              Positive
            </button>
            <button className={props.referencePointLabel === "Negative" ? "selected" : ""} onClick={() => props.setReferencePointLabel("Negative")}>
              Negative
            </button>
          </div>
        </div>
        {commonPrompts}
      </div>
    );
  }

  return (
    <div className="mode-surface sketch-mode">
      <SketchCanvas ref={props.sketchRef} />
      <div className="prompt-stack">
        {strategyControl}
        <label>
          <span>Target object</span>
          <input value={props.targetCaption} onChange={(event) => props.setTargetCaption(event.target.value)} />
        </label>
        <label>
          <span>Material and style</span>
          <input value={props.attrs} onChange={(event) => props.setAttrs(event.target.value)} />
        </label>
        <label>
          <span>Video prompt</span>
          <textarea value={props.videoCaption} onChange={(event) => props.setVideoCaption(event.target.value)} rows={3} />
        </label>
      </div>
    </div>
  );
}

export interface SketchCanvasHandle {
  exportPng: () => string;
}

const SketchCanvas = forwardRef<SketchCanvasHandle>((_, ref) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [brush, setBrush] = useState(12);
  const [tool, setTool] = useState<"brush" | "eraser">("brush");
  const historyRef = useRef<ImageData[]>([]);
  const redoRef = useRef<ImageData[]>([]);
  const [, forceHistoryPaint] = useState(0);
  const drawing = useRef(false);

  const getCanvasContext = useCallback(() => canvasRef.current?.getContext("2d", { willReadFrequently: true }), []);

  const snapshot = useCallback(() => {
    const canvas = canvasRef.current;
    const context = getCanvasContext();
    if (!canvas || !context) return;
    historyRef.current = [...historyRef.current.slice(-15), context.getImageData(0, 0, canvas.width, canvas.height)];
    redoRef.current = [];
    forceHistoryPaint((value) => value + 1);
  }, [getCanvasContext]);

  const clear = () => {
    const canvas = canvasRef.current;
    const context = getCanvasContext();
    if (!canvas || !context) return;
    snapshot();
    context.fillStyle = "#f7f8fb";
    context.fillRect(0, 0, canvas.width, canvas.height);
  };

  useEffect(() => {
    clear();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useImperativeHandle(ref, () => ({
    exportPng: () => canvasRef.current?.toDataURL("image/png") || "",
  }));

  const point = (event: PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) / rect.width) * canvas.width,
      y: ((event.clientY - rect.top) / rect.height) * canvas.height,
    };
  };

  const start = (event: PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const context = getCanvasContext();
    if (!canvas || !context) return;
    snapshot();
    drawing.current = true;
    canvas.setPointerCapture(event.pointerId);
    const p = point(event);
    context.beginPath();
    context.moveTo(p.x, p.y);
  };

  const move = (event: PointerEvent<HTMLCanvasElement>) => {
    if (!drawing.current) return;
    const canvas = canvasRef.current;
    const context = getCanvasContext();
    if (!canvas || !context) return;
    const p = point(event);
    context.lineCap = "round";
    context.lineJoin = "round";
    context.lineWidth = brush;
    context.strokeStyle = tool === "brush" ? "#05070b" : "#f7f8fb";
    context.lineTo(p.x, p.y);
    context.stroke();
  };

  const stop = () => {
    drawing.current = false;
  };

  const undo = () => {
    const canvas = canvasRef.current;
    const context = getCanvasContext();
    const previous = historyRef.current.at(-1);
    if (!canvas || !context || !previous) return;
    redoRef.current = [...redoRef.current, context.getImageData(0, 0, canvas.width, canvas.height)];
    context.putImageData(previous, 0, 0);
    historyRef.current = historyRef.current.slice(0, -1);
    forceHistoryPaint((value) => value + 1);
  };

  const redo = () => {
    const canvas = canvasRef.current;
    const context = getCanvasContext();
    const next = redoRef.current.at(-1);
    if (!canvas || !context || !next) return;
    historyRef.current = [...historyRef.current, context.getImageData(0, 0, canvas.width, canvas.height)];
    context.putImageData(next, 0, 0);
    redoRef.current = redoRef.current.slice(0, -1);
    forceHistoryPaint((value) => value + 1);
  };

  return (
    <div className="sketch-board">
      <div className="tool-strip">
        <button className={tool === "brush" ? "selected" : ""} onClick={() => setTool("brush")} title="Brush">
          <Brush size={17} />
        </button>
        <button className={tool === "eraser" ? "selected" : ""} onClick={() => setTool("eraser")} title="Eraser">
          <Eraser size={17} />
        </button>
        <button onClick={undo} title="Undo">
          <Undo2 size={17} />
        </button>
        <button onClick={redo} title="Redo">
          <RotateCcw size={17} />
        </button>
        <button onClick={clear} title="Clear">
          <Trash2 size={17} />
        </button>
        <label className="brush-size">
          <Square size={14} />
          <input type="range" min="2" max="40" value={brush} onChange={(event) => setBrush(Number(event.target.value))} />
        </label>
      </div>
      <canvas ref={canvasRef} width={720} height={480} onPointerDown={start} onPointerMove={move} onPointerUp={stop} onPointerCancel={stop} />
    </div>
  );
});

function AdvancedDrawer({
  open,
  onClose,
  seed,
  setSeed,
  cfgScale,
  setCfgScale,
  dilateSize,
  setDilateSize,
  referenceStrategy,
  job,
  session,
}: {
  open: boolean;
  onClose: () => void;
  seed: number;
  setSeed: (value: number) => void;
  cfgScale: number;
  setCfgScale: (value: number) => void;
  dilateSize: number;
  setDilateSize: (value: number) => void;
  referenceStrategy: ReferenceStrategy;
  job: JobPayload | null;
  session: SessionPayload | null;
}) {
  return (
    <aside className={`advanced-drawer ${open ? "open" : ""}`}>
      <header>
        <div>
          <p className="eyebrow">Advanced</p>
          <h2>Debug and controls</h2>
        </div>
        <button onClick={onClose} title="Close">
          <X size={20} />
        </button>
      </header>
      <label>
        <span>Seed</span>
        <input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
      </label>
      <label>
        <span>CFG scale</span>
        <input type="range" min="1" max="10" step="0.1" value={cfgScale} onChange={(event) => setCfgScale(Number(event.target.value))} />
        <strong>{cfgScale.toFixed(1)}</strong>
      </label>
      <label>
        <span>Dilate size</span>
        <input type="range" min="0" max="32" step="1" value={dilateSize} onChange={(event) => setDilateSize(Number(event.target.value))} />
        <strong>{dilateSize}</strong>
      </label>
      <div className="debug-box">
        <span>Session</span>
        <code>{session?.session_id || "none"}</code>
      </div>
      <div className="debug-box">
        <span>Job</span>
        <code>{job ? `${job.type}:${job.state}` : "idle"}</code>
      </div>
      <div className="debug-box">
        <span>Sketch defaults</span>
        <code>
          {referenceStrategy === "lama_background"
            ? "lama_background / video_target / propagation / lama_cleaned_video"
            : "mask_twist / video_target / propagation / full_video"}
        </code>
      </div>
    </aside>
  );
}

export default App;
