"""AI Vehicle Detection System - Streamlit user interface.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import streamlit as st

from detector import ModelLoadError, VehicleDetector
from reports import build_run_metadata, build_summary_table, load_frame_records, save_summary_csv, summary_csv_bytes
from tracker import TRACKING_DISCLAIMER, tracking_available
from utils import (
    AVAILABLE_MODELS,
    DEFAULT_CONFIDENCE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU,
    DEFAULT_MODEL,
    MODELS_DIR,
    OUTPUT_DIR,
    REPORTS_DIR,
    SUPPORTED_VIDEO_EXTENSIONS,
    VEHICLE_CLASS_NAMES,
    InsufficientDiskSpaceError,
    VideoInfo,
    VideoValidationError,
    cuda_available,
    describe_device,
    ensure_directories,
    format_seconds,
    probe_video,
    save_uploaded_bytes,
    transcode_to_h264,
)
from video_processor import ProcessingConfig, ProcessingProgress, ProcessingResult, VideoProcessor, stream_base_name

CUSTOM_MODEL_OPTION = "Custom weights (.pt path)…"
CLASS_EMOJI = {"car": "🚗", "motorcycle": "🏍️", "bus": "🚌", "truck": "🚚"}

st.set_page_config(page_title="AI Vehicle Detection System", page_icon="🚗", layout="wide")
ensure_directories()


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False, max_entries=3)
def load_detector(model_name: str, device: str, imgsz: int) -> VehicleDetector:
    """Load the YOLO model once per (model, device, image size) and reuse it across runs."""
    return VehicleDetector(model_name=model_name, device=device, imgsz=imgsz)


@st.cache_data(show_spinner=False)
def check_tracking() -> tuple[bool, str]:
    return tracking_available()


# --------------------------------------------------------------------------- #
# Sidebar: configuration
# --------------------------------------------------------------------------- #
def sidebar_settings() -> dict:
    st.sidebar.header("⚙️ Model configuration")
    local_weights = sorted(p.name for p in MODELS_DIR.glob("*.pt") if p.name not in AVAILABLE_MODELS)
    model_options = list(AVAILABLE_MODELS) + local_weights + [CUSTOM_MODEL_OPTION]
    model_choice = st.sidebar.selectbox(
        "YOLO model",
        model_options,
        index=model_options.index(DEFAULT_MODEL),
        help="'n' (nano) is fastest; 's'/'m' are more accurate but slower. Official weights download automatically "
        "into the models/ folder on first use.",
    )
    model_name = model_choice
    if model_choice == CUSTOM_MODEL_OPTION:
        model_name = st.sidebar.text_input("Path to .pt weights", placeholder=r"C:\models\my_model.pt").strip()

    has_cuda = cuda_available()
    device_options = ["auto", "cpu"] + (["cuda"] if has_cuda else [])
    device = st.sidebar.selectbox(
        "Device",
        device_options,
        help="'auto' uses an NVIDIA GPU when PyTorch can see one, otherwise the CPU.",
    )
    if not has_cuda:
        st.sidebar.caption("No CUDA GPU detected by PyTorch - running on CPU.")
    imgsz = st.sidebar.select_slider(
        "Inference image size", options=[320, 416, 480, 640, 800, 960, 1280], value=DEFAULT_IMAGE_SIZE,
        help="Larger sizes detect small/distant vehicles better but are slower.",
    )

    st.sidebar.header("🎯 Detection settings")
    confidence = st.sidebar.slider("Confidence threshold", 0.05, 0.95, DEFAULT_CONFIDENCE, 0.05)
    iou = st.sidebar.slider("NMS IoU threshold", 0.10, 0.90, DEFAULT_IOU, 0.05,
                            help="Lower values merge overlapping boxes more aggressively.")
    mode = st.sidebar.radio("Detection mode", ["Vehicles only", "All COCO classes (80)"])
    all_classes = mode.startswith("All")
    selected_classes = st.sidebar.multiselect(
        "Vehicle classes",
        list(VEHICLE_CLASS_NAMES),
        default=list(VEHICLE_CLASS_NAMES),
        format_func=lambda n: f"{CLASS_EMOJI.get(n, '')} {n}",
        disabled=all_classes,
    )

    st.sidebar.header("🧭 Tracking & reports")
    track_ok, track_msg = check_tracking()
    enable_tracking = st.sidebar.toggle(
        "Enable ByteTrack tracking", value=False, disabled=not track_ok,
        help="Assigns IDs to vehicles across frames to estimate unique vehicles. " + (track_msg if not track_ok else ""),
    )
    if not track_ok:
        st.sidebar.caption(f"⚠️ {track_msg}")
    save_records = st.sidebar.toggle("Save frame-level detection records (CSV)", value=True)
    max_frames = st.sidebar.number_input(
        "Max frames to process (0 = whole video)", min_value=0, value=0, step=100,
        help="Useful for a quick test on long videos.",
    )

    return dict(
        model_name=model_name, device=device, imgsz=int(imgsz), confidence=float(confidence), iou=float(iou),
        all_classes=all_classes, selected_classes=selected_classes, enable_tracking=enable_tracking,
        save_records=save_records, max_frames=int(max_frames),
    )


def get_detector(settings: dict) -> VehicleDetector | None:
    """Load (or fetch from cache) the detector, showing friendly errors."""
    if not settings["model_name"]:
        st.error("Enter the path to your custom .pt weights in the sidebar.")
        return None
    try:
        with st.spinner(f"Loading {Path(settings['model_name']).name} (downloads on first use)…"):
            return load_detector(settings["model_name"], settings["device"], settings["imgsz"])
    except (ModelLoadError, ValueError) as exc:
        st.error(f"❌ {exc}")
    except Exception as exc:  # unexpected (e.g. broken torch install)
        st.error(f"❌ Unexpected error while loading the model: {exc}")
    return None


def build_config(settings: dict, detector: VehicleDetector) -> ProcessingConfig | None:
    if settings["all_classes"]:
        class_ids = None
    else:
        if not settings["selected_classes"]:
            st.warning("Select at least one vehicle class, or switch to 'All COCO classes'.")
            return None
        try:
            class_ids = detector.class_ids_for(settings["selected_classes"])
        except ModelLoadError as exc:
            st.error(f"❌ {exc}")
            return None
    return ProcessingConfig(
        confidence=settings["confidence"],
        iou=settings["iou"],
        class_ids=class_ids,
        enable_tracking=settings["enable_tracking"],
        save_frame_records=settings["save_records"],
        max_frames=settings["max_frames"],
    )


# --------------------------------------------------------------------------- #
# Input selection
# --------------------------------------------------------------------------- #
def select_input_video() -> Path | None:
    source = st.radio("Video source", ["Upload a video", "Video file on this computer"], horizontal=True)
    if source == "Upload a video":
        uploaded = st.file_uploader(
            "Upload a video (MP4, AVI or MOV)", type=[ext.lstrip(".") for ext in SUPPORTED_VIDEO_EXTENSIONS]
        )
        if uploaded is None:
            return None
        # Save each upload only once, not on every Streamlit rerun.
        key = f"{uploaded.name}-{uploaded.size}"
        saved = st.session_state.setdefault("saved_uploads", {})
        if key not in saved or not Path(saved[key]).is_file():
            try:
                saved[key] = str(save_uploaded_bytes(uploaded.getbuffer(), uploaded.name))
            except (VideoValidationError, InsufficientDiskSpaceError) as exc:
                st.error(f"❌ {exc}")
                return None
        return Path(saved[key])

    raw = st.text_input("Full path to a video file", placeholder=r"C:\Videos\traffic.mp4  or  /home/me/traffic.mp4")
    raw = raw.strip().strip('"').strip("'")
    return Path(raw).expanduser() if raw else None


def show_video_info(info: VideoInfo) -> None:
    cols = st.columns(4)
    cols[0].metric("Resolution", info.resolution)
    cols[1].metric("Frame rate", f"{info.fps:.2f} fps")
    cols[2].metric("Frames", f"{info.frame_count:,}" if info.frame_count else "unknown")
    cols[3].metric("Duration", format_seconds(info.duration_s) if info.duration_s else "unknown")


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
def render_results(result: ProcessingResult, key: str) -> None:
    st.subheader("📊 Results")
    for warning in result.warnings:
        st.warning(warning)

    output = Path(result.output_path)
    if output.is_file():
        if result.browser_compatible:
            st.video(str(output))
        else:
            st.info("The browser preview may not play this mp4v file; download it and open it in VLC or another player.")

    st.markdown("##### Detections across frames")
    st.caption(
        "These are **frame-level detections**: a vehicle visible for 100 frames counts 100 times. "
        "They are not counts of unique vehicles."
    )
    counts = result.detection_counts
    row1 = st.columns(5)
    row1[0].metric("Total vehicle detections", f"{result.total_vehicle_detections:,}")
    for col, name in zip(row1[1:], VEHICLE_CLASS_NAMES):
        col.metric(f"{CLASS_EMOJI[name]} {name.capitalize()} detections", f"{counts.get(name, 0):,}")
    if result.config.class_ids is None and result.total_detections != result.total_vehicle_detections:
        st.caption(f"All-classes mode: {result.total_detections:,} detections in total including non-vehicle objects.")

    if result.unique_counts is not None:
        st.markdown("##### Unique tracked vehicles (estimate)")
        row_u = st.columns(5)
        unique_vehicles = sum(result.unique_counts.get(n, 0) for n in VEHICLE_CLASS_NAMES)
        row_u[0].metric("Unique vehicle track IDs", f"{unique_vehicles:,}")
        for col, name in zip(row_u[1:], VEHICLE_CLASS_NAMES):
            col.metric(f"{CLASS_EMOJI[name]} Unique {name}s", f"{result.unique_counts.get(name, 0):,}")
        st.info(TRACKING_DISCLAIMER, icon="ℹ️")

    st.markdown("##### Performance")
    row2 = st.columns(4)
    row2[0].metric("Frames processed", f"{result.frames_processed:,}")
    row2[1].metric("Average processing FPS", f"{result.average_fps:.1f}")
    row2[2].metric("Processing time", format_seconds(result.processing_time_s))
    row2[3].metric("Device", result.device)

    st.markdown("##### Detection summary")
    summary_df = build_summary_table(result)
    st.dataframe(summary_df, hide_index=True)

    st.markdown("##### Downloads")
    d1, d2, d3, d4 = st.columns(4)
    if output.is_file():
        with open(output, "rb") as fh:
            d1.download_button("⬇️ Annotated video", fh.read(), file_name=output.name, mime="video/mp4",
                                key=f"{key}_dl_video")
    d2.download_button(
        "⬇️ Summary CSV", summary_csv_bytes(result),
        file_name=f"{Path(result.input_path).stem}_summary.csv", mime="text/csv", key=f"{key}_dl_summary",
    )
    if result.frame_records_path and Path(result.frame_records_path).is_file():
        records = Path(result.frame_records_path)
        with open(records, "rb") as fh:
            d3.download_button("⬇️ Frame-level records CSV", fh.read(), file_name=records.name, mime="text/csv",
                                key=f"{key}_dl_records")
    if result.track_table is not None and not result.track_table.empty:
        d4.download_button(
            "⬇️ Track table CSV", result.track_table.to_csv(index=False).encode("utf-8"),
            file_name=f"{Path(result.input_path).stem}_tracks.csv", mime="text/csv", key=f"{key}_dl_tracks",
        )

    if st.button("💾 Save summary report to reports/ folder", key=f"{key}_save_report"):
        path = save_summary_csv(result)
        st.success(f"Saved {path}")

    with st.expander("Run details"):
        st.dataframe(build_run_metadata(result), hide_index=True)
        st.caption(f"Output video: {result.output_path}")
        if result.frame_records_path:
            st.caption(f"Frame records: {result.frame_records_path}")

    if result.frame_records_path:
        with st.expander("Frame-level detection records (first 500 rows)"):
            st.dataframe(load_frame_records(Path(result.frame_records_path), max_rows=500), hide_index=True)
    if result.track_table is not None and not result.track_table.empty:
        with st.expander("Tracked objects"):
            st.dataframe(result.track_table, hide_index=True)


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def video_tab(settings: dict) -> None:
    video_path = select_input_video()
    info: VideoInfo | None = None
    if video_path is not None:
        try:
            info = probe_video(video_path)
            show_video_info(info)
        except VideoValidationError as exc:
            st.error(f"❌ {exc}")

    start = st.button("▶️ Start Detection", type="primary", disabled=info is None)
    if start and info is not None and video_path is not None:
        detector = get_detector(settings)
        if detector is None:
            return
        config = build_config(settings, detector)
        if config is None:
            return
        st.session_state.pop("result", None)
        run_video(detector, config, video_path, info)

    if "result" in st.session_state:
        st.divider()
        render_results(st.session_state["result"], key="video")
        if st.button("🗑️ Clear results"):
            st.session_state.pop("result", None)
            st.rerun()


def run_video(detector: VehicleDetector, config: ProcessingConfig, video_path: Path, info: VideoInfo) -> None:
    status = st.status(f"Processing on {describe_device(detector.device)}…", expanded=True)
    progress_bar = status.progress(0.0, text="Starting…")
    preview = status.empty()
    last_update = [0.0]

    def on_progress(p: ProcessingProgress) -> None:
        now = time.perf_counter()
        if now - last_update[0] < 0.25 and p.preview_bgr is None and p.frames_processed != p.total_frames:
            return  # throttle UI updates
        last_update[0] = now
        fps = p.frames_processed / p.elapsed_s if p.elapsed_s > 0 else 0.0
        if p.total_frames > 0:
            remaining = (p.total_frames - p.frames_processed) / fps if fps > 0 else 0
            text = (f"Frame {p.frames_processed:,}/{p.total_frames:,} · {fps:.1f} FPS · "
                    f"~{format_seconds(remaining)} left · {p.vehicle_detections:,} vehicle detections")
            progress_bar.progress(p.fraction, text=text)
        else:
            progress_bar.progress(0.0, text=f"Frame {p.frames_processed:,} · {fps:.1f} FPS")
        if p.preview_bgr is not None:
            preview.image(cv2.cvtColor(p.preview_bgr, cv2.COLOR_BGR2RGB), caption="Live preview")

    try:
        result = VideoProcessor(detector).process_video(video_path, config, progress_callback=on_progress)
    except (VideoValidationError, InsufficientDiskSpaceError) as exc:
        status.update(label="Processing failed", state="error")
        st.error(f"❌ {exc}")
        return
    except Exception as exc:
        status.update(label="Processing failed", state="error")
        st.error(f"❌ Unexpected error during processing: {exc}")
        st.caption("If you are using a GPU and ran out of memory, choose a smaller model/image size or the CPU.")
        return

    progress_bar.progress(1.0, text="Encoding complete")
    preview.empty()
    status.update(
        label=f"✅ Done - {result.frames_processed:,} frames in {format_seconds(result.processing_time_s)} "
        f"({result.average_fps:.1f} FPS)",
        state="complete",
        expanded=False,
    )
    st.session_state["result"] = result


def live_tab(settings: dict) -> None:
    st.markdown(
        "Run detection on a webcam or network stream (RTSP/HTTP). The camera must be attached to the machine "
        "running Streamlit. Click **Stop** (or wait for the time limit) to finish."
    )
    kind = st.radio("Source", ["Webcam", "Stream URL"], horizontal=True)
    source: int | str
    if kind == "Webcam":
        source = int(st.number_input("Camera index", min_value=0, max_value=10, value=0))
    else:
        source = st.text_input("Stream URL", placeholder="rtsp://user:pass@192.168.1.10:554/stream").strip()
    max_seconds = st.slider("Time limit (seconds)", 10, 600, 60, 10)
    save_output = st.toggle("Save annotated recording", value=True)

    c1, c2 = st.columns([1, 1])
    start = c1.button("▶️ Start camera", type="primary", disabled=(kind == "Stream URL" and not source))
    c2.button("⏹️ Stop")  # Any click reruns the script, which interrupts the running loop below.

    pending = st.session_state.get("live_session")
    if not start and pending:
        finish_interrupted_live_session(pending)

    if start:
        detector = get_detector(settings)
        if detector is None:
            return
        config = build_config(settings, detector)
        if config is None:
            return
        base = stream_base_name(source)
        session = {"base": base, "frames": 0, "vehicle_detections": 0, "elapsed": 0.0, "save": save_output}
        st.session_state["live_session"] = session
        frame_slot = st.empty()
        stats_slot = st.empty()

        def on_frame(frame, progress: ProcessingProgress) -> None:
            fps = progress.frames_processed / progress.elapsed_s if progress.elapsed_s > 0 else 0.0
            frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            stats_slot.caption(
                f"Frames: {progress.frames_processed:,} · {fps:.1f} FPS · "
                f"vehicle detections: {progress.vehicle_detections:,} · {progress.elapsed_s:.0f}s / {max_seconds}s"
            )
            session.update(frames=progress.frames_processed, vehicle_detections=progress.vehicle_detections,
                           elapsed=progress.elapsed_s)

        try:
            result = VideoProcessor(detector).process_stream(
                source, config, on_frame, max_seconds=max_seconds, save_output=save_output, base_name=base
            )
        except (VideoValidationError, InsufficientDiskSpaceError) as exc:
            st.session_state.pop("live_session", None)
            st.error(f"❌ {exc}")
            return
        st.session_state.pop("live_session", None)
        st.session_state["live_result"] = result

    if "live_result" in st.session_state:
        st.divider()
        render_results(st.session_state["live_result"], key="live")


def finish_interrupted_live_session(session: dict) -> None:
    """After Stop interrupted a live run, convert its recording and show what was captured."""
    st.session_state.pop("live_session", None)
    st.session_state.pop("live_result", None)
    st.success(
        f"Live session stopped after {session['frames']:,} frames ({session['elapsed']:.0f}s), "
        f"{session['vehicle_detections']:,} vehicle detections across frames."
    )
    raw = OUTPUT_DIR / f"{session['base']}_mp4v.mp4"
    final = OUTPUT_DIR / f"{session['base']}.mp4"
    if session["save"] and raw.is_file() and raw.stat().st_size > 0:
        with st.spinner("Finalising recording…"):
            converted = transcode_to_h264(raw, final)
        if converted:
            raw.unlink(missing_ok=True)
        video = final if converted else raw
        if converted:
            st.video(str(video))
        with open(video, "rb") as fh:
            st.download_button("⬇️ Download recording", fh.read(), file_name=video.name, mime="video/mp4",
                               key="live_stopped_dl_video")
    records = REPORTS_DIR / f"{session['base']}_frames.csv"
    if records.is_file():
        with open(records, "rb") as fh:
            st.download_button("⬇️ Frame-level records CSV", fh.read(), file_name=records.name, mime="text/csv",
                               key="live_stopped_dl_records")


def about_tab() -> None:
    st.markdown(
        f"""
**Pipeline:** video → OpenCV reads one frame at a time → YOLO detects objects (restricted to the selected
classes) → optional ByteTrack assigns IDs → boxes, labels and confidences are drawn → frames are written to an
MP4 → the file is re-encoded to H.264 so browsers can play it → statistics and CSV reports are produced.

**Classes (COCO IDs):** car (2), motorcycle (3), bus (5), truck (7).

**Detections vs. unique vehicles:** detection counts are summed over every frame. Unique counts are only
available with tracking enabled and are estimates. {TRACKING_DISCLAIMER}

**Files:** uploads are stored in `input_videos/`, annotated videos in `output_videos/`, CSV reports in
`reports/`, and model weights in `models/`.
"""
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("🚗 AI Vehicle Detection System")
    st.markdown(
        "Detect **cars, motorcycles, buses and trucks** in videos with a pretrained YOLO model. "
        "Upload a clip, tune the settings in the sidebar, and download the annotated video and reports."
    )
    settings = sidebar_settings()
    tab_video, tab_live, tab_about = st.tabs(["📁 Video file", "🎥 Live camera", "ℹ️ About"])
    with tab_video:
        video_tab(settings)
    with tab_live:
        live_tab(settings)
    with tab_about:
        about_tab()


main()
