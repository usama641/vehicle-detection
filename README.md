# AI Vehicle Detection System

Detect **cars, motorcycles, buses and trucks** in videos and live camera streams with a pretrained
Ultralytics YOLO model and OpenCV. A **Next.js** web app (`web/`) talks to a **FastAPI** backend (`api/`) that
draws colour-coded bounding boxes with class labels and confidence scores, saves an annotated MP4 you can play
in a browser, and reports detection statistics. Optional ByteTrack tracking estimates how many unique vehicles
appeared.

```
vehicle_detection/
├── web/   # Frontend - Next.js (React, TypeScript, Tailwind CSS)
└── api/   # Backend  - FastAPI + YOLO/OpenCV detection pipeline
```

---

## 1. Architecture

```
  ┌──────────────────┐  REST: settings, upload  ┌──────────────┐   job queue   ┌──────────────────┐
  │ web/  (Next.js)  │─────────────────────────▶│ api/main.py  │──────────────▶│ api/jobs.py      │
  │ localhost:3000   │◀─ status · MJPEG preview ─│  (FastAPI)   │◀── progress ───│ 1 worker thread  │
  └──────────────────┘   annotated MP4 · CSVs    └──────────────┘               └────────┬─────────┘
                                                                                         ▼
         cv2.VideoCapture ── frame (BGR) ──▶ video_processor.VideoProcessor (one frame at a time)
                                                        │
                                                        ▼
                                              ┌──────────────────┐
                                              │ detector.py      │  YOLO.predict() / YOLO.track()
                                              │ VehicleDetector  │  (ByteTrack via tracker.py)
                                              └───────┬──────────┘
                                                      ▼
                                  draw boxes ▶ cv2.VideoWriter (mp4v) ▶ ffmpeg ▶ H.264 MP4
                                                      ▼
                                             reports.py (Pandas CSV)
```

**Pipeline per video**

1. In the web app the user uploads a video (or gives a path on the API machine / a camera index).
2. The API validates the file (extension, can be opened, first frame decodes) and returns its metadata.
3. Clicking *Start Detection* creates a **job**. Jobs run one at a time on a single worker thread, so the
   cached model and its tracker state are never shared between concurrent runs.
4. The YOLO model is loaded **once** per (model, device, image size) and cached by the job manager.
5. OpenCV reads the video **one frame at a time** (long videos never sit in memory).
6. YOLO runs on the frame, restricted to the selected classes (`classes=[2, 3, 5, 7]`) so unrelated classes
   are dropped inside the model's NMS step. With tracking on, `model.track(persist=True, tracker="bytetrack.yaml")` is used.
7. Boxes, `class confidence` labels (plus `#ID` when tracking) and a small info panel are drawn.
   The web app polls the job status and shows annotated frames through an MJPEG stream.
8. Frames are written at the original resolution and FPS to an MPEG-4 intermediate file.
9. The bundled ffmpeg (from `imageio-ffmpeg`) re-encodes it to **H.264/yuv420p** so browsers and common
   players can play it. If that fails the mp4v file is kept (plays in VLC) and the UI says so.
10. Statistics, summary tables and CSV downloads are shown.

Because processing happens in a background job (not in the browser request), you can switch tabs, reload
the page or close it and come back; the web app reconnects to the last job.

### Project structure

```
vehicle_detection/
├── README.md
├── web/                       # FRONTEND - Next.js 16 (App Router, TypeScript, Tailwind CSS)
│   ├── src/app/               # layout.tsx, page.tsx (settings sidebar + tabs), globals.css
│   ├── src/components/        # SettingsPanel, VideoTab, LiveTab, JobMonitor, Results, AboutTab, ui
│   ├── src/hooks/             # useJob (status polling), useStoredState (localStorage)
│   ├── src/lib/               # api.ts (typed API client), format.ts
│   ├── .env.example           # NEXT_PUBLIC_API_URL
│   └── package.json
└── api/                       # BACKEND - FastAPI + detection pipeline
    ├── main.py                # REST endpoints (config, uploads, jobs, previews, downloads)
    ├── jobs.py                # Background job queue, progress, preview frames, model cache
    ├── detector.py            # VehicleDetector: model loading/download, device, inference, box drawing
    ├── video_processor.py     # Frame loop for files and live streams, video writing, statistics
    ├── tracker.py             # ByteTrack config, availability check, unique-ID statistics
    ├── reports.py             # Pandas summary tables and CSV export
    ├── utils.py               # Constants, paths, validation, device selection, disk checks, ffmpeg
    ├── validate_setup.py      # CLI environment check + end-to-end smoke test (no UI)
    ├── streamlit_app.py       # Optional legacy Streamlit UI (same pipeline)
    ├── tests/                 # pytest: test_pipeline.py (pipeline), test_api.py (REST API)
    ├── requirements.txt
    ├── models/                # YOLO weights (downloaded automatically)
    ├── input_videos/          # saved uploads
    ├── output_videos/         # annotated videos
    └── reports/               # CSV reports (summary + frame-level records)
```

### REST API

Interactive documentation: <http://127.0.0.1:8000/docs>.

| Method & path | Purpose |
|---|---|
| `GET /api/config` | Models, devices, image sizes, defaults, vehicle classes, tracking availability |
| `POST /api/videos` | Upload a video (multipart `file`); returns `video_id` + metadata |
| `POST /api/videos/local` | Use a video already on the API machine (`{"path": ...}`) |
| `POST /api/jobs/video` | Start processing (`{"video_id", "settings"}`) |
| `POST /api/jobs/live` | Start a webcam/stream session (`{"source", "max_seconds", "save_output", "settings"}`) |
| `GET /api/jobs/{id}` | Status, progress and (when done) full results |
| `POST /api/jobs/{id}/cancel` | Cancel a video job / stop a live session (recording is still finalised) |
| `GET /api/jobs/{id}/stream` | MJPEG stream of annotated frames (`<img src>`) |
| `GET /api/jobs/{id}/video` | Annotated MP4 (`?download=true` for an attachment) |
| `GET /api/jobs/{id}/summary.csv`, `records.csv`, `tracks.csv` | CSV downloads |
| `GET /api/jobs/{id}/records?limit=500` | Frame-level records as JSON |
| `POST /api/jobs/{id}/report` | Save the summary report to `api/reports/` |

---

## 2. Prerequisites

- **Python 3.10, 3.11, 3.12 or 3.13** (64-bit) for the API. Check with `python --version`.
  - Windows: install from <https://www.python.org/downloads/> and tick **"Add python.exe to PATH"**.
- **Node.js 20.9 or newer** for the web app. Check with `node --version` (install from <https://nodejs.org/>).
- ~3 GB free disk space (PyTorch is large) plus room for output videos.
- Internet access on first run (to download packages and the ~6 MB `yolov8n.pt` model).
- Optional: an NVIDIA GPU with a recent driver for GPU inference.

## 3. Installation

### Backend (api/)

Windows (PowerShell):

```powershell
cd "C:\Users\My PC\vehicle_detection"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r api\requirements.txt
```

If PowerShell blocks activation, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
(or use `.\.venv\Scripts\activate.bat` from cmd.exe).

Linux / macOS:

```bash
cd vehicle_detection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r api/requirements.txt
```

On minimal Linux servers OpenCV may need system libraries: `sudo apt install libgl1 libglib2.0-0`.

### Frontend (web/)

```bash
cd web
npm install
```

The web app calls the API at `http://127.0.0.1:8000` by default. To change it, copy `web/.env.example` to
`web/.env.local` and edit `NEXT_PUBLIC_API_URL` (restart `npm run dev` afterwards). If the web app is served
from another origin than `http://localhost:3000`, allow it on the API with the `VD_CORS_ORIGINS` environment
variable (comma-separated list).

### YOLO model

Nothing to do: the first time a model is used, Ultralytics downloads it into `api/models/`
(e.g. `api/models/yolov8n.pt`). To work **offline**, download the weights manually from
<https://github.com/ultralytics/assets/releases> (e.g. `yolov8n.pt`, `yolo11n.pt`) and put them in `api/models/`.
Any custom `.pt` file in `api/models/` also appears in the model dropdown.

| Model | Speed | Accuracy | Notes |
|---|---|---|---|
| `yolov8n.pt` (default) | fastest | good | best choice on CPU |
| `yolo11n.pt` | fastest | slightly better | newer architecture, same API |
| `yolov8s.pt` / `yolo11s.pt` | medium | better | good on GPU |
| `yolov8m.pt` / `yolo11m.pt` | slow on CPU | best of the list | GPU recommended |

## 4. Running the application

Use two terminals.

**Terminal 1 - API** (with the virtual environment activated):

```bash
cd api
uvicorn main:app --host 127.0.0.1 --port 8000
```

API docs: <http://127.0.0.1:8000/docs>.

**Terminal 2 - web app:**

```bash
cd web
npm run dev          # development, http://localhost:3000
# or: npm run build && npm start   (production build)
```

Open <http://localhost:3000>. Stop either server with `Ctrl+C`.

The original Streamlit UI still works against the same pipeline: `cd api` then `streamlit run streamlit_app.py`.

## 5. Sample detection workflow

1. Get a traffic video (e.g. a free dashcam/traffic clip from Pexels or Pixabay), or use any MP4/AVI/MOV you have.
2. Start the API and the web app (§4) and open <http://localhost:3000>.
3. In the settings sidebar keep `yolov8n.pt`, device `auto`, confidence `0.40`, mode **Vehicles only** with all
   four classes. For a quick test set **Max frames** to `300`.
4. In **📁 Video file**, drop/upload the clip (or choose *Video file on the API machine* and paste the full path,
   which avoids uploading large files). The resolution, FPS, frame count and duration appear.
5. Click **▶️ Start Detection**. The progress bar shows frames, FPS and time remaining, with a live preview.
   **✖ Cancel** stops the job.
6. When finished you get the annotated video, detection metrics, performance numbers and a summary table.
7. Download the annotated video, summary CSV and frame-level CSV, or click
   *Save summary report to reports/ folder*.
8. Optional: turn on **Enable ByteTrack tracking** and run again to see `#ID` labels and unique-vehicle estimates.

### Without the UI

```bash
cd api
python validate_setup.py                              # checks + generated sample video
python validate_setup.py input_videos/traffic.mp4 --track --max-frames 300
```

### Live camera

Open the **🎥 Live camera** tab, choose a webcam index (usually `0`) or an RTSP/HTTP URL, set a time limit and
click **Start camera**. **Stop** ends the session; the recording is finalised and offered for download.
The camera must be attached to the computer running the API (not the browser's machine).

## 6. Understanding the statistics

| Metric | Meaning |
|---|---|
| **Vehicle detections (across frames)** | Sum of boxes over every frame. A car visible for 100 frames counts **100 times**. This is *not* a count of vehicles. |
| Car / Motorcycle / Bus / Truck detections | Same, per class. |
| **Unique tracked IDs (estimate)** | Only with tracking. Number of distinct ByteTrack IDs, with each ID assigned to its most frequent class. |
| Frames processed | Frames read and analysed. |
| Average processing FPS | Frames processed ÷ processing time (detection + drawing + writing). |
| Processing time | Wall-clock time of the frame loop (excludes the final H.264 encode). |

**Tracking caveat:** tracking IDs can change when a vehicle is occluded, leaves and re-enters the frame, or is
missed for several frames, so one real vehicle can receive several IDs. Unique counts are estimates and IDs do
not identify real-world vehicles.

Frame-level CSV columns: `frame_index, timestamp_s, class_id, class_name, confidence, x1, y1, x2, y2, track_id`.

## 7. CPU and GPU configuration

- **Device = auto** uses `cuda:0` if PyTorch can see an NVIDIA GPU, otherwise the CPU. The sidebar and results
  show the device that was actually used; the app never claims GPU use without CUDA being available.
- On Windows, `pip install -r api/requirements.txt` installs a **CPU-only** PyTorch. For GPU inference install a
  CUDA build first (pick the command for your CUDA version at <https://pytorch.org/get-started/locally/>), e.g.:

  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  pip install -r api/requirements.txt
  ```

  If torch was already installed CPU-only: `pip uninstall -y torch torchvision` first.
- Verify: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.
- On GPU the app uses FP16 (half precision) automatically.
- CPU speed tips: use a nano model, lower the inference image size to 480 or 320, or use *Max frames* for tests.
  Typical CPU speed with `yolov8n` at 640 px is roughly 8–25 FPS depending on the processor.

## 8. Running the tests

```bash
cd api
python -m pytest -v
```

The suite checks: model loading and COCO class IDs, opening videos, helpful errors for missing / empty /
corrupted / wrong-type files, bus detection on Ultralytics' sample image, vehicle-only vs all-classes
filtering, bounding-box drawing, end-to-end processing (output re-opens with the same resolution and frame
count), statistics consistency (summary totals = frame-level CSV rows), the max-frames limit, and ByteTrack IDs.
Model tests are skipped automatically if the model cannot be downloaded.

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `Python was not found` (Windows) | Install Python from python.org with "Add to PATH", or disable the Store alias under *Settings → Apps → Advanced app settings → App execution aliases*. |
| `uvicorn: command not found` | Activate the virtual environment, or run `python -m uvicorn main:app` from `api/`. |
| Web app shows "Cannot reach the API" | Start the API (§4). Check `NEXT_PUBLIC_API_URL` in `web/.env.local` matches the API address, and `VD_CORS_ORIGINS` if the web app is not on `localhost:3000`. |
| `npm: command not found` | Install Node.js 20.9+ from <https://nodejs.org/> and open a new terminal. |
| Model download fails / offline | Download the `.pt` file manually into `api/models/` (see §3). Corporate proxies: set `HTTPS_PROXY`. |
| "CUDA was requested but no usable NVIDIA GPU was found" | Install a CUDA-enabled PyTorch (§7) or choose `cpu`/`auto`. |
| `CUDA out of memory` | Use a smaller model or image size, or the CPU. |
| Video won't open / "no frames could be decoded" | The file is corrupted or uses an unusual codec. Re-encode: `ffmpeg -i input.mov -c:v libx264 output.mp4`. |
| Output plays in VLC but not in the browser | ffmpeg conversion failed; a warning is shown. Reinstall `imageio-ffmpeg` (`pip install --force-reinstall imageio-ffmpeg`) or install ffmpeg on PATH. |
| Upload is slow or fails for huge files | Choose *Video file on the API machine* and paste the path; no upload needed. |
| "Not enough free disk space" | Free space or process fewer frames. Old outputs can be deleted from `api/output_videos/` and `api/input_videos/`. |
| Tracking toggle disabled | `pip install lap`. |
| Webcam not found | Try camera index 1 or 2; close other apps using the camera; on Linux check permissions for `/dev/video0`. |
| `ImportError: libGL.so.1` (Linux) | `sudo apt install libgl1 libglib2.0-0`, or `pip install opencv-python-headless` instead of `opencv-python`. |
| Processing very slow | You're on CPU: use `yolov8n.pt`/`yolo11n.pt` with image size 480 or 320. |
| "Waiting for the previous job to finish…" | Jobs run one at a time on the API. Wait, or cancel the running job. |

## 10. Compatibility notes

- Uses the current Ultralytics API: `YOLO(weights)`, `model.predict(...)`, `model.track(..., persist=True,
  tracker="bytetrack.yaml")`, `Results.boxes.{xyxy,conf,cls,id}`. Tested-against range: `ultralytics>=8.3,<9`,
  which supports both YOLOv8 and YOLO11 weights.
- Class IDs are looked up by name from `model.names`, so custom models that contain `car`, `bus`, etc. also work.
- OpenCV pip wheels write MPEG-4 Part 2 (`mp4v`) reliably but not H.264, hence the ffmpeg re-encode step.
- The API is designed as a local, single-user tool: it has no authentication, binds to `127.0.0.1` by default,
  and can read any video path on its machine (*Video file on the API machine*). Do not expose it to the
  internet as-is. Jobs run one at a time so cached models never share tracker state between runs; job state is
  kept in memory and lost when the API restarts (the files in `api/output_videos/` and `api/reports/` remain).

## 11. Future improvements

- Line-crossing / region-of-interest counting for directional traffic counts.
- Speed estimation with camera calibration.
- Persistent job store / Celery workers so jobs survive API restarts and can run on several machines.
- Frame skipping and batched inference for faster processing of long videos.
- Keep the original audio track in the output.
- Charts of detections over time; heatmaps of vehicle positions.
- Licence-plate detection/recognition with a dedicated model (subject to privacy law).
- Export to ONNX/TensorRT/OpenVINO for faster CPU/GPU inference.
- Fine-tuning on local traffic data (e.g. rickshaws, vans) for better accuracy.
- Docker image for one-command deployment.
