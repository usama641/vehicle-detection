# Vehicle Detection - Web (Next.js)

Frontend for the AI Vehicle Detection System. It talks to the FastAPI backend in `../api`.
See the main [README](../README.md) for the full setup.

```bash
npm install
npm run dev        # http://localhost:3000  (start the API first: cd ../api && uvicorn main:app --port 8000)
npm run build && npm start   # production
```

Configuration: copy `.env.example` to `.env.local` and set `NEXT_PUBLIC_API_URL` if the API is not at
`http://127.0.0.1:8000`.

| Path | Contents |
|---|---|
| `src/app/page.tsx` | Page shell: settings sidebar + Video file / Live camera / About tabs |
| `src/components/SettingsPanel.tsx` | Model, device, image size, thresholds, classes, tracking, max frames |
| `src/components/VideoTab.tsx` | Upload or local path, video metadata, start detection |
| `src/components/LiveTab.tsx` | Webcam / RTSP stream sessions |
| `src/components/JobSection.tsx`, `JobMonitor.tsx` | Job polling, progress bar, MJPEG live preview, cancel/stop |
| `src/components/Results.tsx` | Annotated video, metrics, summary table, downloads, run details |
| `src/lib/api.ts` | Typed API client |
