"""Tests for the REST API used by the Next.js frontend.

Run from the api/ folder:  python -m pytest -v
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture()
def client():
    with TestClient(main.app) as c:
        yield c


def wait_for(client: TestClient, job_id: str, timeout: float = 300) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.2)
    raise TimeoutError(f"Job {job_id} did not finish")


def test_health_and_config(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"status": "ok"}
    config = client.get("/api/config").json()
    assert config["default_model"] in config["models"]
    assert {c["name"] for c in config["vehicle_classes"]} == {"car", "motorcycle", "bus", "truck"}
    assert "auto" in config["devices"] and "cpu" in config["devices"]


def test_upload_rejects_wrong_extension(client: TestClient) -> None:
    res = client.post("/api/videos", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert res.status_code == 422
    assert "Unsupported file type" in res.json()["detail"]


def test_upload_rejects_corrupted_video(client: TestClient) -> None:
    res = client.post("/api/videos", files={"file": ("broken.mp4", b"not a real video" * 100, "video/mp4")})
    assert res.status_code == 422


def test_local_path_missing(client: TestClient, tmp_path: Path) -> None:
    res = client.post("/api/videos/local", json={"path": str(tmp_path / "missing.mp4")})
    assert res.status_code == 422
    assert "not found" in res.json()["detail"]


def test_job_rejects_empty_class_selection(client: TestClient, sample_video: Path) -> None:
    video = client.post("/api/videos/local", json={"path": str(sample_video)}).json()
    res = client.post("/api/jobs/video", json={"video_id": video["video_id"], "settings": {"classes": []}})
    assert res.status_code == 422


def test_unknown_job_and_video(client: TestClient) -> None:
    assert client.get("/api/jobs/doesnotexist").status_code == 404
    res = client.post("/api/jobs/video", json={"video_id": "nope.mp4"})
    assert res.status_code == 404


def test_end_to_end_video_job(client: TestClient, sample_video: Path, detector, isolated_dirs) -> None:
    with open(sample_video, "rb") as fh:
        res = client.post("/api/videos", files={"file": ("bus_scene.mp4", fh, "video/mp4")})
    assert res.status_code == 201, res.text
    video = res.json()
    uploaded = main.INPUT_DIR / video["video_id"]
    assert video["info"]["width"] == 854 and video["info"]["frame_count"] == 30

    settings = {"model_name": "yolov8n.pt", "device": "cpu", "confidence": 0.25, "max_frames": 20}
    res = client.post("/api/jobs/video", json={"video_id": video["video_id"], "settings": settings})
    assert res.status_code == 202, res.text
    job = wait_for(client, res.json()["id"])
    assert job["status"] == "completed", job["error"]

    result = job["result"]
    output = None
    try:
        assert result["frames_processed"] == 20
        assert result["detection_counts"].get("bus", 0) > 0
        assert result["total_vehicle_detections"] == sum(result["detection_counts"].values())
        assert result["summary_table"][-1]["Class"] == "TOTAL (vehicles)"

        video_res = client.get(f"/api/jobs/{job['id']}/video")
        assert video_res.status_code == 200 and len(video_res.content) > 0
        output = Path(main.manager.get(job["id"]).result.output_path)

        records = client.get(f"/api/jobs/{job['id']}/records?limit=1000").json()
        assert len(records) == result["total_detections"]
        summary_csv = client.get(f"/api/jobs/{job['id']}/summary.csv")
        assert summary_csv.headers["content-type"].startswith("text/csv")
        assert client.get(f"/api/jobs/{job['id']}/tracks.csv").status_code == 404  # tracking was off
    finally:
        uploaded.unlink(missing_ok=True)
        if output is not None:
            output.unlink(missing_ok=True)
