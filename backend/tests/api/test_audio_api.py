"""API endpoint tests for audio processing routes.

All database and Celery interactions are mocked — tests verify routing,
response shapes, and HTTP status codes only.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app

SESSION_ID = str(uuid.uuid4())


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /audio/process
# ---------------------------------------------------------------------------


def test_start_audio_processing_returns_202(client):
    """POST /audio/process should return 202 with task_id."""
    with patch("app.api.v1.audio.AudioService") as MockService:
        instance = MockService.return_value
        instance.start_analysis = AsyncMock(
            return_value={
                "session_id": SESSION_ID,
                "task_id": "celery-task-abc123",
                "status": "queued",
            }
        )
        response = client.post(
            "/api/v1/audio/process",
            json={"session_id": SESSION_ID},
        )

    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data
    assert data["task_id"] == "celery-task-abc123"
    assert data["status"] == "queued"


def test_start_audio_processing_passes_config_overrides(client):
    """POST /audio/process should forward config_overrides to the service."""
    with patch("app.api.v1.audio.AudioService") as MockService:
        instance = MockService.return_value
        instance.start_analysis = AsyncMock(
            return_value={
                "session_id": SESSION_ID,
                "task_id": "task-xyz",
                "status": "queued",
            }
        )
        response = client.post(
            "/api/v1/audio/process",
            json={
                "session_id": SESSION_ID,
                "config_overrides": {"whisper_model_size": "small"},
            },
        )

    assert response.status_code == 202
    call_kwargs = instance.start_analysis.call_args
    assert call_kwargs.kwargs["config"] == {"whisper_model_size": "small"}


# ---------------------------------------------------------------------------
# GET /audio/status/{session_id}
# ---------------------------------------------------------------------------


def test_get_status_returns_status(client):
    """GET /audio/status/{session_id} should return status."""
    with patch("app.api.v1.audio.AudioService") as MockService:
        instance = MockService.return_value
        instance.get_analysis_status = AsyncMock(
            return_value={
                "session_id": SESSION_ID,
                "task_id": "celery-task-abc123",
                "status": "processing",
                "progress": 0.45,
                "error": None,
            }
        )
        response = client.get(f"/api/v1/audio/status/{SESSION_ID}")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "processing"
    assert data["progress"] == 0.45


def test_get_status_returns_404_for_unknown_session(client):
    """GET /audio/status/{session_id} should propagate 404 from service."""
    with patch("app.api.v1.audio.AudioService") as MockService:
        instance = MockService.return_value
        instance.get_analysis_status = AsyncMock(
            side_effect=HTTPException(status_code=404, detail="Not found")
        )
        response = client.get(f"/api/v1/audio/status/{SESSION_ID}")

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /audio/results/{session_id}
# ---------------------------------------------------------------------------


def test_get_results_returns_409_if_not_complete(client):
    """GET /audio/results/{id} should return 409 if not complete."""
    with patch("app.api.v1.audio.AudioService") as MockService:
        instance = MockService.return_value
        instance.get_analysis_results = AsyncMock(
            side_effect=HTTPException(
                status_code=409, detail="Analysis not complete"
            )
        )
        response = client.get(f"/api/v1/audio/results/{SESSION_ID}")

    assert response.status_code == 409


def test_get_results_returns_audio_result_schema(client):
    """GET /audio/results/{id} should return a valid AudioResultSchema on success."""
    with patch("app.api.v1.audio.AudioService") as MockService:
        instance = MockService.return_value
        instance.get_analysis_results = AsyncMock(
            return_value={
                "video_path": "/tmp/test.mp4",
                "duration_seconds": 120.0,
                "sample_rate": 16000,
                "segments": [],
                "turns": [],
                "transcripts": [],
                "interruptions": [],
                "processing_time_seconds": 5.2,
                "session_metrics": None,
            }
        )
        response = client.get(f"/api/v1/audio/results/{SESSION_ID}")

    assert response.status_code == 200
    data = response.json()
    assert data["duration_seconds"] == 120.0
    assert data["sample_rate"] == 16000


# ---------------------------------------------------------------------------
# GET /audio/results/{session_id}/transcripts
# ---------------------------------------------------------------------------


def test_get_transcripts_returns_list(client):
    """GET /audio/results/{id}/transcripts should return list."""
    with patch("app.api.v1.audio.AudioService") as MockService:
        instance = MockService.return_value
        instance.get_transcripts = AsyncMock(
            return_value=[
                {
                    "start": 0.0,
                    "end": 2.0,
                    "speaker_id": "speaker_0",
                    "text": "Hola",
                    "words": [],
                    "language": "es",
                    "no_speech_prob": 0.0,
                }
            ]
        )
        response = client.get(f"/api/v1/audio/results/{SESSION_ID}/transcripts")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["speaker_id"] == "speaker_0"
    assert data[0]["text"] == "Hola"


def test_get_transcripts_returns_empty_list(client):
    """GET /audio/results/{id}/transcripts should return empty list when none exist."""
    with patch("app.api.v1.audio.AudioService") as MockService:
        instance = MockService.return_value
        instance.get_transcripts = AsyncMock(return_value=[])
        response = client.get(f"/api/v1/audio/results/{SESSION_ID}/transcripts")

    assert response.status_code == 200
    assert response.json() == []
