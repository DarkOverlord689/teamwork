"""Tests for the FrameExtractor module (T5.1)."""

import os
import tempfile

import cv2
import numpy as np
import pytest

from backend.app.core.vision.config import VisionConfig
from backend.app.core.vision.frame_extractor import FrameExtractor


@pytest.fixture
def config():
    return VisionConfig(fps=2.0, max_frames=50)


@pytest.fixture
def extractor(config):
    ext = FrameExtractor(config=config, device="cpu")
    ext.load()
    yield ext
    ext.unload()


@pytest.fixture
def synthetic_video(tmp_path):
    """Create a small synthetic video (30 frames at 10 fps = 3 seconds)."""
    video_path = str(tmp_path / "test_video.avi")
    width, height, fps, num_frames = 64, 48, 10.0, 30

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for i in range(num_frames):
        # Each frame is a different shade of grey so they are distinguishable
        frame = np.full((height, width, 3), fill_value=i * 8, dtype=np.uint8)
        writer.write(frame)

    writer.release()
    return video_path, num_frames, fps


class TestFrameExtractorLifecycle:
    def test_load_unload(self):
        ext = FrameExtractor(device="cpu")
        assert not ext.is_loaded
        ext.load()
        assert ext.is_loaded
        ext.unload()
        assert not ext.is_loaded

    def test_default_config(self):
        ext = FrameExtractor(device="cpu")
        assert ext.config.fps == 3.0
        assert ext.config.max_frames == 500


class TestGetVideoProperties:
    def test_valid_video(self, extractor, synthetic_video):
        path, expected_frames, expected_fps = synthetic_video
        total, duration, native_fps = extractor.get_video_properties(path)
        assert total == expected_frames
        assert native_fps == pytest.approx(expected_fps, abs=0.5)
        assert duration == pytest.approx(expected_frames / expected_fps, abs=0.5)

    def test_file_not_found(self, extractor):
        with pytest.raises(FileNotFoundError):
            extractor.get_video_properties("/nonexistent/video.mp4")

    def test_invalid_file(self, extractor, tmp_path):
        bad_file = tmp_path / "bad.avi"
        bad_file.write_text("not a video")
        with pytest.raises(RuntimeError):
            extractor.get_video_properties(str(bad_file))


class TestProcessFrames:
    def test_extracts_frames(self, extractor, synthetic_video):
        path, total, fps = synthetic_video
        frames = extractor.process(path)
        assert len(frames) > 0
        # Each element is (frame_number, timestamp, ndarray)
        for frame_num, ts, arr in frames:
            assert isinstance(frame_num, int)
            assert isinstance(ts, float)
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (48, 64, 3)

    def test_respects_fps(self, synthetic_video):
        """Extracting at 1 fps from a 3-second video should yield ~3 frames."""
        path, _, _ = synthetic_video
        config = VisionConfig(fps=1.0, max_frames=500)
        ext = FrameExtractor(config=config, device="cpu")
        ext.load()
        frames = ext.process(path)
        ext.unload()
        # 3-second video at 1 fps: expect roughly 3 frames
        assert 2 <= len(frames) <= 4

    def test_max_frames_cap(self, synthetic_video):
        path, _, _ = synthetic_video
        config = VisionConfig(fps=10.0, max_frames=5)
        ext = FrameExtractor(config=config, device="cpu")
        ext.load()
        frames = ext.process(path)
        ext.unload()
        assert len(frames) <= 5

    def test_override_fps_in_process(self, extractor, synthetic_video):
        path, _, _ = synthetic_video
        frames = extractor.process(path, fps=1.0, max_frames=100)
        assert 2 <= len(frames) <= 4

    def test_file_not_found(self, extractor):
        with pytest.raises(FileNotFoundError):
            extractor.process("/nonexistent/video.mp4")

    def test_timestamps_increase(self, extractor, synthetic_video):
        path, _, _ = synthetic_video
        frames = extractor.process(path)
        timestamps = [ts for _, ts, _ in frames]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]
