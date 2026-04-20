from __future__ import annotations

from typing import List

import cv2
import numpy as np

from src.v2.models import LocalPlateResult
from src.video_processor import VehicleMode, VideoProcessor


def _make_frame(value: int) -> np.ndarray:
    return np.full((48, 96, 3), value, dtype=np.uint8)


class _FakeCapture:
    def __init__(self, frames: List[np.ndarray], fps: float = 30.0):
        self.frames = [frame.copy() for frame in frames]
        self.fps = fps
        self.index = 0
        self.opened = True

    def isOpened(self):
        return self.opened

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index].copy()
        self.index += 1
        return True, frame

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self.frames))
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self.fps)
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.frames[0].shape[1])
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.frames[0].shape[0])
        if prop_id == cv2.CAP_PROP_FOURCC:
            return float(cv2.VideoWriter_fourcc(*'mp4v'))
        if prop_id == cv2.CAP_PROP_POS_MSEC:
            return ((self.index - 1) / self.fps) * 1000.0 if self.index else 0.0
        return 0.0

    def release(self):
        self.opened = False


class _FakePipeline:
    def __init__(self, plate_text: str = 'ABC1D23', confidence: float = 0.93):
        self.calls = 0
        self.plate_text = plate_text
        self.confidence = confidence

    def process_image(self, frame, detector_confidence=None):
        del detector_confidence
        self.calls += 1
        return [
            LocalPlateResult(
                plate_text=self.plate_text,
                confidence=self.confidence,
                detection_confidence=0.88,
                format_type='mercosul',
                is_valid=True,
                original_crop=frame[8:28, 12:60].copy(),
                bbox=(12, 8, 60, 28),
                normalized_crop=frame[8:28, 12:60].copy(),
                preprocessed_image=frame[8:28, 12:60].copy(),
                ocr_engine='paddle_ocr',
                char_confidences=[(char, self.confidence) for char in self.plate_text],
            )
        ]


def test_process_video_with_v2_pipeline_collects_results(monkeypatch, tmp_path):
    frames = [_make_frame(60), _make_frame(90), _make_frame(120)]
    monkeypatch.setattr(cv2, 'VideoCapture', lambda _: _FakeCapture(frames, fps=24.0))
    processor = VideoProcessor(
        skip_frames=1,
        generate_output_video=False,
        output_dir=str(tmp_path),
        vehicle_mode=VehicleMode.MOVING,
        enable_temporal_voting=False,
    )
    pipeline = _FakePipeline()
    progress_events = []

    result = processor.process_video(
        'sample.mp4',
        pipeline=pipeline,
        detector_confidence=0.4,
        progress_callback=lambda current, total, frame_result: progress_events.append(
            (current, total, None if frame_result is None else frame_result.plates_found)
        ),
    )

    assert pipeline.calls == 3
    assert result.total_frames == 3
    assert result.processed_frames == 3
    assert result.total_detections == 3
    assert len(result.frame_results) == 3
    assert 'ABC1D23' in result.unique_plates
    assert result.unique_plates['ABC1D23']['total_detections'] == 3
    assert progress_events[-1][:2] == (3, 3)


def test_process_video_stationary_mode_triggers_early_stop(monkeypatch, tmp_path):
    frames = [_make_frame(50), _make_frame(55), _make_frame(60), _make_frame(65), _make_frame(70)]
    monkeypatch.setattr(cv2, 'VideoCapture', lambda _: _FakeCapture(frames, fps=30.0))
    processor = VideoProcessor(
        skip_frames=1,
        generate_output_video=False,
        output_dir=str(tmp_path),
        vehicle_mode=VehicleMode.STATIONARY,
        enable_temporal_voting=False,
    )
    processor.sharpness_filter = False
    pipeline = _FakePipeline(confidence=0.97)

    result = processor.process_video('stationary.mp4', pipeline=pipeline, detector_confidence=0.4)

    assert result.total_frames == 5
    assert result.processed_frames == processor.min_stable_detections
    assert pipeline.calls == processor.min_stable_detections
    assert 'ABC1D23' in result.unique_plates
