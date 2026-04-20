from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.v2.ui import display as display_module


class _MetricColumn:
    def __init__(self):
        self.values = []

    def metric(self, label, value):
        self.values.append((label, value))


class _ContextManager:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_display_video_results_renders_ranked_table_timeline_and_video(monkeypatch, tmp_path):
    video_file = tmp_path / 'result.mp4'
    video_file.write_bytes(b'video-bytes')

    subheader = MagicMock()
    dataframe = MagicMock()
    info = MagicMock()
    video = MagicMock()
    expander = MagicMock(return_value=_ContextManager())
    monkeypatch.setattr(display_module.st, 'subheader', subheader)
    monkeypatch.setattr(display_module.st, 'columns', lambda count: [_MetricColumn() for _ in range(count)])
    monkeypatch.setattr(display_module.st, 'dataframe', dataframe)
    monkeypatch.setattr(display_module.st, 'info', info)
    monkeypatch.setattr(display_module.st, 'expander', expander)
    monkeypatch.setattr(display_module.st, 'video', video)

    video_result = SimpleNamespace(
        processed_frames=12,
        skipped_frames=3,
        unique_plates={'ABC1D23': {'plate_text': 'ABC1D23'}},
        processing_fps=7.5,
        output_video_path=str(video_file),
    )
    video_processor = SimpleNamespace(
        rank_unique_plates=lambda unique_plates: {
            'ABC1D23': {
                'plate_text': 'ABC1D23',
                'best_confidence': 0.94,
                'total_detections': 4,
                'first_seen_frame': 2,
                'last_seen_frame': 9,
            }
        },
        build_confirmed_reading=lambda info_row: info_row['plate_text'],
        generate_timeline=lambda result: [{'frame': 2, 'plate': 'ABC1D23'}],
    )

    display_module.display_video_results(video_result, video_processor)

    subheader.assert_called_once_with('Resultado do video')
    assert dataframe.call_count == 2
    info.assert_not_called()
    expander.assert_called_once_with('Timeline de deteccoes')
    video.assert_called_once_with(b'video-bytes')


def test_display_video_results_shows_empty_state(monkeypatch):
    info = MagicMock()
    monkeypatch.setattr(display_module.st, 'subheader', MagicMock())
    monkeypatch.setattr(display_module.st, 'columns', lambda count: [_MetricColumn() for _ in range(count)])
    monkeypatch.setattr(display_module.st, 'dataframe', MagicMock())
    monkeypatch.setattr(display_module.st, 'expander', MagicMock(return_value=_ContextManager()))
    monkeypatch.setattr(display_module.st, 'video', MagicMock())
    monkeypatch.setattr(display_module.st, 'info', info)

    video_result = SimpleNamespace(
        processed_frames=0,
        skipped_frames=0,
        unique_plates={},
        processing_fps=0.0,
        output_video_path='',
    )
    video_processor = SimpleNamespace(
        rank_unique_plates=lambda unique_plates: {},
        build_confirmed_reading=lambda info_row: info_row['plate_text'],
        generate_timeline=lambda result: [],
    )

    display_module.display_video_results(video_result, video_processor)

    info.assert_called_once_with('Nenhuma placa consolidada no video.')