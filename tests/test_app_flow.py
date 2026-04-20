from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np

import app
from src.premium_alpr import PremiumALPRResult
from src.v2.models import LocalPlateResult


def _local_result(plate_text: str = 'ABC1D23') -> LocalPlateResult:
    image = np.zeros((24, 64, 3), dtype=np.uint8)
    return LocalPlateResult(
        plate_text=plate_text,
        confidence=0.91,
        detection_confidence=0.88,
        format_type='mercosul',
        is_valid=True,
        original_crop=image,
        bbox=(1, 2, 30, 12),
        normalized_crop=image,
        preprocessed_image=image,
        ocr_engine='paddle_ocr',
        char_confidences=[(char, 0.91) for char in plate_text],
    )


def test_store_local_results_updates_session_state():
    session = {}
    results = [_local_result()]

    app._store_local_results(results, 'frame.jpg', session_state=session)

    assert session['v2_local_results'] == results
    assert session['v2_local_image_name'] == 'frame.jpg'


def test_store_premium_result_updates_session_state():
    session = {}
    premium = PremiumALPRResult(success=True, plate_text='ABC1D23', provider='platerecognizer')

    app._store_premium_result(premium, 123.4, session_state=session)

    assert session['v2_premium_result'] is premium
    assert session['v2_premium_time_ms'] == 123.4


def test_store_video_result_updates_session_state():
    session = {}
    video_result = SimpleNamespace(processed_frames=3)

    app._store_video_result(video_result, session_state=session)

    assert session['v2_video_result'] is video_result


def test_decode_uploaded_image_returns_decoded_image_and_original_bytes():
    image = np.full((16, 20, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode('.jpg', image)
    assert ok is True

    uploaded_file = SimpleNamespace(getvalue=lambda: encoded.tobytes())

    decoded, raw_bytes = app._decode_uploaded_image(uploaded_file)

    assert decoded is not None
    assert decoded.shape == image.shape
    assert raw_bytes == encoded.tobytes()


def test_render_image_outputs_calls_local_display_and_comparison(monkeypatch):
    session = {
        'v2_local_results': [_local_result()],
        'v2_premium_result': PremiumALPRResult(
            success=True,
            plate_text='ABC1D23',
            provider='platerecognizer',
        ),
        'v2_premium_time_ms': 87.5,
    }
    local_display = MagicMock()
    comparison_display = MagicMock()
    monkeypatch.setattr(app, '_display_local_results', local_display)
    monkeypatch.setattr(app, 'display_premium_api_comparison', comparison_display)

    app._render_image_outputs(session_state=session)

    local_display.assert_called_once_with(session['v2_local_results'])
    comparison_display.assert_called_once()
    args = comparison_display.call_args.args
    assert len(args[0]) == 1
    assert args[0][0] is session['v2_local_results'][0]
    assert args[1].plate_text == 'ABC1D23'
    assert comparison_display.call_args.kwargs['premium_time_ms'] == 87.5


def test_display_local_results_uses_summary_and_cards(monkeypatch):
    results = [_local_result('ABC1D23'), _local_result('XYZ9K88')]
    summary_display = MagicMock()
    card_display = MagicMock()
    monkeypatch.setattr(app, 'display_summary_table', summary_display)
    monkeypatch.setattr(app, 'display_local_result', card_display)
    monkeypatch.setattr(app.st, 'columns', lambda count: [f'col-{index}' for index in range(count)])

    app._display_local_results(results)

    summary_display.assert_called_once_with(results)
    assert card_display.call_count == 2
    assert card_display.call_args_list[0].args == (results[0], 'col-0')
    assert card_display.call_args_list[1].args == (results[1], 'col-1')


def test_build_video_processor_maps_v2_config_fields():
    config = SimpleNamespace(
        video=SimpleNamespace(
            skip_frames=4,
            max_frames=99,
            generate_output_video=True,
            output_dir='data/results',
            confidence_threshold=0.66,
            vehicle_mode='moving',
            enable_temporal_voting=True,
            temporal_strategy='majority',
            temporal_min_observations=3,
        )
    )

    processor = app._build_video_processor(config)

    assert processor.skip_frames == 4
    assert processor.max_frames == 99
    assert processor.generate_output_video is True
    assert processor.output_dir == Path('data/results')
    assert processor.confidence_threshold == 0.66
    assert processor.vehicle_mode == 'moving'
    assert processor.enable_temporal_voting is True
    assert processor.output_dir.exists()


def test_render_video_outputs_builds_processor_and_delegates(monkeypatch):
    video_result = SimpleNamespace(
        processed_frames=12,
        skipped_frames=3,
        unique_plates={'ABC1D23': {'plate_text': 'ABC1D23'}},
        processing_fps=7.5,
        output_video_path='video.mp4',
    )
    session = {'v2_video_result': video_result}
    config = SimpleNamespace(video=SimpleNamespace())
    build_processor = MagicMock(return_value=SimpleNamespace(name='processor'))
    render_video = MagicMock()
    monkeypatch.setattr(app, '_build_video_processor', build_processor)
    monkeypatch.setattr(app, 'display_video_results', render_video)

    app._render_video_outputs(config, session_state=session)

    build_processor.assert_called_once_with(config)
    render_video.assert_called_once_with(video_result, build_processor.return_value)


def test_render_video_outputs_skips_when_session_has_no_video_result(monkeypatch):
    build_processor = MagicMock()
    render_video = MagicMock()
    monkeypatch.setattr(app, '_build_video_processor', build_processor)
    monkeypatch.setattr(app, 'display_video_results', render_video)

    app._render_video_outputs(SimpleNamespace(video=SimpleNamespace()), session_state={})

    build_processor.assert_not_called()
    render_video.assert_not_called()