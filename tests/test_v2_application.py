from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from src.v2.application import (
    build_service_bundle,
    decode_uploaded_image,
    ensure_service_bundle,
    run_local_image_analysis,
    run_premium_image_analysis,
    run_video_analysis,
)
from src.v2.contracts import ServiceBundle


def test_decode_uploaded_image_decodes_bgr_and_preserves_bytes():
    image = np.full((12, 18, 3), 160, dtype=np.uint8)
    ok, encoded = cv2.imencode('.jpg', image)
    assert ok is True

    decoded, raw_bytes = decode_uploaded_image(SimpleNamespace(getvalue=lambda: encoded.tobytes()))

    assert decoded is not None
    assert decoded.shape == image.shape
    assert raw_bytes == encoded.tobytes()


def test_build_service_bundle_attaches_premium_provider(monkeypatch):
    pipeline = SimpleNamespace(premium_provider=None)
    premium = SimpleNamespace(provider='platerecognizer')
    monkeypatch.setattr(
        'src.v2.application.LocalAnalysisPipeline.from_settings',
        lambda config, project_dir, model_path: pipeline,
    )
    monkeypatch.setattr(
        'src.v2.application.PremiumAnalysisService.from_settings',
        lambda premium_config: premium,
    )

    config = SimpleNamespace(premium=SimpleNamespace())
    bundle = build_service_bundle(config, Path('project'), 'model.pt')

    assert isinstance(bundle, ServiceBundle)
    assert bundle.pipeline is pipeline
    assert bundle.premium is premium
    assert pipeline.premium_provider is premium


def test_ensure_service_bundle_reuses_existing_services():
    pipeline = SimpleNamespace(name='pipeline')
    premium = SimpleNamespace(name='premium')
    session = {
        'v2_pipeline': pipeline,
        'v2_premium': premium,
        'v2_signature': ('model.pt', ('sig',)),
    }
    config = SimpleNamespace(signature=lambda: ('sig',))

    bundle = ensure_service_bundle(session, config, Path('project'), 'model.pt')

    assert bundle.pipeline is pipeline
    assert bundle.premium is premium


def test_ensure_service_bundle_rebuilds_when_signature_changes():
    built = ServiceBundle(
        pipeline=SimpleNamespace(name='new-pipeline'),
        premium=SimpleNamespace(name='new-premium'),
    )
    session = {}
    config = SimpleNamespace(signature=lambda: ('sig',))

    bundle = ensure_service_bundle(
        session,
        config,
        Path('project'),
        'model.pt',
        builder=lambda cfg, project_dir, model_path: built,
    )

    assert bundle is built
    assert session['v2_pipeline'] is built.pipeline
    assert session['v2_premium'] is built.premium
    assert session['v2_signature'] == ('model.pt', ('sig',))


def test_run_local_image_analysis_passes_expected_arguments():
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    calls = []
    pipeline = SimpleNamespace(
        process_image=lambda *args, **kwargs: calls.append((args, kwargs)) or ['ok']
    )
    config = SimpleNamespace(detector=SimpleNamespace(confidence=0.44))

    result = run_local_image_analysis(pipeline, config, image, b'image-bytes', 'plate.jpg')

    assert result == ['ok']
    assert calls[0][0] == (image,)
    assert calls[0][1]['detector_confidence'] == 0.44
    assert calls[0][1]['image_bytes'] == b'image-bytes'
    assert calls[0][1]['input_file_path'] == 'plate.jpg'


def test_run_premium_image_analysis_returns_result_and_elapsed_time():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    premium_result = SimpleNamespace(success=True)
    premium_service = SimpleNamespace(analyze_full_image=lambda payload: premium_result)
    perf_values = iter([10.0, 10.125])

    result, elapsed_ms = run_premium_image_analysis(
        premium_service,
        image,
        perf_counter=lambda: next(perf_values),
    )

    assert result is premium_result
    assert elapsed_ms == 125.0


def test_run_video_analysis_uses_temp_file_and_cleans_it_up(tmp_path):
    seen = {}

    def _processor_factory(config):
        return SimpleNamespace(
            process_video=lambda path, pipeline, detector_confidence, progress_callback: seen.update(
                {
                    'path': path,
                    'path_exists_during_call': Path(path).exists(),
                    'file_bytes': Path(path).read_bytes(),
                    'pipeline': pipeline,
                    'detector_confidence': detector_confidence,
                    'progress_callback': progress_callback,
                }
            )
            or SimpleNamespace(processed_frames=2)
        )

    config = SimpleNamespace(detector=SimpleNamespace(confidence=0.77))
    pipeline = SimpleNamespace(name='pipeline')
    callback = lambda current, total, frame_result: None

    result = run_video_analysis(
        b'video-bytes',
        'sample.mp4',
        pipeline,
        config,
        progress_callback=callback,
        processor_factory=_processor_factory,
    )

    assert result.processed_frames == 2
    assert seen['path_exists_during_call'] is True
    assert seen['file_bytes'] == b'video-bytes'
    assert seen['pipeline'] is pipeline
    assert seen['detector_confidence'] == 0.77
    assert seen['progress_callback'] is callback
    assert Path(seen['path']).exists() is False