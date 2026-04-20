from src.video_processor import VehicleMode
from src.v2.config import build_v2_config


def test_build_v2_config_forces_paddle_local_engine():
    raw = {
        'models': {'detector': {'default': 'custom.pt', 'dir': 'models/yolo'}},
        'ocr': {'engine': 'glm', 'paddle': {'lang': 'pt'}},
        'pipeline': {'crop_margin': 0.12, 'ocr_confidence_threshold': 0.55},
    }

    config = build_v2_config(raw)

    assert config.detector.model_name == 'custom.pt'
    assert config.detector.crop_margin == 0.12
    assert config.ocr.engine == 'paddle'
    assert config.ocr.confidence_threshold == 0.55


def test_build_v2_config_reads_premium_and_video_fields(monkeypatch):
    monkeypatch.setenv('PLATE_RECOGNIZER_API_KEY', 'x' * 20)
    raw = {
        'premium_api': {'enabled': True, 'provider': 'platerecognizer'},
        'video': {'skip_frames': None, 'vehicle_mode': 'stationary'},
        'temporal_voting': {'enabled': False},
    }

    config = build_v2_config(raw)

    assert config.premium.enabled is True
    assert config.premium.api_key == 'x' * 20
    assert config.video.skip_frames == 5
    assert config.video.vehicle_mode == VehicleMode.STATIONARY
    assert config.video.enable_temporal_voting is False


def test_build_v2_config_reads_artifacts_and_scenarios():
    raw = {
        'artifacts': {'enabled': False, 'confidence_threshold': 0.7},
        'scenarios': {
            'low_light': {'brightness_threshold': 70.0},
            'small_plate': {'height_threshold': 48},
        },
        'evaluation': {'reports_dir': 'data/custom-eval'},
        'calibration': {'detector_thresholds': [0.15, 0.25]},
    }

    config = build_v2_config(raw)

    assert config.artifacts.enabled is False
    assert config.artifacts.confidence_threshold == 0.7
    assert config.scenarios.low_light.brightness_threshold == 70.0
    assert config.scenarios.small_plate.height_threshold == 48
    assert config.evaluation.reports_dir == 'data/custom-eval'
    assert config.calibration.detector_thresholds == [0.15, 0.25]


def test_build_v2_config_reads_quality_forensic_and_report_fields():
    raw = {
        'ocr': {'top_k_candidates': 7},
        'quality': {'enabled': False, 'snr_review_threshold': 12.0},
        'forensic': {'jpeg_quality': 85, 'review_threshold': 0.61},
        'reports': {'output_dir': 'data/custom-reports', 'prefer_artifact_dir': False},
        'models': {'detector': {'sahi_retry_confidence_threshold': 0.42}},
    }

    config = build_v2_config(raw)

    assert config.ocr.top_k_candidates == 7
    assert config.quality.enabled is False
    assert config.quality.snr_review_threshold == 12.0
    assert config.forensic.jpeg_quality == 85
    assert config.forensic.review_threshold == 0.61
    assert config.reports.output_dir == 'data/custom-reports'
    assert config.reports.prefer_artifact_dir is False
    assert config.detector.sahi_retry_confidence_threshold == 0.42


def test_build_v2_config_reads_llm_validation_fields():
    raw = {
        'llm_validation': {
            'enabled': True,
            'base_url': 'http://ollama.local:11434',
            'model': 'qwen3.5:9b-q8_0',
            'timeout': 12.5,
            'allow_override': False,
            'ambiguity_gap_threshold': 0.08,
            'min_decision_confidence': 0.82,
        }
    }

    config = build_v2_config(raw)

    assert config.llm_validation.enabled is True
    assert config.llm_validation.base_url == 'http://ollama.local:11434'
    assert config.llm_validation.model == 'qwen3.5:9b-q8_0'
    assert config.llm_validation.timeout == 12.5
    assert config.llm_validation.allow_override is False
    assert config.llm_validation.ambiguity_gap_threshold == 0.08
    assert config.llm_validation.min_decision_confidence == 0.82