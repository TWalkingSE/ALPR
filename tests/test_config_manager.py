# tests/test_config_manager.py
"""
Testes unitários para src/config_manager.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.config_manager import load_config, get_default_config, _deep_merge


class TestGetDefaultConfig:
    def test_returns_dict(self):
        cfg = get_default_config()
        assert isinstance(cfg, dict)

    def test_has_required_sections(self):
        cfg = get_default_config()
        for section in (
            'models', 'ocr', 'premium_api', 'pipeline', 'temporal_voting', 'video',
            'artifacts', 'scenarios', 'evaluation', 'calibration', 'quality', 'forensic', 'reports',
            'llm_validation',
        ):
            assert section in cfg, f"Default config missing '{section}'"

    def test_detector_defaults(self):
        cfg = get_default_config()
        det = cfg['models']['detector']
        # Sincronizado com config.yaml (default 0.25 para câmeras de rodovia)
        assert det['confidence'] == 0.25
        assert det['device'] == 'auto'
        assert det['default'] == 'yolo11l-plate.pt'
        assert det['enable_sahi'] is True

    def test_ocr_defaults(self):
        """A v2 usa PaddleOCR como engine local padrão."""
        cfg = get_default_config()
        ocr = cfg['ocr']
        assert ocr['engine'] == 'paddle'
        assert ocr['try_multiple_variants'] is True
        assert ocr['paddle']['lang'] == 'pt'
        assert ocr['paddle']['use_angle_cls'] is True

    def test_premium_api_defaults(self):
        """Tier Premium (Plate Recognizer) — desabilitado por padrão."""
        cfg = get_default_config()
        pa = cfg['premium_api']
        assert pa['enabled'] is False
        assert pa['provider'] == 'platerecognizer'
        assert pa['regions'] == ['br']

    def test_pipeline_has_v2_keys(self):
        """Pipeline deve expor apenas thresholds usados pela v2."""
        cfg = get_default_config()
        p = cfg['pipeline']
        assert 'crop_margin' in p
        assert 'ocr_confidence_threshold' in p
        assert 'fallback_confidence_threshold' in p
        assert 'llm_confidence_threshold' not in p

    def test_artifact_defaults(self):
        cfg = get_default_config()
        artifacts = cfg['artifacts']
        assert artifacts['enabled'] is True
        assert artifacts['save_invalid'] is True
        assert artifacts['confidence_threshold'] == 0.75

    def test_quality_and_report_defaults(self):
        cfg = get_default_config()
        assert cfg['quality']['enabled'] is True
        assert cfg['quality']['snr_review_threshold'] == 9.0
        assert cfg['forensic']['jpeg_quality'] == 90
        assert cfg['reports']['output_dir'] == 'data/results/reports'
        assert cfg['llm_validation']['enabled'] is False
        assert cfg['llm_validation']['base_url'] == 'http://127.0.0.1:11434'


class TestDeepMerge:
    def test_simple_merge(self):
        base = {'a': 1, 'b': 2}
        override = {'b': 3, 'c': 4}
        result = _deep_merge(base, override)
        assert result == {'a': 1, 'b': 3, 'c': 4}

    def test_nested_merge(self):
        base = {'x': {'a': 1, 'b': 2}}
        override = {'x': {'b': 3, 'c': 4}}
        result = _deep_merge(base, override)
        assert result == {'x': {'a': 1, 'b': 3, 'c': 4}}

    def test_override_non_dict(self):
        base = {'x': {'a': 1}}
        override = {'x': 'string_value'}
        result = _deep_merge(base, override)
        assert result == {'x': 'string_value'}

    def test_empty_override(self):
        base = {'a': 1, 'b': 2}
        result = _deep_merge(base, {})
        assert result == {'a': 1, 'b': 2}


class TestLoadConfig:
    def test_loads_from_yaml(self, config_path):
        """Testa que config.yaml é carregado e merged com defaults."""
        cfg = load_config(config_path)
        assert isinstance(cfg, dict)
        assert 'models' in cfg
        assert 'pipeline' in cfg

    def test_fallback_to_defaults(self, tmp_path):
        """Se o arquivo não existe, retorna defaults."""
        cfg = load_config("nonexistent_config_file_12345.yaml")
        defaults = get_default_config()
        assert cfg == defaults
