# tests/conftest.py
"""
Fixtures compartilhadas para os testes do projeto ALPR.
"""

import sys
from pathlib import Path

# Garantir que o diretório raiz do projeto está no path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest


@pytest.fixture
def project_root():
    """Retorna o caminho raiz do projeto."""
    return PROJECT_ROOT


@pytest.fixture
def config_path(project_root):
    """Retorna o caminho relativo do config.yaml."""
    return "config.yaml"


@pytest.fixture
def sample_config():
    """Configuração mínima para testes unitários (sem I/O).

    Alinhada com o schema enxuto da v2.
    """
    return {
        'models': {
            'detector': {
                'confidence': 0.25,
                'default': 'yolo11l-plate.pt',
                'device': 'cpu',
                'dir': 'models/yolo',
                'enable_sahi': False,
            }
        },
        'ocr': {
            'engine': 'paddle',
            'try_multiple_variants': False,
            'max_variants': 3,
            'paddle': {
                'lang': 'pt',
                'use_gpu': False,
                'use_angle_cls': True,
                'det_limit_side_len': 960,
                'rec_batch_num': 4,
                'min_score': 0.3,
            },
        },
        'premium_api': {
            'enabled': False,
            'provider': 'platerecognizer',
            'api_key': '',
            'regions': ['br'],
            'min_confidence': 0.5,
            'timeout': 30,
            'log_all_calls': False,
        },
        'pipeline': {
            'crop_margin': 0.05,
            'ocr_confidence_threshold': 0.6,
            'fallback_confidence_threshold': 0.8,
        },
        'temporal_voting': {
            'enabled': True,
            'strategy': 'hybrid',
            'min_observations': 2,
        },
        'video': {
            'vehicle_mode': 'moving',
            'skip_frames': 3,
            'max_frames': 0,
            'generate_output_video': False,
            'output_dir': 'data/results',
            'confidence_threshold': 0.6,
        },
        'artifacts': {
            'enabled': False,
            'output_dir': 'data/results/artifacts',
            'save_invalid': True,
            'save_low_confidence': True,
            'confidence_threshold': 0.75,
            'max_saved_per_run': 10,
        },
        'scenarios': {
            'low_light': {
                'brightness_threshold': 90.0,
                'contrast_threshold': 40.0,
                'ocr_confidence_threshold': 0.5,
                'fallback_confidence_threshold': 0.72,
                'max_variants': 6,
            },
            'small_plate': {
                'area_ratio_threshold': 0.015,
                'height_threshold': 60,
                'ocr_confidence_threshold': 0.55,
                'fallback_confidence_threshold': 0.75,
                'max_variants': 6,
            },
            'video': {
                'moving': {
                    'skip_frames': 2,
                    'temporal_min_observations': 2,
                },
                'stationary': {
                    'skip_frames': 5,
                    'temporal_min_observations': 3,
                },
            },
        },
        'evaluation': {
            'fixtures_dir': 'data/fixtures',
            'manifest_path': 'data/fixtures/manifest.template.json',
            'reports_dir': 'data/results/evaluation',
        },
        'calibration': {
            'detector_thresholds': [0.20, 0.25],
            'ocr_thresholds': [0.55, 0.60],
            'fallback_thresholds': [0.70, 0.80],
        },
    }
