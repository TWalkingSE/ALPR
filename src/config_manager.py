# src/config_manager.py
"""
Gerenciamento centralizado de configuração do ALPR.
Carrega config.yaml, fornece defaults e faz merge.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

# Diretório base do projeto
PROJECT_DIR = Path(__file__).parent.parent


def get_default_config() -> Dict[str, Any]:
    """Retorna configurações padrão enxutas da ALPR v2."""
    return {
        'models': {
            'detector': {
                'dir': 'models/yolo',
                'default': 'yolo11l-plate.pt',
                'confidence': 0.25,
                'device': 'auto',
                'enable_sahi': True,
                'sahi_slice_size': 640,
                'sahi_overlap_ratio': 0.25,
                'sahi_retry_confidence_threshold': 0.55,
                'sahi_retry_area_ratio_threshold': 0.01,
                'sahi_retry_large_image_threshold': 1600,
                'sahi_merge_iou_threshold': 0.45,
            },
        },
        'ocr': {
            'engine': 'paddle',
            'try_multiple_variants': True,
            'max_variants': 5,
            'top_k_candidates': 5,
            'paddle': {
                'lang': 'pt',
                'use_gpu': True,
                'use_angle_cls': True,
                'det_limit_side_len': 960,
                'rec_batch_num': 6,
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
            'log_all_calls': True,
        },
        'pipeline': {
            'crop_margin': 0.10,
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
            'skip_frames': 2,
            'max_frames': 0,
            'generate_output_video': True,
            'output_dir': 'data/results',
        },
        'artifacts': {
            'enabled': True,
            'output_dir': 'data/results/artifacts',
            'save_invalid': True,
            'save_low_confidence': True,
            'confidence_threshold': 0.75,
            'max_saved_per_run': 50,
        },
        'scenarios': {
            'low_light': {
                'brightness_threshold': 90.0,
                'contrast_threshold': 40.0,
                'ocr_confidence_threshold': 0.50,
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
            'detector_thresholds': [0.20, 0.25, 0.30],
            'ocr_thresholds': [0.55, 0.60, 0.65],
            'fallback_thresholds': [0.70, 0.80, 0.90],
        },
        'quality': {
            'enabled': True,
            'low_quality_threshold': 0.45,
            'snr_review_threshold': 9.0,
            'motion_blur_review_threshold': 0.35,
        },
        'forensic': {
            'enabled': True,
            'jpeg_quality': 90,
            'review_threshold': 0.55,
            'high_risk_threshold': 0.75,
        },
        'reports': {
            'enabled': True,
            'output_dir': 'data/results/reports',
            'prefer_artifact_dir': True,
        },
        'llm_validation': {
            'enabled': False,
            'base_url': 'http://127.0.0.1:11434',
            'model': '',
            'timeout': 20.0,
            'allow_override': True,
            'ambiguity_gap_threshold': 0.12,
            'min_decision_confidence': 0.70,
        },
    }


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Carrega configurações do arquivo YAML com merge sobre defaults.

    Args:
        config_path: Caminho relativo ao diretório do projeto

    Returns:
        Dicionário de configuração completo
    """
    defaults = get_default_config()
    config_file = PROJECT_DIR / config_path

    if not config_file.exists():
        logger.warning(f"Arquivo {config_path} não encontrado. Usando configurações padrão.")
        return defaults

    try:
        with open(config_file, encoding='utf-8') as f:
            file_config = yaml.safe_load(f) or {}

        # Deep merge: file_config sobrescreve defaults
        merged = _deep_merge(defaults, file_config)
        logger.info(f"Configuração carregada de {config_file}")
        return merged

    except Exception as e:
        logger.error(f"Erro ao carregar config: {e}")
        return defaults


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Merge profundo de dois dicionários.
    Override sobrescreve base, recursivamente para sub-dicts.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
