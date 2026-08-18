# tests/test_config_defaults_match_yaml.py
"""
Trava o drift entre `get_default_config()` e o `config.yaml` versionado.

Os defaults do código são o fallback usado quando o `config.yaml` some ou falha
no parse. Quando as duas fontes divergem, esse fallback entrega silenciosamente
um schema diferente do documentado — foi o que aconteceu com
`vehicle_attributes`, `video.confidence_threshold` e a zona de silêncio do
PaddleOCR, presentes só no YAML.
"""

from pathlib import Path

import pytest
import yaml

from src.config_manager import get_default_config

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_YAML = PROJECT_ROOT / 'config.yaml'


@pytest.fixture(scope='module')
def yaml_config():
    return yaml.safe_load(CONFIG_YAML.read_text(encoding='utf-8')) or {}


def _leaf_paths(config, prefix=''):
    """Caminhos pontilhados de todas as chaves folha do dicionário."""
    paths = set()
    for key, value in config.items():
        path = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            paths |= _leaf_paths(value, path)
        else:
            paths.add(path)
    return paths


def test_yaml_has_no_key_missing_from_defaults(yaml_config):
    missing = _leaf_paths(yaml_config) - _leaf_paths(get_default_config())
    assert not missing, (
        'Chaves presentes no config.yaml e ausentes em get_default_config(): '
        + ', '.join(sorted(missing))
    )


def test_defaults_have_no_key_missing_from_yaml(yaml_config):
    missing = _leaf_paths(get_default_config()) - _leaf_paths(yaml_config)
    assert not missing, (
        'Chaves presentes em get_default_config() e ausentes no config.yaml: '
        + ', '.join(sorted(missing))
    )


def test_top_level_sections_match(yaml_config):
    assert set(yaml_config) == set(get_default_config())
