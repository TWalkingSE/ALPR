"""Testes para src/v2/vehicle_attributes.py — atributos do veículo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.v2.vehicle_attributes import (
    COLOR_UNKNOWN,
    VehicleAttributeAnalyzer,
    VehicleAttributes,
    detect_dominant_color,
)


def _solid(b, g, r, h=64, w=64):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (b, g, r)
    return img


class TestDetectDominantColor:
    def test_red(self):
        color, conf = detect_dominant_color(_solid(0, 0, 255))
        assert color == 'vermelho'
        assert conf > 0.9

    def test_blue(self):
        color, _ = detect_dominant_color(_solid(255, 0, 0))
        assert color == 'azul'

    def test_green(self):
        color, _ = detect_dominant_color(_solid(0, 255, 0))
        assert color == 'verde'

    def test_black(self):
        color, _ = detect_dominant_color(_solid(0, 0, 0))
        assert color == 'preto'

    def test_white(self):
        color, _ = detect_dominant_color(_solid(255, 255, 255))
        assert color == 'branco'

    def test_empty_input(self):
        color, conf = detect_dominant_color(np.array([]))
        assert color == COLOR_UNKNOWN
        assert conf == 0.0

    def test_none_input(self):
        color, conf = detect_dominant_color(None)
        assert color == COLOR_UNKNOWN

    def test_grayscale_input_supported(self):
        gray = np.full((40, 40), 10, dtype=np.uint8)
        color, _ = detect_dominant_color(gray)
        assert color == 'preto'


class _FakeClassifier:
    def __init__(self, make='Fiat', model='Uno', conf=0.8):
        self._ret = (make, model, conf)

    def predict(self, vehicle_crop):
        return self._ret


class _BrokenClassifier:
    def predict(self, vehicle_crop):
        raise RuntimeError("modelo indisponível")


class TestVehicleAttributeAnalyzer:
    def test_disabled_returns_disabled_source(self):
        analyzer = VehicleAttributeAnalyzer(enabled=False)
        attrs = analyzer.analyze(_solid(0, 0, 255, h=200, w=200), (80, 150, 120, 175))
        assert attrs.source == 'disabled'
        assert attrs.color == COLOR_UNKNOWN

    def test_color_only_without_classifier(self):
        analyzer = VehicleAttributeAnalyzer(enabled=True)
        full = _solid(255, 0, 0, h=300, w=300)  # azul
        attrs = analyzer.analyze(full, (130, 220, 170, 250))
        assert attrs.source == 'color_only'
        assert attrs.color == 'azul'
        assert attrs.make == ''
        assert attrs.model == ''

    def test_with_classifier_returns_make_model(self):
        analyzer = VehicleAttributeAnalyzer(
            enabled=True, make_model_classifier=_FakeClassifier()
        )
        full = _solid(0, 0, 0, h=300, w=300)
        attrs = analyzer.analyze(full, (130, 220, 170, 250))
        assert attrs.source == 'analyzed'
        assert attrs.make == 'Fiat'
        assert attrs.model == 'Uno'
        assert attrs.make_model_confidence == pytest.approx(0.8)

    def test_classifier_exception_degrades_gracefully(self):
        analyzer = VehicleAttributeAnalyzer(
            enabled=True, make_model_classifier=_BrokenClassifier()
        )
        full = _solid(0, 0, 255, h=300, w=300)
        attrs = analyzer.analyze(full, (130, 220, 170, 250))
        # Não deve crashar; marca/modelo vazios, mas a cor ainda é computada.
        assert attrs.make == ''
        assert attrs.color == 'vermelho'

    def test_missing_image_unavailable(self):
        analyzer = VehicleAttributeAnalyzer(enabled=True)
        attrs = analyzer.analyze(None, (10, 10, 50, 30))
        assert attrs.source == 'unavailable'

    def test_missing_bbox_unavailable(self):
        analyzer = VehicleAttributeAnalyzer(enabled=True)
        attrs = analyzer.analyze(_solid(0, 0, 0, h=100, w=100), None)
        assert attrs.source == 'unavailable'

    def test_roi_bbox_within_image(self):
        analyzer = VehicleAttributeAnalyzer(enabled=True)
        full = _solid(0, 0, 0, h=400, w=400)
        attrs = analyzer.analyze(full, (180, 300, 220, 330))
        assert attrs.vehicle_bbox is not None
        x1, y1, x2, y2 = attrs.vehicle_bbox
        assert 0 <= x1 < x2 <= 400
        assert 0 <= y1 < y2 <= 400

    def test_to_dict_serializable(self):
        attrs = VehicleAttributes(color='azul', color_confidence=0.9, vehicle_bbox=(1, 2, 3, 4))
        data = attrs.to_dict()
        assert data['color'] == 'azul'
        assert data['vehicle_bbox'] == [1, 2, 3, 4]
