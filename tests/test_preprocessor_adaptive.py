"""
Testes para o modo adaptativo do ImagePreprocessor:
- Sem quality_result → comportamento padrão (compat)
- quality_score >= 0.75 (Excelente) → menos variantes (sem multi_binarization, sem augmentation)
- quality_score [0.5, 0.75) (Suficiente) → comportamento padrão
- quality_score < 0.25 (Insuficiente) → modo agressivo (sharpen forte)
"""

import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from src.preprocessor import ImagePreprocessor


@dataclass
class FakeQualityResult:
    quality_score: float = 0.5
    quality_label: str = "Suficiente"


@pytest.fixture
def preprocessor():
    return ImagePreprocessor(
        enhance_contrast=True,
        remove_noise=True,
        sharpen=True,
        adaptive_threshold=True,
        morphological_cleanup=True,
        multi_binarization=True,
        adaptive_clahe=True,
        use_nlmeans_denoising=True,
    )


@pytest.fixture
def sample_plate():
    """Placa sintética 80x240 em escala de cinza convertida p/ BGR."""
    img = np.full((80, 240, 3), 200, dtype=np.uint8)
    # Alguns "caracteres" em preto
    for i in range(7):
        x = 20 + i * 30
        img[20:60, x:x + 18] = 30
    return img


class TestAdaptiveMode:

    def test_no_quality_result_behaves_as_default(self, preprocessor, sample_plate):
        out_default = preprocessor.process(sample_plate)
        out_explicit = preprocessor.process(sample_plate, quality_result=None)
        assert len(out_default) == len(out_explicit)
        assert len(out_default) >= 3

    def test_excelente_produces_fewer_variants(self, preprocessor, sample_plate):
        baseline = preprocessor.process(sample_plate)
        excelente = preprocessor.process(
            sample_plate, quality_result=FakeQualityResult(quality_score=0.85)
        )
        assert len(excelente) < len(baseline)
        # modo adaptativo registrado
        assert preprocessor._adaptive_mode == "excelente"

    def test_suficiente_matches_default_count(self, preprocessor, sample_plate):
        baseline = preprocessor.process(sample_plate)
        suficiente = preprocessor.process(
            sample_plate, quality_result=FakeQualityResult(quality_score=0.60)
        )
        assert len(suficiente) == len(baseline)
        assert preprocessor._adaptive_mode == "suficiente"

    def test_insuficiente_enables_aggressive_sharpen(self, preprocessor, sample_plate):
        preprocessor.process(
            sample_plate, quality_result=FakeQualityResult(quality_score=0.10)
        )
        assert preprocessor._adaptive_mode == "insuficiente"
        assert preprocessor._aggressive_sharpen_runtime is True

    def test_critica_mode_keeps_multi_bin_and_augmentation(self, preprocessor, sample_plate):
        out = preprocessor.process(
            sample_plate, quality_result=FakeQualityResult(quality_score=0.35)
        )
        assert preprocessor._adaptive_mode == "critica"
        assert preprocessor._enable_multi_bin_runtime is True
        assert preprocessor._enable_augmentation_runtime is True
        assert len(out) >= 5

    def test_excelente_disables_augmentation(self, preprocessor, sample_plate):
        preprocessor.process(
            sample_plate, quality_result=FakeQualityResult(quality_score=0.90)
        )
        assert preprocessor._enable_augmentation_runtime is False
        assert preprocessor._enable_multi_bin_runtime is False
