"""Testes unitários para src/ocr/paddle_engine.py — PaddleOCR Engine."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.ocr.paddle_engine import PaddleOCREngine, _paddle_available


def _dummy_image(h=100, w=300):
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _dummy_gray_image(h=100, w=300):
    return np.full((h, w), 128, dtype=np.uint8)


class TestAvailability:
    def test_is_available_false_when_not_installed(self):
        """Sem paddleocr instalado, engine deve reportar not available sem crash."""
        with patch('src.ocr.paddle_engine._paddle_available', return_value=False):
            engine = PaddleOCREngine()
            assert engine.is_available is False
            # Recognize também não pode crashar
            assert engine.recognize(_dummy_image()) == []

    def test_recognize_empty_input(self):
        with patch('src.ocr.paddle_engine._paddle_available', return_value=False):
            engine = PaddleOCREngine()
            assert engine.recognize(None) == []
            assert engine.recognize(np.array([])) == []


class TestCleanPlateText:
    def test_strips_non_alnum(self):
        assert PaddleOCREngine._clean_plate_text("ABC-1234") == "ABC1234"

    def test_uppercases(self):
        assert PaddleOCREngine._clean_plate_text("abc1d23") == "ABC1D23"

    def test_extracts_plate_from_longer_string(self):
        # "CARRO ABC1D23 PLACA" → com chars pegamos a sequência válida
        # (após remoção, teremos "CARROABC1D23PLACA" e regex captura ABC1D23)
        result = PaddleOCREngine._clean_plate_text("ABC1D23XX")
        assert result == "ABC1D23"

    def test_empty(self):
        assert PaddleOCREngine._clean_plate_text("") == ""
        assert PaddleOCREngine._clean_plate_text("   ") == ""


class TestFormatRank:
    def test_mercosul_top_rank(self):
        assert PaddleOCREngine._format_rank("ABC1D23") == 2

    def test_old_top_rank(self):
        assert PaddleOCREngine._format_rank("ABC1234") == 2

    def test_seven_chars_medium(self):
        assert PaddleOCREngine._format_rank("ABCDEFG") == 1

    def test_other_zero(self):
        assert PaddleOCREngine._format_rank("ABC") == 0


class TestCombineConfidence:
    def test_mercosul_boost(self):
        c = PaddleOCREngine._combine_confidence(0.95, "ABC1D23")
        assert c > 0.85

    def test_invalid_format_lower(self):
        # Score nativo alto mas formato inválido → confiança intermediária
        c_invalid = PaddleOCREngine._combine_confidence(0.95, "XYZ")
        c_valid = PaddleOCREngine._combine_confidence(0.95, "ABC1D23")
        assert c_invalid < c_valid

    def test_bounds(self):
        assert PaddleOCREngine._combine_confidence(0.0, "") == pytest.approx(0.0, abs=0.01)
        assert PaddleOCREngine._combine_confidence(1.0, "ABC1D23") <= 1.0


class TestParsePaddleOutput:
    def test_empty_output(self):
        assert PaddleOCREngine._parse_paddle_output([]) == []
        assert PaddleOCREngine._parse_paddle_output(None) == []
        assert PaddleOCREngine._parse_paddle_output([None]) == []

    def test_single_plate_mercosul(self):
        # Estrutura nativa PaddleOCR: [[ [bbox, (text, score)], ... ]]
        raw = [[
            [[[0, 0], [100, 0], [100, 50], [0, 50]], ("ABC1D23", 0.92)],
        ]]
        results = PaddleOCREngine._parse_paddle_output(raw)
        assert len(results) == 1
        assert results[0]['text'] == 'ABC1D23'
        assert results[0]['engine'] == 'paddle_ocr'
        assert results[0]['confidence'] > 0.85
        # char_confidences deve ser preenchido
        assert len(results[0]['char_confidences']) == 7

    def test_single_plate_old_format(self):
        raw = [[
            [[[0, 0], [100, 0], [100, 50], [0, 50]], ("ABC1234", 0.88)],
        ]]
        results = PaddleOCREngine._parse_paddle_output(raw)
        assert len(results) == 1
        assert results[0]['text'] == 'ABC1234'

    def test_multiple_fragments_concatenate(self):
        # PaddleOCR às vezes segmenta linha → duas detecções "ABC" + "1D23"
        raw = [[
            [[[0, 0], [50, 0], [50, 50], [0, 50]], ("ABC", 0.90)],
            [[[50, 0], [100, 0], [100, 50], [50, 50]], ("1D23", 0.88)],
        ]]
        results = PaddleOCREngine._parse_paddle_output(raw)
        assert len(results) == 1
        assert results[0]['text'] == 'ABC1D23'

    def test_noise_fragment_ignored_when_best_valid(self):
        raw = [[
            [[[0, 0], [100, 0], [100, 50], [0, 50]], ("ABC1D23", 0.95)],
            [[[0, 60], [100, 60], [100, 100], [0, 100]], ("X", 0.30)],
        ]]
        results = PaddleOCREngine._parse_paddle_output(raw)
        assert len(results) == 1
        assert results[0]['text'] == 'ABC1D23'


class TestRecognizeWithMock:
    def test_recognize_calls_paddle(self):
        """Com _ocr mockado e is_available=True, recognize delega para a API Paddle 3.x."""
        with patch('src.ocr.paddle_engine._paddle_available', return_value=False):
            engine = PaddleOCREngine()

        # Injetar manualmente para simular estado "instalado"
        fake_paddle = MagicMock()
        fake_paddle.predict.return_value = [{
            'rec_texts': ['ABC1D23'],
            'rec_scores': [0.91],
            'rec_polys': [[[0, 0], [100, 0], [100, 50], [0, 50]]],
        }]
        engine._ocr = fake_paddle
        engine.is_available = True

        results = engine.recognize(_dummy_image())
        assert len(results) == 1
        assert results[0]['text'] == 'ABC1D23'
        assert results[0]['engine'] == 'paddle_ocr'
        fake_paddle.predict.assert_called_once()

    def test_recognize_converts_grayscale_before_calling_paddle(self):
        with patch('src.ocr.paddle_engine._paddle_available', return_value=False):
            engine = PaddleOCREngine()

        fake_paddle = MagicMock()
        fake_paddle.predict.return_value = [{
            'rec_texts': ['ABC1D23'],
            'rec_scores': [0.91],
            'rec_polys': [[[0, 0], [100, 0], [100, 50], [0, 50]]],
        }]
        engine._ocr = fake_paddle
        engine.is_available = True

        results = engine.recognize(_dummy_gray_image())

        assert len(results) == 1
        prepared = fake_paddle.predict.call_args.args[0]
        assert prepared.shape == (100, 300, 3)
        assert prepared.dtype == np.uint8

    def test_recognize_handles_paddle_exception(self):
        with patch('src.ocr.paddle_engine._paddle_available', return_value=False):
            engine = PaddleOCREngine()

        fake_paddle = MagicMock()
        fake_paddle.predict.side_effect = RuntimeError("CUDA OOM")
        engine._ocr = fake_paddle
        engine.is_available = True

        # Não deve propagar a exceção
        results = engine.recognize(_dummy_image())
        assert results == []


class TestOCRManagerIntegration:
    """PaddleOCR deve ser aceito pelo OCRManager sem alteração (polimorfismo)."""

    def test_manager_accepts_paddle_engine(self):
        from src.ocr.manager import OCRManager

        with patch('src.ocr.paddle_engine._paddle_available', return_value=False):
            engine = PaddleOCREngine()

        # is_available=False → recognize retorna [] mas não crasha o manager
        manager = OCRManager(engine=engine, auto_fallback_on_failure=False)
        results = manager.recognize(_dummy_image())
        assert results == []
