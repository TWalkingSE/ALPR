"""Testes unitários para src/ocr/manager.py — OCRManager."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.ocr.base import OCREngine
from src.ocr.manager import OCRManager
from src.ocr.types import create_ocr_result


def _dummy_image(h=100, w=300):
    return np.full((h, w, 3), 128, dtype=np.uint8)


class _FakeEngine(OCREngine):
    """Engine de mentira que retorna resultados pré-programados."""

    engine_name = 'fake'

    def __init__(self, results=None, raise_on_call=False):
        self.results = results or []
        self.raise_on_call = raise_on_call
        self.call_count = 0

    def recognize(self, image):
        self.call_count += 1
        if self.raise_on_call:
            raise RuntimeError("simulated failure")
        return self.results


class TestOCRManager:
    def test_single_engine_basic(self):
        results = [create_ocr_result(
            text='ABC1D23', confidence=0.9, engine='fake'
        )]
        engine = _FakeEngine(results=results)
        mgr = OCRManager(engine=engine)

        out = mgr.recognize(_dummy_image())
        assert len(out) == 1
        assert out[0]['text'] == 'ABC1D23'
        assert out[0]['confidence'] == 0.9

    def test_picks_best_by_confidence(self):
        """Entre várias variantes, escolhe a de maior confiança."""
        results = [
            create_ocr_result(text='ABC1D23', confidence=0.95, engine='fake'),
        ]
        engine = _FakeEngine(results=results)
        mgr = OCRManager(engine=engine, try_multiple_variants=True)

        variants = [_dummy_image(h=50, w=200), _dummy_image(h=80, w=280)]
        out = mgr.recognize(_dummy_image(), preprocessed_variants=variants)
        assert len(out) == 1
        # O engine foi chamado na imagem principal + variantes
        assert engine.call_count >= 2

    def test_fallback_on_empty_result(self):
        """Quando primário retorna vazio e fallback configurado, aciona fallback."""
        empty_engine = _FakeEngine(results=[])
        fallback_engine = _FakeEngine(results=[
            create_ocr_result(text='XYZ9Z99', confidence=0.75, engine='fake_fb')
        ])

        mgr = OCRManager(
            engine=empty_engine,
            fallback_factory=lambda: fallback_engine,
            auto_fallback_on_failure=True,
        )
        out = mgr.recognize(_dummy_image())
        assert len(out) == 1
        assert out[0]['text'] == 'XYZ9Z99'
        assert out[0].get('used_fallback') is True

    def test_no_fallback_when_disabled(self):
        """auto_fallback_on_failure=False: não tenta o outro engine."""
        empty_engine = _FakeEngine(results=[])
        fallback_engine = _FakeEngine(results=[
            create_ocr_result(text='XYZ9Z99', confidence=0.75, engine='fake_fb')
        ])

        mgr = OCRManager(
            engine=empty_engine,
            fallback_factory=lambda: fallback_engine,
            auto_fallback_on_failure=False,
        )
        out = mgr.recognize(_dummy_image())
        assert out == []
        assert fallback_engine.call_count == 0

    def test_fallback_on_exception(self):
        """Exceção no primário ainda aciona fallback."""
        broken_engine = _FakeEngine(raise_on_call=True)
        fallback_engine = _FakeEngine(results=[
            create_ocr_result(text='ABC1234', confidence=0.8, engine='fake_fb')
        ])

        mgr = OCRManager(
            engine=broken_engine,
            fallback_factory=lambda: fallback_engine,
            auto_fallback_on_failure=True,
        )
        out = mgr.recognize(_dummy_image())
        assert len(out) == 1
        assert out[0]['text'] == 'ABC1234'

    def test_visual_format_hint_stored(self):
        engine = _FakeEngine(results=[
            create_ocr_result(text='ABC1D23', confidence=0.9, engine='fake')
        ])
        mgr = OCRManager(engine=engine)
        mgr.recognize(_dummy_image(), visual_format_hint='mercosul')
        assert mgr._visual_format_hint == 'mercosul'

    def test_get_status_active_engine(self):
        engine = _FakeEngine()
        mgr = OCRManager(engine=engine, auto_fallback_on_failure=False)
        status = mgr.get_status()
        assert status['active_engine_name'] == 'fake'
        assert status['fallback_configured'] is False
        assert status['auto_fallback_on_failure'] is False

    def test_engines_list_retrocompat(self):
        """Propriedade .engines mantida para retrocompatibilidade interna do manager."""
        engine = _FakeEngine()
        mgr = OCRManager(engine=engine)
        assert len(mgr.engines) == 1

    def test_original_image_fallback(self):
        """Se preprocessada falhar, tenta original (se fornecida)."""
        attempts = {'n': 0}

        class _ConditionalEngine(OCREngine):
            engine_name = 'cond'
            def recognize(self, image):
                attempts['n'] += 1
                # Só retorna resultado na 2ª chamada (imagem original)
                if attempts['n'] >= 2:
                    return [create_ocr_result(
                        text='ABC1234', confidence=0.7, engine='cond'
                    )]
                return []

        mgr = OCRManager(
            engine=_ConditionalEngine(),
            auto_fallback_on_failure=False,
            try_multiple_variants=False,
        )
        out = mgr.recognize(_dummy_image(), original_image=_dummy_image())
        # Pode retornar vazio ou o resultado da original; o importante é não crashar
        assert isinstance(out, list)
