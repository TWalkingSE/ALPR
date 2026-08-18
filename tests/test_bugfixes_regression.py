# tests/test_bugfixes_regression.py
"""Regressões dos bugs corrigidos na revisão do ALPR 2.0."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.v2.ollama_validation import OllamaSmartValidator
from src.video_processor import VideoProcessor


class TestVideoWriterUnboundLocal:
    """`finally` referenciava `video_writer` antes da atribuição.

    Se a leitura de metadados do vídeo falhasse, o UnboundLocalError no
    `finally` mascarava a exceção original e tornava o erro real
    indiagnosticável.
    """

    def test_metadata_failure_surfaces_original_error(self, tmp_path):
        processor = VideoProcessor(
            output_dir=str(tmp_path),
            generate_output_video=True,
            enable_temporal_voting=False,
        )

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = RuntimeError('falha real ao ler metadados')

        with (
            patch('src.video_processor.cv2.VideoCapture', return_value=cap),
            pytest.raises(RuntimeError, match='falha real ao ler metadados'),
        ):
            processor.process_video('qualquer.mp4', pipeline=MagicMock())

        cap.release.assert_called_once()


class TestMercosulVowelRule:
    """A placa Mercosul (LLLNLNN) aceita QUALQUER letra na 5ª posição.

    O prompt enviado ao LLM afirmava o contrário, instruindo o modelo a
    descartar candidatos válidos — em contradição direta com o validador.
    """

    def test_prompt_does_not_forbid_vowels(self):
        validator = OllamaSmartValidator(model='fake-model')
        captured = {}

        def _fake_post(url, json=None, **kwargs):
            captured['prompt'] = json['prompt']
            raise RuntimeError('curto-circuito: só queremos inspecionar o prompt')

        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=None)
        client.post = _fake_post

        with patch('src.v2.ollama_validation.httpx.Client', return_value=client):
            validator.validate_candidates(
                raw_text='ABC1A23',
                current_plate='ABC1A23',
                candidates=[{'text': 'ABC1A23'}, {'text': 'ABC1O23'}],
                validation_details={},
                quality_metrics={},
                char_confidences=[],
                scenario_tags=[],
            )

        prompt = captured['prompt']
        assert 'cannot_be_vowel' not in prompt
        assert 'mercosul_position_5_accepts_any_letter' in prompt


class TestRankUniquePlatesPurity:
    """`rank_unique_plates` roda a cada rerun do Streamlit e mutava a entrada."""

    def test_does_not_mutate_input(self):
        plates = {
            'ABC1D23': {
                'plate_text': 'ABC1D23',
                'best_confidence': 0.9,
                'total_detections': 3,
                'all_confidences': [0.9, 0.8],
                'quality_scores': [0.9],
                'char_confirmation_scores': [0.85],
                'temporal_span_frames': 4,
            }
        }
        snapshot = {k: dict(v) for k, v in plates.items()}

        ranked = VideoProcessor.rank_unique_plates(plates)

        assert plates == snapshot, 'a entrada não deve ser modificada'
        assert 'composite_score' in next(iter(ranked.values()))

    def test_repeated_calls_are_stable(self):
        plates = {
            'ABC1D23': {
                'plate_text': 'ABC1D23',
                'best_confidence': 0.9,
                'total_detections': 3,
                'all_confidences': [0.9, 0.8],
                'quality_scores': [0.9],
                'char_confirmation_scores': [0.85],
                'temporal_span_frames': 4,
            }
        }
        first = VideoProcessor.rank_unique_plates(plates)['ABC1D23']['composite_score']
        second = VideoProcessor.rank_unique_plates(plates)['ABC1D23']['composite_score']
        assert first == second


class TestPreprocessorVariantBudget:
    """O orçamento não pode alterar o conjunto entregue ao OCR.

    As binarizações só são puladas quando o consumidor — que as prioriza por
    último — comprovadamente não chegaria a usá-las.
    """

    @staticmethod
    def _is_binary(image):
        return getattr(image, 'ndim', 0) == 2 and np.unique(image).size <= 4

    @classmethod
    def _prioritized(cls, variants):
        non_binary = [v for v in variants if not cls._is_binary(v)]
        binary = [v for v in variants if cls._is_binary(v)]
        return non_binary + binary

    def test_budget_preserves_delivered_variants(self):
        from src.preprocessor import ImagePreprocessor

        preprocessor = ImagePreprocessor(deskew=False)
        image = np.random.default_rng(0).integers(0, 255, (100, 300, 3)).astype(np.uint8)

        for budget in (1, 2, 4, 6):
            sem_orcamento = self._prioritized(preprocessor.process(image)[1:])[:budget]
            com_orcamento = self._prioritized(
                preprocessor.process(image, max_variants=budget)[1:]
            )[:budget]

            assert len(sem_orcamento) == len(com_orcamento), f'budget={budget}'
            for antes, depois in zip(sem_orcamento, com_orcamento, strict=True):
                assert np.array_equal(antes, depois), f'budget={budget}'

    def test_generous_budget_still_produces_binarizations(self):
        from src.preprocessor import ImagePreprocessor

        preprocessor = ImagePreprocessor(deskew=False)
        image = np.random.default_rng(0).integers(0, 255, (100, 300, 3)).astype(np.uint8)

        variants = preprocessor.process(image, max_variants=99)[1:]
        assert any(self._is_binary(v) for v in variants)
