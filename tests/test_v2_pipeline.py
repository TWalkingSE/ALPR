from __future__ import annotations

import numpy as np

from src.validator import PlateValidator
from src.v2.config import AppConfig
from src.v2.pipeline import LocalAnalysisPipeline


class _FakeDetector:
    is_loaded = True

    def detect(self, image, confidence=None):
        return [{'bbox': (10, 10, 60, 30), 'confidence': 0.88}]

    def extract_plate_regions(self, image, detections, add_margin=0.10):
        del detections, add_margin
        return [{'image': image[10:30, 10:60], 'bbox': (10, 10, 60, 30), 'confidence': 0.88}]


class _FakeNormalizer:
    def normalize(self, image):
        return image


class _FakePreprocessor:
    def process(self, image, quality_result=None):
        del quality_result
        return [image]


class _FakeOCRManager:
    def __init__(self):
        self.engines = ['fake']

    def recognize(self, image, original_image=None, preprocessed_variants=None, visual_format_hint=None):
        del image, original_image, preprocessed_variants, visual_format_hint
        return [{
            'text': 'ABC1D23',
            'confidence': 0.92,
            'engine': 'paddle_ocr',
            'char_confidences': [('A', 0.9)] * 7,
        }]


class _FakeNgram:
    def rank_candidates(self, candidates, ocr_confidences=None):
        del ocr_confidences
        return [(candidate, 0.5) for candidate in candidates]


class _FakeLLMValidator:
    def __init__(self):
        self.allow_override = True
        self.ambiguity_gap_threshold = 0.12
        self.min_decision_confidence = 0.70
        self.last_resolved_model = 'qwen3.5:9b-q8_0'
        self.model = 'qwen3.5:9b-q8_0'

    def validate_candidates(
        self,
        raw_text,
        current_plate,
        candidates,
        validation_details,
        quality_metrics,
        char_confidences,
        scenario_tags,
    ):
        del raw_text, current_plate, validation_details, quality_metrics, char_confidences, scenario_tags
        assert len(candidates) >= 2
        return {
            'enabled': True,
            'performed': True,
            'available': True,
            'model': self.model,
            'selected_plate': 'ABC1D23',
            'decision_confidence': 0.84,
            'should_override': True,
            'reason': 'mercosul candidate preferred',
        }


def test_local_pipeline_process_image_returns_valid_result():
    image = np.zeros((60, 160, 3), dtype=np.uint8)
    pipeline = LocalAnalysisPipeline(
        detector=_FakeDetector(),
        geometric_normalizer=_FakeNormalizer(),
        preprocessor=_FakePreprocessor(),
        ocr_engine=_FakeOCRManager(),
        validator=PlateValidator(),
        ngram_model=_FakeNgram(),
        config=AppConfig(),
    )

    results = pipeline.process_image(image)

    assert len(results) == 1
    assert results[0].plate_text == 'ABC1D23'
    assert results[0].is_valid is True
    assert results[0].format_type == 'mercosul'
    assert results[0].ocr_engine == 'paddle_ocr'
    assert 'low_light' in results[0].scenario_tags
    assert 'snr' in results[0].quality_metrics
    assert results[0].validation_details['is_valid'] is True
    assert 'tampering_score' in results[0].forensic_analysis


def test_local_pipeline_generates_structured_report(tmp_path):
    image = np.zeros((80, 180, 3), dtype=np.uint8)
    config = AppConfig()
    config.artifacts.output_dir = str(tmp_path / 'artifacts')
    config.reports.output_dir = str(tmp_path / 'reports')

    pipeline = LocalAnalysisPipeline(
        detector=_FakeDetector(),
        geometric_normalizer=_FakeNormalizer(),
        preprocessor=_FakePreprocessor(),
        ocr_engine=_FakeOCRManager(),
        validator=PlateValidator(),
        ngram_model=_FakeNgram(),
        config=config,
    )

    results = pipeline.process_image(image, image_bytes=b'test-image-bytes', input_file_path='sample.jpg')

    assert len(results) == 1
    assert results[0].report_path
    assert results[0].report_payload['source']['sha256']
    assert results[0].report_payload['recognition']['plate_text'] == 'ABC1D23'


def test_local_pipeline_saves_artifacts_for_invalid_low_confidence_result(tmp_path):
    class _LowConfidenceOCR:
        def __init__(self):
            self.engines = ['fake']

        def recognize(self, image, original_image=None, preprocessed_variants=None, visual_format_hint=None):
            del image, original_image, preprocessed_variants, visual_format_hint
            return [{
                'text': 'A8C1023',
                'confidence': 0.42,
                'engine': 'paddle_ocr',
                'char_confidences': [('A', 0.7), ('8', 0.4), ('C', 0.7), ('1', 0.6), ('0', 0.3), ('2', 0.6), ('3', 0.6)],
            }]

    image = np.zeros((60, 160, 3), dtype=np.uint8)
    config = AppConfig()
    config.artifacts.output_dir = str(tmp_path / 'artifacts')
    config.artifacts.enabled = True
    config.artifacts.confidence_threshold = 0.8
    pipeline = LocalAnalysisPipeline(
        detector=_FakeDetector(),
        geometric_normalizer=_FakeNormalizer(),
        preprocessor=_FakePreprocessor(),
        ocr_engine=_LowConfidenceOCR(),
        validator=PlateValidator(),
        ngram_model=_FakeNgram(),
        config=config,
    )

    results = pipeline.process_image(image)

    assert len(results) == 1
    assert results[0].artifact_dir
    assert 'metadata' in results[0].artifact_files
    assert results[0].report_path


def test_local_pipeline_applies_optional_llm_override_after_top_k():
    class _AmbiguousOCR:
        def __init__(self):
            self.engines = ['fake']

        def recognize(self, image, original_image=None, preprocessed_variants=None, visual_format_hint=None):
            del image, original_image, preprocessed_variants, visual_format_hint
            return [{
                'text': 'ABC1023',
                'confidence': 0.71,
                'engine': 'paddle_ocr',
                'char_confidences': [('A', 0.9), ('B', 0.9), ('C', 0.9), ('1', 0.7), ('0', 0.4), ('2', 0.8), ('3', 0.8)],
            }]

    image = np.zeros((60, 160, 3), dtype=np.uint8)
    config = AppConfig()
    config.llm_validation.enabled = True
    pipeline = LocalAnalysisPipeline(
        detector=_FakeDetector(),
        geometric_normalizer=_FakeNormalizer(),
        preprocessor=_FakePreprocessor(),
        ocr_engine=_AmbiguousOCR(),
        validator=PlateValidator(),
        ngram_model=_FakeNgram(),
        config=config,
    )
    pipeline.llm_validator = _FakeLLMValidator()
    pipeline._build_alternatives = lambda **kwargs: [
        {
            'text': 'ABC1023',
            'probability': 0.51,
            'changes': 'no-change',
            'support_count': 1,
            'is_valid': True,
            'format_type': 'old',
            'validation_score': 0.92,
            'score_breakdown': {},
        },
        {
            'text': 'ABC1D23',
            'probability': 0.49,
            'changes': '5:0->D',
            'support_count': 1,
            'is_valid': True,
            'format_type': 'mercosul',
            'validation_score': 0.93,
            'score_breakdown': {},
        },
    ]

    results = pipeline.process_image(image)

    assert len(results) == 1
    assert results[0].plate_text == 'ABC1D23'
    assert results[0].format_type == 'mercosul'
    assert results[0].llm_validation['performed'] is True
    assert results[0].llm_validation['applied_override'] is True
    assert results[0].llm_validation['model'] == 'qwen3.5:9b-q8_0'
    assert 'llm_override' in results[0].warnings
    assert results[0].report_payload['llm_validation']['applied_override'] is True