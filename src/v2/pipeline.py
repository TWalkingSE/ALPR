"""Minimal, deterministic local ALPR pipeline for v2."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.constants import SIMILAR_CHARS, get_confusion_weight
from src.detector import PlateDetector
from src.geometric_normalizer import GeometricNormalizer
from src.ocr.manager import OCRManager
from src.ocr.paddle_engine import PaddleOCREngine
from src.plate_patterns import PlateNgramModel
from src.preprocessor import ImagePreprocessor
from src.validator import PlateValidator, is_plausible_plate_prefix
from src.v2.config import AppConfig
from src.v2.forensics import ForensicAnalyzer
from src.v2.models import LocalPlateResult, normalize_plate_text
from src.v2.ollama_validation import OllamaSmartValidator
from src.v2.quality import QualityAssessor
from src.v2.reporting import ReportBuilder

logger = logging.getLogger(__name__)


@dataclass
class PlateContextProfile:
    """Heuristic context used to adapt OCR thresholds per plate crop."""

    tags: List[str]
    brightness: float
    contrast: float
    sharpness: float
    snr: float
    motion_blur: float
    edge_density: float
    highlight_clipping: float
    shadow_clipping: float
    quality_band: str
    area_ratio: float
    quality_score: float
    effective_ocr_threshold: float
    effective_fallback_threshold: float
    effective_max_variants: int


class LocalAnalysisPipeline:
    """Offline-first local analysis pipeline for ALPR v2."""

    def __init__(
        self,
        detector: PlateDetector,
        geometric_normalizer: GeometricNormalizer,
        preprocessor: ImagePreprocessor,
        ocr_engine: OCRManager,
        validator: PlateValidator,
        ngram_model: PlateNgramModel,
        config: AppConfig,
    ):
        self.detector = detector
        self.geometric_normalizer = geometric_normalizer
        self.preprocessor = preprocessor
        self.ocr_engine = ocr_engine
        self.validator = validator
        self.ngram_model = ngram_model
        self.config = config
        self.ocr_confidence_threshold = config.ocr.confidence_threshold
        self.fallback_threshold = config.ocr.fallback_threshold
        self.top_k_candidates = config.ocr.top_k_candidates
        self.llm_threshold = config.llm_validation.min_decision_confidence
        self.premium_provider = None
        self._artifact_output_dir = Path(config.artifacts.output_dir)
        self._saved_artifacts = 0
        self.quality_assessor = QualityAssessor() if config.quality.enabled else None
        self.llm_validator = (
            OllamaSmartValidator.from_settings(config.llm_validation)
            if config.llm_validation.enabled
            else None
        )
        self.forensic_analyzer = (
            ForensicAnalyzer(
                jpeg_quality=config.forensic.jpeg_quality,
                review_threshold=config.forensic.review_threshold,
                high_risk_threshold=config.forensic.high_risk_threshold,
            )
            if config.forensic.enabled
            else None
        )
        self.report_builder = ReportBuilder(
            enabled=config.reports.enabled,
            output_dir=config.reports.output_dir,
            prefer_artifact_dir=config.reports.prefer_artifact_dir,
        )

    @classmethod
    def from_settings(
        cls,
        config: AppConfig,
        project_dir: Path,
        model_path: Optional[str] = None,
    ) -> 'LocalAnalysisPipeline':
        detector_model_path = model_path or str(
            project_dir / config.detector.models_dir / config.detector.model_name
        )

        detector = PlateDetector(
            model_path=detector_model_path,
            confidence=config.detector.confidence,
            device='auto' if config.detector.use_gpu else 'cpu',
            enable_sahi=config.detector.enable_sahi,
            sahi_slice_size=config.detector.sahi_slice_size,
            sahi_overlap_ratio=config.detector.sahi_overlap_ratio,
            sahi_retry_confidence_threshold=config.detector.sahi_retry_confidence_threshold,
            sahi_retry_area_ratio_threshold=config.detector.sahi_retry_area_ratio_threshold,
            sahi_retry_large_image_threshold=config.detector.sahi_retry_large_image_threshold,
            sahi_merge_iou_threshold=config.detector.sahi_merge_iou_threshold,
        )
        geometric_normalizer = GeometricNormalizer(
            enabled=True,
            perspective_correction=True,
            rotation_correction=True,
            contrast_equalization=False,
            standard_resize=True,
            target_width=300,
            target_height=100,
        )
        preprocessor = ImagePreprocessor(
            enhance_contrast=True,
            remove_noise=True,
            sharpen=True,
            adaptive_threshold=True,
            optimize_for_brazilian_plates=True,
            morphological_cleanup=True,
            deskew=False,
            multi_binarization=config.ocr.try_multiple_variants,
            adaptive_clahe=True,
            use_nlmeans_denoising=True,
        )

        paddle_engine = PaddleOCREngine(
            lang=config.ocr.lang,
            use_gpu=config.ocr.use_gpu,
            use_angle_cls=config.ocr.use_angle_cls,
            det_limit_side_len=config.ocr.det_limit_side_len,
            rec_batch_num=config.ocr.rec_batch_num,
            min_score=config.ocr.min_score,
        )
        if not paddle_engine.is_available:
            detail = getattr(paddle_engine, 'init_error', None) or 'PaddleOCR indisponivel'
            raise RuntimeError(detail)

        ocr_engine = OCRManager(
            engine=paddle_engine,
            fallback_factory=None,
            auto_fallback_on_failure=False,
            try_multiple_variants=config.ocr.try_multiple_variants,
            max_variants=config.ocr.max_variants,
        )

        return cls(
            detector=detector,
            geometric_normalizer=geometric_normalizer,
            preprocessor=preprocessor,
            ocr_engine=ocr_engine,
            validator=PlateValidator(),
            ngram_model=PlateNgramModel(enabled=True),
            config=config,
        )

    def process_image(
        self,
        image: np.ndarray,
        detector_confidence: Optional[float] = None,
        return_all_plates: bool = True,
        image_bytes: Optional[bytes] = None,
        input_file_path: Optional[str] = None,
        temporal_prior: Optional[Dict[str, float]] = None,
    ) -> List[LocalPlateResult]:
        """Run the local pipeline on a single image."""
        if image is None or image.size == 0:
            return []

        effective_detector_confidence = self._resolve_detector_confidence(
            image,
            detector_confidence,
        )
        detections = self.detector.detect(
            image,
            confidence=effective_detector_confidence,
        )
        if not detections:
            return []

        regions = self.detector.extract_plate_regions(
            image,
            detections,
            add_margin=self.config.detector.crop_margin,
        )

        results: List[LocalPlateResult] = []
        for region in regions:
            result = self._process_plate_region(
                region,
                source_shape=image.shape[:2],
                temporal_prior=temporal_prior,
                image_bytes=image_bytes,
                input_file_path=input_file_path,
            )
            if result is not None:
                results.append(result)

        results.sort(
            key=lambda item: (
                item.is_valid,
                item.confidence,
                item.quality_score,
                item.detection_confidence,
            ),
            reverse=True,
        )
        if not return_all_plates and results:
            return [results[0]]
        return results

    def process_video_frame(
        self,
        frame: np.ndarray,
        detector_confidence: Optional[float] = None,
        temporal_prior: Optional[Dict[str, float]] = None,
    ) -> List[LocalPlateResult]:
        """Compatibility helper for video processing."""
        return self.process_image(
            frame,
            detector_confidence=detector_confidence,
            temporal_prior=temporal_prior,
        )

    def _process_plate_region(
        self,
        region: Dict[str, Any],
        source_shape: Optional[Tuple[int, int]] = None,
        temporal_prior: Optional[Dict[str, float]] = None,
        image_bytes: Optional[bytes] = None,
        input_file_path: Optional[str] = None,
    ) -> Optional[LocalPlateResult]:
        start_total = time.perf_counter()
        step_times: Dict[str, float] = {}
        warnings: List[str] = []

        original_crop = region['image']
        bbox = tuple(region['bbox'])
        detection_confidence = float(region.get('confidence', 0.0))

        step_start = time.perf_counter()
        normalized = self.geometric_normalizer.normalize(original_crop)
        step_times['geometric_normalization'] = (time.perf_counter() - step_start) * 1000

        quality_assessment = None
        if self.quality_assessor is not None:
            step_start = time.perf_counter()
            quality_assessment = self.quality_assessor.assess(normalized)
            step_times['quality_assessment'] = (time.perf_counter() - step_start) * 1000

        context_profile = self._analyze_plate_context(
            region,
            normalized,
            source_shape,
            quality_assessment=quality_assessment,
        )

        step_start = time.perf_counter()
        variants = self.preprocessor.process(normalized, quality_result=context_profile)
        step_times['preprocessing'] = (time.perf_counter() - step_start) * 1000
        if not variants:
            warnings.append('preprocess_empty')
            variants = [normalized]

        step_start = time.perf_counter()
        ocr_results = self.ocr_engine.recognize(
            image=normalized,
            original_image=normalized,
            preprocessed_variants=variants[1:context_profile.effective_max_variants],
            visual_format_hint=None,
        )
        step_times['ocr'] = (time.perf_counter() - step_start) * 1000
        if not ocr_results:
            return None

        ocr_best = ocr_results[0]
        raw_text = str(ocr_best.get('text', '')).strip().upper()
        normalized_raw = normalize_plate_text(raw_text)
        confidence = float(ocr_best.get('confidence', 0.0))
        raw_char_confidences = ocr_best.get('char_confidences', [])

        step_start = time.perf_counter()
        validation_details = self.validator.describe_validation(normalized_raw)
        validated_plate = validation_details.get('suggested_plate') or self.validator.validate(normalized_raw)
        step_times['validation'] = (time.perf_counter() - step_start) * 1000

        if validated_plate:
            final_text = validated_plate
            normalized_final = normalize_plate_text(validated_plate)
            is_valid = True
        else:
            final_text = normalized_raw or raw_text
            normalized_final = normalize_plate_text(final_text)
            is_valid = False

        format_type = validation_details.get('format', self._detect_format(normalized_final))
        char_confidences = self._build_char_confidences(
            final_text,
            raw_char_confidences,
            confidence,
        )

        alternatives: List[Dict[str, Any]] = []
        if (not is_valid) or confidence < context_profile.effective_fallback_threshold:
            step_start = time.perf_counter()
            alternatives = self._build_alternatives(
                raw_text=normalized_raw,
                validated_text=normalized_final,
                confidence=confidence,
                raw_char_confidences=raw_char_confidences,
                ocr_candidates=ocr_results,
                context_profile=context_profile,
                temporal_prior=temporal_prior,
            )
            step_times['candidate_ranking'] = (time.perf_counter() - step_start) * 1000

        llm_validation: Dict[str, Any] = {}
        if self.llm_validator is not None:
            step_start = time.perf_counter()
            llm_validation = self._maybe_run_llm_validation(
                raw_text=normalized_raw,
                current_plate=final_text,
                confidence=confidence,
                is_valid=is_valid,
                format_type=format_type,
                alternatives=alternatives,
                validation_details=validation_details,
                quality_metrics={
                    'brightness': context_profile.brightness,
                    'contrast': context_profile.contrast,
                    'sharpness': context_profile.sharpness,
                    'snr': context_profile.snr,
                    'motion_blur': context_profile.motion_blur,
                    'edge_density': context_profile.edge_density,
                    'highlight_clipping': context_profile.highlight_clipping,
                    'shadow_clipping': context_profile.shadow_clipping,
                    'area_ratio': context_profile.area_ratio,
                    'ocr_threshold': context_profile.effective_ocr_threshold,
                    'fallback_threshold': context_profile.effective_fallback_threshold,
                },
                char_confidences=char_confidences,
                scenario_tags=context_profile.tags,
                effective_fallback_threshold=context_profile.effective_fallback_threshold,
            )
            step_times['llm_validation'] = (time.perf_counter() - step_start) * 1000

            if llm_validation.get('applied_override'):
                final_text = str(llm_validation.get('final_plate', final_text)).strip().upper()
                normalized_final = normalize_plate_text(final_text)
                validation_details = self.validator.describe_validation(normalized_final)
                validated_plate = (
                    validation_details.get('suggested_plate')
                    or self.validator.validate(normalized_final)
                )
                if validated_plate:
                    final_text = validated_plate
                    normalized_final = normalize_plate_text(validated_plate)
                    is_valid = True
                else:
                    is_valid = False
                format_type = validation_details.get('format', self._detect_format(normalized_final))
                char_confidences = self._build_char_confidences(
                    final_text,
                    raw_char_confidences,
                    confidence,
                )
                warnings.append('llm_override')

        result = LocalPlateResult(
            plate_text=final_text,
            confidence=confidence,
            detection_confidence=detection_confidence,
            format_type=format_type,
            is_valid=is_valid,
            original_crop=original_crop,
            bbox=bbox,
            normalized_crop=normalized,
            preprocessed_image=variants[-1] if variants else normalized,
            ocr_engine=str(ocr_best.get('engine', 'paddle_ocr')),
            char_confidences=char_confidences,
            alternative_plates=alternatives,
            processing_time_ms=0.0,
            pipeline_steps_time=step_times,
            raw_ocr_text=raw_text,
            normalized_text=normalized_final,
            warnings=warnings,
            scenario_tags=context_profile.tags,
            quality_score=context_profile.quality_score,
            quality_metrics={
                'brightness': context_profile.brightness,
                'contrast': context_profile.contrast,
                'sharpness': context_profile.sharpness,
                'snr': context_profile.snr,
                'motion_blur': context_profile.motion_blur,
                'edge_density': context_profile.edge_density,
                'highlight_clipping': context_profile.highlight_clipping,
                'shadow_clipping': context_profile.shadow_clipping,
                'area_ratio': context_profile.area_ratio,
                'ocr_threshold': context_profile.effective_ocr_threshold,
                'fallback_threshold': context_profile.effective_fallback_threshold,
            },
            quality_assessment=quality_assessment.to_dict() if quality_assessment is not None else {},
            validation_details=validation_details,
            llm_validation=llm_validation,
            detector_metadata=dict(region.get('detector_metadata', {})),
        )

        if self.forensic_analyzer is not None:
            step_start = time.perf_counter()
            forensic_analysis = self.forensic_analyzer.analyze(original_crop)
            step_times['forensic_review'] = (time.perf_counter() - step_start) * 1000
            result.forensic_analysis = forensic_analysis.to_dict()

        step_start = time.perf_counter()
        artifact_dir, artifact_files = self._maybe_save_artifacts(result)
        step_times['artifact_persistence'] = (time.perf_counter() - step_start) * 1000
        result.artifact_dir = artifact_dir
        result.artifact_files = artifact_files

        result.processing_time_ms = (time.perf_counter() - start_total) * 1000
        if self.report_builder.enabled:
            step_start = time.perf_counter()
            report_payload, report_path = self.report_builder.generate(
                result,
                image_bytes=image_bytes,
                input_file_path=input_file_path,
            )
            step_times['reporting'] = (time.perf_counter() - step_start) * 1000
            result.report_payload = report_payload
            result.report_path = report_path

        result.processing_time_ms = (time.perf_counter() - start_total) * 1000
        return result

    def _resolve_detector_confidence(
        self,
        image: np.ndarray,
        detector_confidence: Optional[float],
    ) -> float:
        effective_confidence = (
            float(detector_confidence)
            if detector_confidence is not None
            else float(self.config.detector.confidence)
        )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        brightness = float(np.mean(gray)) if gray.size else 0.0

        if brightness < self.config.scenarios.low_light.brightness_threshold:
            effective_confidence = max(0.05, effective_confidence - 0.05)

        if (
            self.config.detector.enable_sahi
            and max(image.shape[:2]) >= self.config.detector.sahi_retry_large_image_threshold
        ):
            effective_confidence = max(0.05, effective_confidence - 0.03)

        return effective_confidence

    def _analyze_plate_context(
        self,
        region: Dict[str, Any],
        normalized_crop: np.ndarray,
        source_shape: Optional[Tuple[int, int]],
        quality_assessment=None,
    ) -> PlateContextProfile:
        gray = (
            cv2.cvtColor(normalized_crop, cv2.COLOR_BGR2GRAY)
            if len(normalized_crop.shape) == 3
            else normalized_crop
        )

        brightness = float(getattr(quality_assessment, 'brightness', np.mean(gray) if gray.size else 0.0))
        contrast = float(getattr(quality_assessment, 'contrast', np.std(gray) if gray.size else 0.0))
        sharpness = float(
            getattr(
                quality_assessment,
                'sharpness',
                cv2.Laplacian(gray, cv2.CV_64F).var() if gray.size else 0.0,
            )
        )
        snr = float(getattr(quality_assessment, 'snr', 0.0))
        motion_blur = float(getattr(quality_assessment, 'motion_blur', 0.0))
        edge_density = float(getattr(quality_assessment, 'edge_density', 0.0))
        highlight_clipping = float(getattr(quality_assessment, 'highlight_clipping', 0.0))
        shadow_clipping = float(getattr(quality_assessment, 'shadow_clipping', 0.0))
        quality_band = str(getattr(quality_assessment, 'quality_band', 'unknown'))

        bbox = region.get('original_bbox') or region.get('bbox') or (0, 0, 0, 0)
        bbox_height = max(0, int(bbox[3]) - int(bbox[1])) if len(bbox) == 4 else 0
        bbox_width = max(0, int(bbox[2]) - int(bbox[0])) if len(bbox) == 4 else 0
        image_area = 0
        if source_shape is not None and len(source_shape) == 2:
            image_area = max(1, int(source_shape[0]) * int(source_shape[1]))
        bbox_area = bbox_width * bbox_height
        area_ratio = float(bbox_area / image_area) if image_area > 0 else 0.0

        low_light = (
            brightness <= self.config.scenarios.low_light.brightness_threshold
            or contrast <= self.config.scenarios.low_light.contrast_threshold
        )
        small_plate = (
            area_ratio > 0
            and area_ratio <= self.config.scenarios.small_plate.area_ratio_threshold
        ) or (
            bbox_height > 0
            and bbox_height <= self.config.scenarios.small_plate.height_threshold
        )

        if quality_assessment is not None:
            quality_score = float(quality_assessment.quality_score)
        else:
            brightness_score = min(1.0, brightness / max(self.config.scenarios.low_light.brightness_threshold, 1.0))
            contrast_score = min(1.0, contrast / max(self.config.scenarios.low_light.contrast_threshold, 1.0))
            sharpness_score = min(1.0, sharpness / 120.0)
            size_reference = max(self.config.scenarios.small_plate.height_threshold, 1)
            size_score = min(1.0, max(float(gray.shape[0]), float(bbox_height)) / size_reference)
            quality_score = float(
                max(
                    0.0,
                    min(
                        1.0,
                        0.30 * brightness_score
                        + 0.20 * contrast_score
                        + 0.30 * sharpness_score
                        + 0.20 * size_score,
                    ),
                )
            )

        tags: List[str] = []
        effective_ocr_threshold = float(self.ocr_confidence_threshold)
        effective_fallback_threshold = float(self.fallback_threshold)
        effective_max_variants = int(self.config.ocr.max_variants)

        if low_light:
            tags.append('low_light')
            effective_ocr_threshold = min(
                effective_ocr_threshold,
                self.config.scenarios.low_light.ocr_confidence_threshold,
            )
            effective_fallback_threshold = min(
                effective_fallback_threshold,
                self.config.scenarios.low_light.fallback_confidence_threshold,
            )
            effective_max_variants = max(
                effective_max_variants,
                self.config.scenarios.low_light.max_variants,
            )

        if small_plate:
            tags.append('small_plate')
            effective_ocr_threshold = min(
                effective_ocr_threshold,
                self.config.scenarios.small_plate.ocr_confidence_threshold,
            )
            effective_fallback_threshold = min(
                effective_fallback_threshold,
                self.config.scenarios.small_plate.fallback_confidence_threshold,
            )
            effective_max_variants = max(
                effective_max_variants,
                self.config.scenarios.small_plate.max_variants,
            )

        if snr and snr < self.config.quality.snr_review_threshold:
            tags.append('low_snr')
            effective_ocr_threshold = max(0.30, effective_ocr_threshold - 0.05)
            effective_max_variants += 1

        if motion_blur > self.config.quality.motion_blur_review_threshold:
            tags.append('motion_blur')
            effective_fallback_threshold = max(0.55, effective_fallback_threshold - 0.05)
            effective_max_variants += 1

        if quality_score < self.config.quality.low_quality_threshold:
            tags.append('low_quality')

        return PlateContextProfile(
            tags=tags,
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
            snr=snr,
            motion_blur=motion_blur,
            edge_density=edge_density,
            highlight_clipping=highlight_clipping,
            shadow_clipping=shadow_clipping,
            quality_band=quality_band,
            area_ratio=area_ratio,
            quality_score=quality_score,
            effective_ocr_threshold=effective_ocr_threshold,
            effective_fallback_threshold=effective_fallback_threshold,
            effective_max_variants=max(1, effective_max_variants),
        )

    def _build_alternatives(
        self,
        raw_text: str,
        validated_text: str,
        confidence: float,
        raw_char_confidences: Any,
        ocr_candidates: Optional[List[Dict[str, Any]]] = None,
        context_profile: Optional[PlateContextProfile] = None,
        temporal_prior: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        candidate_pool: Dict[str, Dict[str, Any]] = {}

        def _register_candidate(
            candidate_text: str,
            source_confidence: float,
            candidate_char_confidences: Any = None,
        ) -> None:
            clean = normalize_plate_text(candidate_text)
            if not clean:
                return

            entry = candidate_pool.setdefault(
                clean,
                {
                    'base_confidence': 0.0,
                    'support_count': 0,
                    'char_confidences': [],
                },
            )
            entry['base_confidence'] = max(entry['base_confidence'], float(source_confidence))
            entry['support_count'] += 1

            normalized_char_confidences = self._build_char_confidences(
                clean,
                candidate_char_confidences,
                source_confidence,
            )
            if len(normalized_char_confidences) > len(entry['char_confidences']):
                entry['char_confidences'] = normalized_char_confidences

        for candidate in (raw_text, validated_text):
            _register_candidate(candidate, confidence, raw_char_confidences)

        for item in ocr_candidates or []:
            item_text = str(item.get('text', '')).strip().upper()
            item_confidence = float(item.get('confidence', confidence))
            item_char_confidences = item.get('char_confidences', [])
            _register_candidate(item_text, item_confidence, item_char_confidences)

            validated_candidate = self.validator.validate(item_text)
            if validated_candidate:
                _register_candidate(validated_candidate, item_confidence, item_char_confidences)

        if raw_text and hasattr(self.validator, '_normalize_to_7_chars'):
            for candidate in self.validator._normalize_to_7_chars(raw_text):
                _register_candidate(candidate, confidence, raw_char_confidences)

        if not candidate_pool:
            return []

        candidates = list(candidate_pool)
        ranked = self.ngram_model.rank_candidates(
            candidates,
            ocr_confidences=[candidate_pool[item]['base_confidence'] for item in candidates],
        )
        ngram_scores = {candidate: float(score) for candidate, score in ranked}

        raw_count = max(1, len(ocr_candidates or []))

        scored_alternatives: List[Dict[str, Any]] = []
        for candidate in candidates:
            entry = candidate_pool[candidate]
            ngram_score = ngram_scores.get(candidate, 0.0)
            ocr_support = self._estimate_candidate_support(
                candidate,
                raw_text,
                entry.get('char_confidences') or raw_char_confidences,
                entry.get('base_confidence', confidence),
            )
            pattern_score = self._candidate_pattern_score(candidate)
            consensus_score = min(1.0, entry.get('support_count', 0) / raw_count)
            temporal_score = min(
                1.0,
                float((temporal_prior or {}).get(normalize_plate_text(candidate), 0.0)),
            )
            validation_snapshot = self.validator.describe_validation(candidate)
            composite_score = (
                0.30 * ngram_score
                + 0.30 * ocr_support
                + 0.20 * pattern_score
                + 0.10 * consensus_score
                + 0.10 * temporal_score
            )

            formatted = self.validator.validate(candidate) or candidate
            scored_alternatives.append({
                'text': formatted,
                'probability': composite_score,
                'changes': self._describe_changes(raw_text, candidate),
                'support_count': int(entry.get('support_count', 0)),
                'is_valid': bool(validation_snapshot.get('is_valid', False)),
                'format_type': validation_snapshot.get('format', 'unknown'),
                'validation_score': max(
                    float(validation_snapshot.get('old_format_score', 0.0)),
                    float(validation_snapshot.get('mercosul_format_score', 0.0)),
                ),
                'score_breakdown': {
                    'ngram': ngram_score,
                    'ocr_support': ocr_support,
                    'pattern': pattern_score,
                    'consensus': consensus_score,
                    'temporal': temporal_score,
                    'context_quality': context_profile.quality_score if context_profile else 0.0,
                },
            })

        scored_alternatives.sort(key=lambda item: item['probability'], reverse=True)
        top_alternatives = scored_alternatives[: self.top_k_candidates]
        total_score = sum(max(0.0, float(item['probability'])) for item in top_alternatives)

        alternatives: List[Dict[str, Any]] = []
        for item in top_alternatives:
            probability = float(item['probability'])
            if total_score > 0:
                probability /= total_score

            alternatives.append({
                'text': item['text'],
                'probability': probability,
                'changes': item['changes'],
                'support_count': item['support_count'],
                'is_valid': item['is_valid'],
                'format_type': item['format_type'],
                'validation_score': item['validation_score'],
                'score_breakdown': item['score_breakdown'],
            })
        return alternatives

    def _candidate_pattern_score(self, candidate: str) -> float:
        clean = normalize_plate_text(candidate)
        if not clean:
            return 0.0

        score = 0.20
        if len(clean) == 7:
            score += 0.15

        if self.validator.is_old_format(clean) or self.validator.is_mercosul_format(clean):
            score += 0.35

        score += 0.20 * is_plausible_plate_prefix(clean)

        if self.validator.is_mercosul_format(clean) and len(clean) > 4 and clean[4] not in 'AEIOU':
            score += 0.10

        return min(1.0, score)

    def _estimate_candidate_support(
        self,
        candidate: str,
        raw_text: str,
        raw_char_confidences: Any,
        overall_confidence: float,
    ) -> float:
        candidate_clean = normalize_plate_text(candidate)
        raw_clean = normalize_plate_text(raw_text)
        if not candidate_clean:
            return 0.0

        normalized_confidences = self._build_char_confidences(
            raw_clean or candidate_clean,
            raw_char_confidences,
            overall_confidence,
        )

        scores: List[float] = []
        for index, candidate_char in enumerate(candidate_clean):
            raw_char = raw_clean[index] if index < len(raw_clean) else ''
            raw_confidence = overall_confidence
            if index < len(normalized_confidences):
                source_char, source_confidence = normalized_confidences[index]
                raw_confidence = float(source_confidence)
                if not raw_char:
                    raw_char = normalize_plate_text(str(source_char))[:1]

            if raw_char == candidate_char:
                char_score = raw_confidence
            elif raw_char and candidate_char in SIMILAR_CHARS.get(raw_char, []):
                char_score = max(
                    0.12,
                    raw_confidence * get_confusion_weight(raw_char, candidate_char)
                    + (1.0 - raw_confidence) * 0.25,
                )
            else:
                char_score = max(0.05, (1.0 - raw_confidence) * 0.60)

            scores.append(min(1.0, char_score))

        return float(np.mean(scores)) if scores else float(overall_confidence)

    def _build_char_confidences(
        self,
        plate_text: str,
        raw_confidences: Any,
        overall_confidence: float,
    ) -> List[tuple[str, float]]:
        clean_text = normalize_plate_text(plate_text)
        normalized: List[tuple[str, float]] = []

        for item in raw_confidences or []:
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            char, confidence = item
            normalized.append((str(char).upper(), float(confidence)))

        if not normalized:
            return [(char, overall_confidence) for char in clean_text]

        if len(normalized) < len(clean_text):
            normalized.extend(
                (clean_text[index], overall_confidence)
                for index in range(len(normalized), len(clean_text))
            )
        return normalized[:len(clean_text)]

    def _build_llm_candidates(
        self,
        current_plate: str,
        confidence: float,
        is_valid: bool,
        format_type: str,
        validation_details: Dict[str, Any],
        alternatives: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        primary_text = normalize_plate_text(current_plate)
        primary_validation_score = max(
            float(validation_details.get('old_format_score', 0.0)),
            float(validation_details.get('mercosul_format_score', 0.0)),
        )

        ordered_candidates = [
            {
                'text': primary_text,
                'probability': float(confidence),
                'is_valid': is_valid,
                'format_type': format_type,
                'validation_score': primary_validation_score,
                'source': 'deterministic_primary',
            },
            *alternatives,
        ]

        for item in ordered_candidates:
            candidate_text = normalize_plate_text(str(item.get('text', '')).strip().upper())
            if not candidate_text:
                continue

            candidate_entry = merged.setdefault(
                candidate_text,
                {
                    'text': candidate_text,
                    'probability': 0.0,
                    'is_valid': False,
                    'format_type': 'unknown',
                    'validation_score': 0.0,
                    'source': 'top_k',
                },
            )
            candidate_entry['probability'] = max(
                float(candidate_entry.get('probability', 0.0)),
                float(item.get('probability', 0.0)),
            )
            candidate_entry['is_valid'] = bool(item.get('is_valid', candidate_entry['is_valid']))
            candidate_entry['format_type'] = str(
                item.get('format_type', candidate_entry['format_type'])
            )
            candidate_entry['validation_score'] = max(
                float(candidate_entry.get('validation_score', 0.0)),
                float(item.get('validation_score', 0.0)),
            )
            if item.get('source') == 'deterministic_primary':
                candidate_entry['source'] = 'deterministic_primary'

        ranked = sorted(
            merged.values(),
            key=lambda item: (
                float(item.get('probability', 0.0)),
                float(item.get('validation_score', 0.0)),
                item.get('source') == 'deterministic_primary',
            ),
            reverse=True,
        )
        return ranked[: self.top_k_candidates]

    def _maybe_run_llm_validation(
        self,
        raw_text: str,
        current_plate: str,
        confidence: float,
        is_valid: bool,
        format_type: str,
        alternatives: List[Dict[str, Any]],
        validation_details: Dict[str, Any],
        quality_metrics: Dict[str, float],
        char_confidences: List[tuple[str, float]],
        scenario_tags: List[str],
        effective_fallback_threshold: float,
    ) -> Dict[str, Any]:
        if self.llm_validator is None:
            return {}

        candidates = self._build_llm_candidates(
            current_plate=current_plate,
            confidence=confidence,
            is_valid=is_valid,
            format_type=format_type,
            validation_details=validation_details,
            alternatives=alternatives,
        )
        if len(candidates) < 2:
            return {
                'enabled': True,
                'performed': False,
                'available': True,
                'reason': 'not_enough_candidates',
                'current_plate': normalize_plate_text(current_plate),
                'candidates': candidates,
            }

        top_probability = float(candidates[0].get('probability', 0.0))
        runner_up_probability = float(candidates[1].get('probability', 0.0))
        ambiguity_gap = max(0.0, top_probability - runner_up_probability)
        should_invoke = (
            (not is_valid)
            or confidence < effective_fallback_threshold
            or ambiguity_gap <= self.llm_validator.ambiguity_gap_threshold
        )
        if not should_invoke:
            return {
                'enabled': True,
                'performed': False,
                'available': True,
                'reason': 'not_needed',
                'current_plate': normalize_plate_text(current_plate),
                'ambiguity_gap': ambiguity_gap,
                'candidates': candidates,
            }

        llm_validation = self.llm_validator.validate_candidates(
            raw_text=raw_text,
            current_plate=current_plate,
            candidates=candidates,
            validation_details=validation_details,
            quality_metrics=quality_metrics,
            char_confidences=char_confidences,
            scenario_tags=scenario_tags,
        )
        llm_validation['current_plate'] = normalize_plate_text(current_plate)
        llm_validation['candidates'] = candidates
        llm_validation['ambiguity_gap'] = ambiguity_gap

        selected_plate = normalize_plate_text(str(llm_validation.get('selected_plate', '')))
        applied_override = (
            bool(llm_validation.get('performed'))
            and self.llm_validator.allow_override
            and bool(llm_validation.get('should_override'))
            and float(llm_validation.get('decision_confidence', 0.0))
            >= self.llm_validator.min_decision_confidence
            and selected_plate
            and selected_plate != normalize_plate_text(current_plate)
        )
        llm_validation['applied_override'] = applied_override
        llm_validation['final_plate'] = selected_plate if applied_override else normalize_plate_text(current_plate)
        return llm_validation

    def _maybe_save_artifacts(self, result: LocalPlateResult) -> Tuple[str, Dict[str, str]]:
        if not self.config.artifacts.enabled:
            return ('', {})

        if self._saved_artifacts >= self.config.artifacts.max_saved_per_run:
            return ('', {})

        should_save_invalid = self.config.artifacts.save_invalid and not result.is_valid
        should_save_low_conf = (
            self.config.artifacts.save_low_confidence
            and result.confidence < self.config.artifacts.confidence_threshold
        )
        if not should_save_invalid and not should_save_low_conf:
            return ('', {})

        self._artifact_output_dir.mkdir(parents=True, exist_ok=True)
        label = normalize_plate_text(result.plate_text or result.raw_ocr_text) or 'unknown'
        reasons = []
        if should_save_invalid:
            reasons.append('invalid')
        if should_save_low_conf:
            reasons.append('lowconf')

        self._saved_artifacts += 1
        artifact_dir = self._artifact_output_dir / (
            f"{time.strftime('%Y%m%d_%H%M%S')}_{self._saved_artifacts:03d}_{label}_{'_'.join(reasons)}"
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)

        artifact_files: Dict[str, str] = {}
        if result.original_crop is not None:
            original_path = artifact_dir / 'original_crop.png'
            self._write_image(original_path, result.original_crop)
            artifact_files['original_crop'] = str(original_path)

        if result.normalized_crop is not None:
            normalized_path = artifact_dir / 'normalized_crop.png'
            self._write_image(normalized_path, result.normalized_crop)
            artifact_files['normalized_crop'] = str(normalized_path)

        if result.preprocessed_image is not None:
            preprocessed_path = artifact_dir / 'preprocessed.png'
            self._write_image(preprocessed_path, result.preprocessed_image)
            artifact_files['preprocessed'] = str(preprocessed_path)

        metadata_path = artifact_dir / 'metadata.json'
        metadata = {
            'plate_text': result.plate_text,
            'raw_ocr_text': result.raw_ocr_text,
            'normalized_text': result.normalized_text,
            'confidence': result.confidence,
            'detection_confidence': result.detection_confidence,
            'format_type': result.format_type,
            'is_valid': result.is_valid,
            'warnings': result.warnings,
            'scenario_tags': result.scenario_tags,
            'quality_score': result.quality_score,
            'quality_metrics': result.quality_metrics,
            'quality_assessment': result.quality_assessment,
            'validation_details': result.validation_details,
            'llm_validation': result.llm_validation,
            'forensic_analysis': result.forensic_analysis,
            'detector_metadata': result.detector_metadata,
            'alternative_plates': result.alternative_plates,
            'artifact_reasons': reasons,
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=True, indent=2),
            encoding='utf-8',
        )
        artifact_files['metadata'] = str(metadata_path)
        return (str(artifact_dir), artifact_files)

    @staticmethod
    def _write_image(path: Path, image: np.ndarray) -> None:
        if image is None or image.size == 0:
            return
        cv2.imwrite(str(path), image)

    def _detect_format(self, plate_text: str) -> str:
        if self.validator.is_mercosul_format(plate_text):
            return 'mercosul'
        if self.validator.is_old_format(plate_text):
            return 'old'
        return 'unknown'

    @staticmethod
    def _describe_changes(raw_text: str, candidate: str) -> str:
        raw = normalize_plate_text(raw_text)
        cand = normalize_plate_text(candidate)
        if not raw:
            return 'candidate'

        changes: List[str] = []
        for index, current in enumerate(cand):
            previous = raw[index] if index < len(raw) else '_'
            if previous != current:
                changes.append(f'{index + 1}:{previous}->{current}')
        return ', '.join(changes) if changes else 'no-change'

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Expose runtime status information for the v2 UI."""
        return {
            'detector_loaded': self.detector is not None and self.detector.is_loaded,
            'geometric_normalizer_enabled': self.geometric_normalizer is not None,
            'preprocessor_enabled': self.preprocessor is not None,
            'ocr_engines_count': len(self.ocr_engine.engines) if hasattr(self.ocr_engine, 'engines') else 1,
            'validator_enabled': self.validator is not None,
            'fallback_enabled': self.ngram_model is not None,
            'ocr_confidence_threshold': self.ocr_confidence_threshold,
            'fallback_threshold': self.fallback_threshold,
            'top_k_candidates': self.top_k_candidates,
            'artifact_capture': self.config.artifacts.enabled,
            'quality_assessment_enabled': self.quality_assessor is not None,
            'llm_validation_enabled': self.llm_validator is not None,
            'llm_validation_model': (
                getattr(self.llm_validator, 'last_resolved_model', '')
                or getattr(self.llm_validator, 'model', '')
            )
            if self.llm_validator is not None
            else '',
            'forensic_review_enabled': self.forensic_analyzer is not None,
            'reporting_enabled': self.report_builder.enabled,
        }