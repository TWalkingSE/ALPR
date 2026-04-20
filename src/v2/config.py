"""Typed configuration for ALPR v2."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.video_processor import VehicleMode


@dataclass
class DetectorConfig:
    models_dir: str = 'models/yolo'
    model_name: str = 'yolo11l-plate.pt'
    confidence: float = 0.25
    device: str = 'auto'
    use_gpu: bool = True
    enable_sahi: bool = True
    sahi_slice_size: int = 640
    sahi_overlap_ratio: float = 0.25
    sahi_retry_confidence_threshold: float = 0.55
    sahi_retry_area_ratio_threshold: float = 0.01
    sahi_retry_large_image_threshold: int = 1600
    sahi_merge_iou_threshold: float = 0.45
    crop_margin: float = 0.10


@dataclass
class OCRConfig:
    engine: str = 'paddle'
    try_multiple_variants: bool = True
    max_variants: int = 5
    top_k_candidates: int = 5
    confidence_threshold: float = 0.6
    fallback_threshold: float = 0.8
    lang: str = 'pt'
    use_gpu: bool = True
    use_angle_cls: bool = True
    det_limit_side_len: int = 960
    rec_batch_num: int = 6
    min_score: float = 0.3


@dataclass
class PremiumConfig:
    enabled: bool = False
    provider: str = 'platerecognizer'
    api_key: str = ''
    regions: List[str] = field(default_factory=lambda: ['br'])
    min_confidence: float = 0.5
    timeout: int = 30
    log_all_calls: bool = True


@dataclass
class ArtifactConfig:
    enabled: bool = True
    output_dir: str = 'data/results/artifacts'
    save_invalid: bool = True
    save_low_confidence: bool = True
    confidence_threshold: float = 0.75
    max_saved_per_run: int = 50


@dataclass
class ScenarioThresholdConfig:
    ocr_confidence_threshold: float = 0.6
    fallback_confidence_threshold: float = 0.8
    max_variants: int = 5


@dataclass
class LowLightScenarioConfig(ScenarioThresholdConfig):
    brightness_threshold: float = 90.0
    contrast_threshold: float = 40.0


@dataclass
class SmallPlateScenarioConfig(ScenarioThresholdConfig):
    area_ratio_threshold: float = 0.015
    height_threshold: int = 60


@dataclass
class VideoScenarioConfig:
    skip_frames: int = 2
    temporal_min_observations: int = 2


@dataclass
class ScenarioConfig:
    low_light: LowLightScenarioConfig = field(default_factory=LowLightScenarioConfig)
    small_plate: SmallPlateScenarioConfig = field(default_factory=SmallPlateScenarioConfig)
    moving_video: VideoScenarioConfig = field(default_factory=VideoScenarioConfig)
    stationary_video: VideoScenarioConfig = field(
        default_factory=lambda: VideoScenarioConfig(skip_frames=5, temporal_min_observations=3)
    )


@dataclass
class EvaluationConfig:
    fixtures_dir: str = 'data/fixtures'
    manifest_path: str = 'data/fixtures/manifest.template.json'
    reports_dir: str = 'data/results/evaluation'


@dataclass
class CalibrationConfig:
    detector_thresholds: List[float] = field(default_factory=lambda: [0.20, 0.25, 0.30])
    ocr_thresholds: List[float] = field(default_factory=lambda: [0.55, 0.60, 0.65])
    fallback_thresholds: List[float] = field(default_factory=lambda: [0.70, 0.80, 0.90])


@dataclass
class QualityConfig:
    enabled: bool = True
    low_quality_threshold: float = 0.45
    snr_review_threshold: float = 9.0
    motion_blur_review_threshold: float = 0.35


@dataclass
class ForensicConfig:
    enabled: bool = True
    jpeg_quality: int = 90
    review_threshold: float = 0.55
    high_risk_threshold: float = 0.75


@dataclass
class ReportConfig:
    enabled: bool = True
    output_dir: str = 'data/results/reports'
    prefer_artifact_dir: bool = True


@dataclass
class LLMValidationConfig:
    enabled: bool = False
    base_url: str = 'http://127.0.0.1:11434'
    model: str = ''
    timeout: float = 20.0
    allow_override: bool = True
    ambiguity_gap_threshold: float = 0.12
    min_decision_confidence: float = 0.70


@dataclass
class VideoConfig:
    vehicle_mode: VehicleMode = VehicleMode.MOVING
    skip_frames: int = 2
    max_frames: int = 0
    generate_output_video: bool = True
    output_dir: str = 'data/results'
    confidence_threshold: float = 0.3
    enable_temporal_voting: bool = True
    temporal_strategy: str = 'hybrid'
    temporal_min_observations: int = 2


@dataclass
class AppConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    premium: PremiumConfig = field(default_factory=PremiumConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    scenarios: ScenarioConfig = field(default_factory=ScenarioConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    forensic: ForensicConfig = field(default_factory=ForensicConfig)
    reports: ReportConfig = field(default_factory=ReportConfig)
    llm_validation: LLMValidationConfig = field(default_factory=LLMValidationConfig)

    def signature(self) -> tuple:
        """Return a stable signature for caching/rebuild decisions."""
        return (
            self.detector.models_dir,
            self.detector.model_name,
            round(self.detector.confidence, 4),
            self.detector.device,
            self.detector.use_gpu,
            self.detector.enable_sahi,
            self.detector.sahi_slice_size,
            round(self.detector.sahi_overlap_ratio, 4),
            round(self.detector.sahi_retry_confidence_threshold, 4),
            round(self.detector.sahi_retry_area_ratio_threshold, 6),
            self.detector.sahi_retry_large_image_threshold,
            round(self.detector.sahi_merge_iou_threshold, 4),
            round(self.detector.crop_margin, 4),
            self.ocr.engine,
            self.ocr.try_multiple_variants,
            self.ocr.max_variants,
            self.ocr.top_k_candidates,
            round(self.ocr.confidence_threshold, 4),
            round(self.ocr.fallback_threshold, 4),
            self.ocr.lang,
            self.ocr.use_gpu,
            self.ocr.use_angle_cls,
            self.ocr.det_limit_side_len,
            self.ocr.rec_batch_num,
            round(self.ocr.min_score, 4),
            self.premium.enabled,
            self.premium.provider,
            tuple(self.premium.regions),
            round(self.premium.min_confidence, 4),
            self.premium.timeout,
            self.video.vehicle_mode.value,
            self.video.skip_frames,
            self.video.max_frames,
            self.video.generate_output_video,
            self.video.output_dir,
            round(self.video.confidence_threshold, 4),
            self.video.enable_temporal_voting,
            self.video.temporal_strategy,
            self.video.temporal_min_observations,
            self.artifacts.enabled,
            self.artifacts.output_dir,
            self.artifacts.save_invalid,
            self.artifacts.save_low_confidence,
            round(self.artifacts.confidence_threshold, 4),
            self.artifacts.max_saved_per_run,
            round(self.scenarios.low_light.brightness_threshold, 4),
            round(self.scenarios.low_light.contrast_threshold, 4),
            round(self.scenarios.low_light.ocr_confidence_threshold, 4),
            round(self.scenarios.low_light.fallback_confidence_threshold, 4),
            self.scenarios.low_light.max_variants,
            round(self.scenarios.small_plate.area_ratio_threshold, 6),
            self.scenarios.small_plate.height_threshold,
            round(self.scenarios.small_plate.ocr_confidence_threshold, 4),
            round(self.scenarios.small_plate.fallback_confidence_threshold, 4),
            self.scenarios.small_plate.max_variants,
            self.scenarios.moving_video.skip_frames,
            self.scenarios.moving_video.temporal_min_observations,
            self.scenarios.stationary_video.skip_frames,
            self.scenarios.stationary_video.temporal_min_observations,
            self.evaluation.fixtures_dir,
            self.evaluation.manifest_path,
            self.evaluation.reports_dir,
            tuple(round(value, 4) for value in self.calibration.detector_thresholds),
            tuple(round(value, 4) for value in self.calibration.ocr_thresholds),
            tuple(round(value, 4) for value in self.calibration.fallback_thresholds),
            self.quality.enabled,
            round(self.quality.low_quality_threshold, 4),
            round(self.quality.snr_review_threshold, 4),
            round(self.quality.motion_blur_review_threshold, 4),
            self.forensic.enabled,
            self.forensic.jpeg_quality,
            round(self.forensic.review_threshold, 4),
            round(self.forensic.high_risk_threshold, 4),
            self.reports.enabled,
            self.reports.output_dir,
            self.reports.prefer_artifact_dir,
            self.llm_validation.enabled,
            self.llm_validation.base_url,
            self.llm_validation.model,
            round(self.llm_validation.timeout, 3),
            self.llm_validation.allow_override,
            round(self.llm_validation.ambiguity_gap_threshold, 4),
            round(self.llm_validation.min_decision_confidence, 4),
        )


def build_v2_config(raw_config: Dict[str, Any]) -> AppConfig:
    """Build the typed v2 configuration from the current YAML structure."""
    detector_cfg = raw_config.get('models', {}).get('detector', {})
    ocr_cfg = raw_config.get('ocr', {})
    paddle_cfg = ocr_cfg.get('paddle', {})
    premium_cfg = raw_config.get('premium_api', {})
    pipeline_cfg = raw_config.get('pipeline', {})
    video_cfg = raw_config.get('video', {})
    artifacts_cfg = raw_config.get('artifacts', {})
    scenarios_cfg = raw_config.get('scenarios', {})
    low_light_cfg = scenarios_cfg.get('low_light', {})
    small_plate_cfg = scenarios_cfg.get('small_plate', {})
    scenario_video_cfg = scenarios_cfg.get('video', {})
    moving_video_cfg = scenario_video_cfg.get('moving', {})
    stationary_video_cfg = scenario_video_cfg.get('stationary', {})
    evaluation_cfg = raw_config.get('evaluation', {})
    calibration_cfg = raw_config.get('calibration', {})
    quality_cfg = raw_config.get('quality', {})
    forensic_cfg = raw_config.get('forensic', {})
    reports_cfg = raw_config.get('reports', {})
    llm_cfg = raw_config.get('llm_validation', {})

    requested_engine = str(ocr_cfg.get('engine', 'paddle')).lower()
    engine = 'paddle' if requested_engine != 'paddle' else requested_engine

    detector = DetectorConfig(
        models_dir=str(detector_cfg.get('dir', 'models/yolo')),
        model_name=str(detector_cfg.get('default', 'yolo11l-plate.pt')),
        confidence=float(detector_cfg.get('confidence', 0.25)),
        device=str(detector_cfg.get('device', 'auto')),
        use_gpu=str(detector_cfg.get('device', 'auto')).lower() != 'cpu',
        enable_sahi=bool(detector_cfg.get('enable_sahi', True)),
        sahi_slice_size=int(detector_cfg.get('sahi_slice_size', 640)),
        sahi_overlap_ratio=float(detector_cfg.get('sahi_overlap_ratio', 0.25)),
        sahi_retry_confidence_threshold=float(detector_cfg.get('sahi_retry_confidence_threshold', 0.55)),
        sahi_retry_area_ratio_threshold=float(detector_cfg.get('sahi_retry_area_ratio_threshold', 0.01)),
        sahi_retry_large_image_threshold=max(256, int(detector_cfg.get('sahi_retry_large_image_threshold', 1600))),
        sahi_merge_iou_threshold=float(detector_cfg.get('sahi_merge_iou_threshold', 0.45)),
        crop_margin=float(pipeline_cfg.get('crop_margin', 0.10)),
    )

    ocr = OCRConfig(
        engine=engine,
        try_multiple_variants=bool(ocr_cfg.get('try_multiple_variants', True)),
        max_variants=max(1, int(ocr_cfg.get('max_variants', 5))),
        top_k_candidates=max(1, int(ocr_cfg.get('top_k_candidates', 5))),
        confidence_threshold=float(pipeline_cfg.get('ocr_confidence_threshold', 0.6)),
        fallback_threshold=float(pipeline_cfg.get('fallback_confidence_threshold', 0.8)),
        lang=str(paddle_cfg.get('lang', 'pt')),
        use_gpu=bool(paddle_cfg.get('use_gpu', True)),
        use_angle_cls=bool(paddle_cfg.get('use_angle_cls', True)),
        det_limit_side_len=int(paddle_cfg.get('det_limit_side_len', 960)),
        rec_batch_num=int(paddle_cfg.get('rec_batch_num', 6)),
        min_score=float(paddle_cfg.get('min_score', 0.3)),
    )

    premium = PremiumConfig(
        enabled=bool(premium_cfg.get('enabled', False)),
        provider=str(premium_cfg.get('provider', 'platerecognizer')),
        api_key=str(
            premium_cfg.get('api_key')
            or os.getenv('PLATE_RECOGNIZER_API_KEY', '')
        ),
        regions=list(premium_cfg.get('regions', ['br']) or ['br']),
        min_confidence=float(premium_cfg.get('min_confidence', 0.5)),
        timeout=int(premium_cfg.get('timeout', 30)),
        log_all_calls=bool(premium_cfg.get('log_all_calls', True)),
    )

    artifacts = ArtifactConfig(
        enabled=bool(artifacts_cfg.get('enabled', True)),
        output_dir=str(artifacts_cfg.get('output_dir', 'data/results/artifacts')),
        save_invalid=bool(artifacts_cfg.get('save_invalid', True)),
        save_low_confidence=bool(artifacts_cfg.get('save_low_confidence', True)),
        confidence_threshold=float(artifacts_cfg.get('confidence_threshold', 0.75)),
        max_saved_per_run=max(1, int(artifacts_cfg.get('max_saved_per_run', 50))),
    )

    scenarios = ScenarioConfig(
        low_light=LowLightScenarioConfig(
            brightness_threshold=float(low_light_cfg.get('brightness_threshold', 90.0)),
            contrast_threshold=float(low_light_cfg.get('contrast_threshold', 40.0)),
            ocr_confidence_threshold=float(
                low_light_cfg.get('ocr_confidence_threshold', pipeline_cfg.get('ocr_confidence_threshold', 0.6))
            ),
            fallback_confidence_threshold=float(
                low_light_cfg.get(
                    'fallback_confidence_threshold',
                    pipeline_cfg.get('fallback_confidence_threshold', 0.8),
                )
            ),
            max_variants=max(1, int(low_light_cfg.get('max_variants', ocr_cfg.get('max_variants', 5)))),
        ),
        small_plate=SmallPlateScenarioConfig(
            area_ratio_threshold=float(small_plate_cfg.get('area_ratio_threshold', 0.015)),
            height_threshold=max(1, int(small_plate_cfg.get('height_threshold', 60))),
            ocr_confidence_threshold=float(
                small_plate_cfg.get('ocr_confidence_threshold', pipeline_cfg.get('ocr_confidence_threshold', 0.6))
            ),
            fallback_confidence_threshold=float(
                small_plate_cfg.get(
                    'fallback_confidence_threshold',
                    pipeline_cfg.get('fallback_confidence_threshold', 0.8),
                )
            ),
            max_variants=max(1, int(small_plate_cfg.get('max_variants', ocr_cfg.get('max_variants', 5)))),
        ),
        moving_video=VideoScenarioConfig(
            skip_frames=max(1, int(moving_video_cfg.get('skip_frames', 2))),
            temporal_min_observations=max(
                1,
                int(moving_video_cfg.get('temporal_min_observations', 2)),
            ),
        ),
        stationary_video=VideoScenarioConfig(
            skip_frames=max(1, int(stationary_video_cfg.get('skip_frames', 5))),
            temporal_min_observations=max(
                1,
                int(stationary_video_cfg.get('temporal_min_observations', 3)),
            ),
        ),
    )

    vehicle_mode = VehicleMode.MOVING
    if str(video_cfg.get('vehicle_mode', '')).lower() == VehicleMode.STATIONARY.value:
        vehicle_mode = VehicleMode.STATIONARY

    scenario_video = (
        scenarios.stationary_video
        if vehicle_mode == VehicleMode.STATIONARY
        else scenarios.moving_video
    )

    skip_frames = video_cfg.get('skip_frames', scenario_video.skip_frames)
    if skip_frames in (None, 0):
        skip_frames = scenario_video.skip_frames

    temporal_cfg = raw_config.get('temporal_voting', {})
    temporal_min_observations = temporal_cfg.get(
        'min_observations',
        scenario_video.temporal_min_observations,
    )
    if temporal_min_observations in (None, 0):
        temporal_min_observations = scenario_video.temporal_min_observations

    video = VideoConfig(
        vehicle_mode=vehicle_mode,
        skip_frames=max(1, int(skip_frames)),
        max_frames=max(0, int(video_cfg.get('max_frames', 0))),
        generate_output_video=bool(video_cfg.get('generate_output_video', True)),
        output_dir=str(video_cfg.get('output_dir', 'data/results')),
        confidence_threshold=float(
            video_cfg.get('confidence_threshold', pipeline_cfg.get('ocr_confidence_threshold', 0.6))
        ),
        enable_temporal_voting=bool(temporal_cfg.get('enabled', True)),
        temporal_strategy=str(temporal_cfg.get('strategy', 'hybrid')),
        temporal_min_observations=max(1, int(temporal_min_observations)),
    )

    evaluation = EvaluationConfig(
        fixtures_dir=str(evaluation_cfg.get('fixtures_dir', 'data/fixtures')),
        manifest_path=str(evaluation_cfg.get('manifest_path', 'data/fixtures/manifest.template.json')),
        reports_dir=str(evaluation_cfg.get('reports_dir', 'data/results/evaluation')),
    )

    calibration = CalibrationConfig(
        detector_thresholds=[
            float(value)
            for value in calibration_cfg.get('detector_thresholds', [0.20, 0.25, 0.30])
        ],
        ocr_thresholds=[
            float(value)
            for value in calibration_cfg.get('ocr_thresholds', [0.55, 0.60, 0.65])
        ],
        fallback_thresholds=[
            float(value)
            for value in calibration_cfg.get('fallback_thresholds', [0.70, 0.80, 0.90])
        ],
    )

    quality = QualityConfig(
        enabled=bool(quality_cfg.get('enabled', True)),
        low_quality_threshold=float(quality_cfg.get('low_quality_threshold', 0.45)),
        snr_review_threshold=float(quality_cfg.get('snr_review_threshold', 9.0)),
        motion_blur_review_threshold=float(quality_cfg.get('motion_blur_review_threshold', 0.35)),
    )

    forensic = ForensicConfig(
        enabled=bool(forensic_cfg.get('enabled', True)),
        jpeg_quality=max(50, min(100, int(forensic_cfg.get('jpeg_quality', 90)))),
        review_threshold=float(forensic_cfg.get('review_threshold', 0.55)),
        high_risk_threshold=float(forensic_cfg.get('high_risk_threshold', 0.75)),
    )

    reports = ReportConfig(
        enabled=bool(reports_cfg.get('enabled', True)),
        output_dir=str(reports_cfg.get('output_dir', 'data/results/reports')),
        prefer_artifact_dir=bool(reports_cfg.get('prefer_artifact_dir', True)),
    )

    llm_validation = LLMValidationConfig(
        enabled=bool(llm_cfg.get('enabled', False)),
        base_url=str(llm_cfg.get('base_url', 'http://127.0.0.1:11434')),
        model=str(llm_cfg.get('model', '')),
        timeout=float(llm_cfg.get('timeout', 20.0)),
        allow_override=bool(llm_cfg.get('allow_override', True)),
        ambiguity_gap_threshold=float(llm_cfg.get('ambiguity_gap_threshold', 0.12)),
        min_decision_confidence=float(llm_cfg.get('min_decision_confidence', 0.70)),
    )

    return AppConfig(
        detector=detector,
        ocr=ocr,
        premium=premium,
        video=video,
        artifacts=artifacts,
        scenarios=scenarios,
        evaluation=evaluation,
        calibration=calibration,
        quality=quality,
        forensic=forensic,
        reports=reports,
        llm_validation=llm_validation,
    )