"""ALPR v2 package.

Parallel implementation focused on the stable local pipeline plus the
existing Plate Recognizer premium path.
"""

from src.v2.application import (
    build_service_bundle,
    build_video_processor,
    decode_uploaded_image,
    ensure_service_bundle,
    run_local_image_analysis,
    run_premium_image_analysis,
    run_video_analysis,
)
from src.v2.config import AppConfig, build_v2_config
from src.v2.contracts import (
    LocalAnalysisProvider,
    PremiumAnalysisProvider,
    ResultPresenter,
    ServiceBundle,
    VideoAnalyzer,
)
from src.v2.evaluation import (
    CalibrationResult,
    EvaluationSummary,
    FixtureEntry,
    PredictionRecord,
    ThresholdCandidate,
    build_prediction_record,
    calibrate_thresholds,
    evaluate_prediction_records,
    load_fixture_manifest,
    write_evaluation_report,
)
from src.v2.models import LocalPlateResult
from src.v2.pipeline import LocalAnalysisPipeline
from src.v2.premium import PremiumAnalysisService
from src.v2.state import AppSessionState, ImageAnalysisState, read_app_state

__all__ = [
    'AppConfig',
    'AppSessionState',
    'CalibrationResult',
    'EvaluationSummary',
    'FixtureEntry',
    'LocalAnalysisPipeline',
    'LocalAnalysisProvider',
    'LocalPlateResult',
    'PredictionRecord',
    'ImageAnalysisState',
    'PremiumAnalysisService',
    'PremiumAnalysisProvider',
    'ResultPresenter',
    'ServiceBundle',
    'ThresholdCandidate',
    'VideoAnalyzer',
    'build_prediction_record',
    'build_service_bundle',
    'build_v2_config',
    'build_video_processor',
    'calibrate_thresholds',
    'decode_uploaded_image',
    'ensure_service_bundle',
    'evaluate_prediction_records',
    'load_fixture_manifest',
    'read_app_state',
    'run_local_image_analysis',
    'run_premium_image_analysis',
    'run_video_analysis',
    'write_evaluation_report',
]