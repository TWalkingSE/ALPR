"""Stable contracts for the ALPR v2 application layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from src.premium_alpr import PremiumALPRResult
from src.v2.models import LocalPlateResult


@runtime_checkable
class LocalAnalysisProvider(Protocol):
    """Contract for the local ALPR provider used by the app layer."""

    premium_provider: Any

    def process_image(
        self,
        image: np.ndarray,
        detector_confidence: Optional[float] = None,
        return_all_plates: bool = True,
        image_bytes: Optional[bytes] = None,
        input_file_path: Optional[str] = None,
        temporal_prior: Optional[dict[str, float]] = None,
    ) -> list[LocalPlateResult]:
        ...

    def process_video_frame(
        self,
        frame: np.ndarray,
        detector_confidence: Optional[float] = None,
        temporal_prior: Optional[dict[str, float]] = None,
    ) -> list[LocalPlateResult]:
        ...

    def get_pipeline_info(self) -> dict[str, Any]:
        ...


@runtime_checkable
class PremiumAnalysisProvider(Protocol):
    """Contract for the Premium API provider used by the app layer."""

    @property
    def available(self) -> bool:
        ...

    @property
    def provider(self) -> str:
        ...

    def analyze_full_image(self, image: np.ndarray) -> PremiumALPRResult:
        ...


@runtime_checkable
class VideoAnalyzer(Protocol):
    """Contract for video analysis used by the v2 app layer."""

    def process_video(
        self,
        video_path: str,
        pipeline: LocalAnalysisProvider,
        detector_confidence: Optional[float] = None,
        progress_callback: Any = None,
    ) -> Any:
        ...

    def rank_unique_plates(self, unique_plates: dict[str, Any]) -> dict[str, Any]:
        ...

    def generate_timeline(self, video_result: Any) -> list[dict[str, Any]]:
        ...

    def build_confirmed_reading(self, info: dict[str, Any]) -> str:
        ...


@runtime_checkable
class ResultPresenter(Protocol):
    """Contract for the v2 presentation layer."""

    def display_local_result(self, result: LocalPlateResult, col) -> None:
        ...

    def display_summary_table(self, results: Sequence[LocalPlateResult]) -> None:
        ...

    def display_premium_api_comparison(
        self,
        local_results: Sequence[LocalPlateResult],
        premium_result: PremiumALPRResult,
        premium_time_ms: float = 0.0,
    ) -> None:
        ...

    def display_video_results(self, video_result: Any, video_processor: VideoAnalyzer) -> None:
        ...


@dataclass(frozen=True)
class ServiceBundle:
    """Runtime services used by the ALPR v2 app."""

    pipeline: LocalAnalysisProvider
    premium: PremiumAnalysisProvider