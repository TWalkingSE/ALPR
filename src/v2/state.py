"""Session state helpers for the ALPR v2 Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, MutableMapping, Optional

from src.v2.contracts import ServiceBundle
from src.v2.models import LocalPlateResult

SESSION_PIPELINE_KEY = 'v2_pipeline'
SESSION_PREMIUM_KEY = 'v2_premium'
SESSION_SIGNATURE_KEY = 'v2_signature'
SESSION_LOCAL_RESULTS_KEY = 'v2_local_results'
SESSION_LOCAL_IMAGE_NAME_KEY = 'v2_local_image_name'
SESSION_PREMIUM_RESULT_KEY = 'v2_premium_result'
SESSION_PREMIUM_TIME_MS_KEY = 'v2_premium_time_ms'
SESSION_VIDEO_RESULT_KEY = 'v2_video_result'


@dataclass
class ImageAnalysisState:
    """Image-related state persisted in the Streamlit session."""

    local_results: list[LocalPlateResult] = field(default_factory=list)
    local_image_name: str = ''
    premium_result: Any = None
    premium_time_ms: float = 0.0


@dataclass
class AppSessionState:
    """Typed view over the Streamlit session state used by v2."""

    pipeline: Any = None
    premium_service: Any = None
    signature: Optional[tuple] = None
    image: ImageAnalysisState = field(default_factory=ImageAnalysisState)
    video_result: Any = None


def read_app_state(session: MutableMapping[str, Any]) -> AppSessionState:
    """Read the typed v2 state from a Streamlit-compatible session mapping."""
    return AppSessionState(
        pipeline=session.get(SESSION_PIPELINE_KEY),
        premium_service=session.get(SESSION_PREMIUM_KEY),
        signature=session.get(SESSION_SIGNATURE_KEY),
        image=ImageAnalysisState(
            local_results=list(session.get(SESSION_LOCAL_RESULTS_KEY, [])),
            local_image_name=str(session.get(SESSION_LOCAL_IMAGE_NAME_KEY, '')),
            premium_result=session.get(SESSION_PREMIUM_RESULT_KEY),
            premium_time_ms=float(session.get(SESSION_PREMIUM_TIME_MS_KEY, 0.0)),
        ),
        video_result=session.get(SESSION_VIDEO_RESULT_KEY),
    )


def store_service_bundle(
    session: MutableMapping[str, Any],
    bundle: ServiceBundle,
    signature: tuple,
) -> None:
    """Persist the active service bundle in session state."""
    session[SESSION_PIPELINE_KEY] = bundle.pipeline
    session[SESSION_PREMIUM_KEY] = bundle.premium
    session[SESSION_SIGNATURE_KEY] = signature


def store_local_results(
    session: MutableMapping[str, Any],
    results: list[LocalPlateResult],
    image_name: str,
) -> None:
    """Persist local image analysis results."""
    session[SESSION_LOCAL_RESULTS_KEY] = list(results)
    session[SESSION_LOCAL_IMAGE_NAME_KEY] = image_name


def store_premium_result(
    session: MutableMapping[str, Any],
    premium_result: Any,
    elapsed_ms: float,
) -> None:
    """Persist Premium API results."""
    session[SESSION_PREMIUM_RESULT_KEY] = premium_result
    session[SESSION_PREMIUM_TIME_MS_KEY] = float(elapsed_ms)


def store_video_result(session: MutableMapping[str, Any], video_result: Any) -> None:
    """Persist the latest video analysis result."""
    session[SESSION_VIDEO_RESULT_KEY] = video_result