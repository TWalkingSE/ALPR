"""Application-layer helpers for the ALPR v2 Streamlit app."""

from __future__ import annotations

import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, MutableMapping, Optional

import cv2
import numpy as np

from src.video_processor import VideoProcessor
from src.v2.config import AppConfig
from src.v2.contracts import (
    LocalAnalysisProvider,
    PremiumAnalysisProvider,
    ServiceBundle,
    VideoAnalyzer,
)
from src.v2.models import LocalPlateResult
from src.v2.pipeline import LocalAnalysisPipeline
from src.v2.premium import PremiumAnalysisService
from src.v2.state import read_app_state, store_service_bundle

ProgressCallback = Callable[[int, int, Any], None]
ServiceBuilder = Callable[[AppConfig, Path, str], ServiceBundle]
VideoProcessorFactory = Callable[[AppConfig], VideoAnalyzer]


def decode_uploaded_image(uploaded_file) -> tuple[Optional[np.ndarray], bytes]:
    """Decode an uploaded image from Streamlit into BGR bytes and pixels."""
    image_bytes = uploaded_file.getvalue()
    if not image_bytes:
        return None, b''
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image, image_bytes


def build_service_bundle(config: AppConfig, project_dir: Path, model_path: str) -> ServiceBundle:
    """Build the runtime services used by the v2 app."""
    pipeline = LocalAnalysisPipeline.from_settings(config, project_dir, model_path=model_path)
    premium = PremiumAnalysisService.from_settings(config.premium)
    pipeline.premium_provider = premium
    return ServiceBundle(pipeline=pipeline, premium=premium)


def ensure_service_bundle(
    session: MutableMapping[str, Any],
    config: AppConfig,
    project_dir: Path,
    model_path: str,
    force_rebuild: bool = False,
    builder: ServiceBuilder = build_service_bundle,
) -> ServiceBundle:
    """Reuse or rebuild the active services according to the config signature."""
    signature = (model_path, config.signature())
    state = read_app_state(session)
    if force_rebuild or state.pipeline is None or state.signature != signature:
        bundle = builder(config, project_dir, model_path)
        store_service_bundle(session, bundle, signature)
        return bundle
    return ServiceBundle(pipeline=state.pipeline, premium=state.premium_service)


def run_local_image_analysis(
    pipeline: LocalAnalysisProvider,
    config: AppConfig,
    image: np.ndarray,
    image_bytes: bytes,
    image_name: str,
) -> list[LocalPlateResult]:
    """Execute the local image pipeline with the v2 defaults."""
    return pipeline.process_image(
        image,
        detector_confidence=config.detector.confidence,
        image_bytes=image_bytes,
        input_file_path=image_name,
    )


def run_premium_image_analysis(
    premium_service: PremiumAnalysisProvider,
    image: np.ndarray,
    perf_counter: Callable[[], float] = time.perf_counter,
):
    """Execute the Premium API analysis and measure elapsed time."""
    start_time = perf_counter()
    result = premium_service.analyze_full_image(image)
    elapsed_ms = (perf_counter() - start_time) * 1000
    return result, elapsed_ms


def build_video_processor(config: AppConfig) -> VideoAnalyzer:
    """Build the video processor configured for v2."""
    return VideoProcessor(
        skip_frames=config.video.skip_frames,
        max_frames=config.video.max_frames,
        generate_output_video=config.video.generate_output_video,
        output_dir=config.video.output_dir,
        confidence_threshold=config.video.confidence_threshold,
        vehicle_mode=config.video.vehicle_mode,
        enable_temporal_voting=config.video.enable_temporal_voting,
        temporal_voting_strategy=config.video.temporal_strategy,
        temporal_min_observations=config.video.temporal_min_observations,
    )


def run_video_analysis(
    video_bytes: bytes,
    filename: str,
    pipeline: LocalAnalysisProvider,
    config: AppConfig,
    progress_callback: Optional[ProgressCallback] = None,
    processor_factory: VideoProcessorFactory = build_video_processor,
):
    """Execute video analysis from uploaded bytes with temp-file lifecycle managed here."""
    suffix = Path(filename).suffix or '.mp4'
    temp_path = None
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name

        video_processor = processor_factory(config)
        return video_processor.process_video(
            temp_path,
            pipeline=pipeline,
            detector_confidence=config.detector.confidence,
            progress_callback=progress_callback,
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)