"""Domain models for ALPR v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class LocalPlateResult:
    """Result of a single local ALPR recognition in v2."""

    plate_text: str
    confidence: float
    detection_confidence: float
    format_type: str
    is_valid: bool
    original_crop: np.ndarray
    bbox: Tuple[int, int, int, int]
    normalized_crop: Optional[np.ndarray] = None
    preprocessed_image: Optional[np.ndarray] = None
    ocr_engine: str = ''
    char_confidences: List[Tuple[str, float]] = field(default_factory=list)
    alternative_plates: List[Dict[str, Any]] = field(default_factory=list)
    processing_time_ms: float = 0.0
    pipeline_steps_time: Dict[str, float] = field(default_factory=dict)
    raw_ocr_text: str = ''
    normalized_text: str = ''
    warnings: List[str] = field(default_factory=list)
    scenario_tags: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    quality_assessment: Dict[str, Any] = field(default_factory=dict)
    validation_details: Dict[str, Any] = field(default_factory=dict)
    llm_validation: Dict[str, Any] = field(default_factory=dict)
    forensic_analysis: Dict[str, Any] = field(default_factory=dict)
    detector_metadata: Dict[str, Any] = field(default_factory=dict)
    artifact_dir: str = ''
    artifact_files: Dict[str, str] = field(default_factory=dict)
    report_path: str = ''
    report_payload: Dict[str, Any] = field(default_factory=dict)


def normalize_plate_text(text: str) -> str:
    """Return an uppercase alphanumeric-only plate string."""
    return ''.join(char for char in text.upper() if char.isalnum())