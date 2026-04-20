"""Structured report generation for ALPR local analyses."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.v2.models import LocalPlateResult, normalize_plate_text


class ReportBuilder:
    """Generate and persist structured ALPR reports."""

    def __init__(
        self,
        enabled: bool = True,
        output_dir: str = 'data/results/reports',
        prefer_artifact_dir: bool = True,
    ):
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir)
        self.prefer_artifact_dir = bool(prefer_artifact_dir)

    def generate(
        self,
        result: LocalPlateResult,
        image_bytes: Optional[bytes] = None,
        input_file_path: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str]:
        if not self.enabled:
            return {}, ''

        payload = self._build_payload(result, image_bytes=image_bytes, input_file_path=input_file_path)
        report_path = self._write_payload(payload, result)
        return payload, report_path

    def _build_payload(
        self,
        result: LocalPlateResult,
        image_bytes: Optional[bytes] = None,
        input_file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        source_hash = hashlib.sha256(
            image_bytes if image_bytes is not None else result.original_crop.tobytes()
        ).hexdigest()

        confidence_band = 'review'
        if result.confidence >= 0.85 and result.is_valid:
            confidence_band = 'high'
        elif result.confidence >= 0.65:
            confidence_band = 'medium'

        timestamp = datetime.now(timezone.utc).isoformat()
        return {
            'report_version': '1.0',
            'generated_at': timestamp,
            'source': {
                'input_file_path': input_file_path or '',
                'sha256': source_hash,
            },
            'recognition': {
                'plate_text': result.plate_text,
                'normalized_text': result.normalized_text,
                'raw_ocr_text': result.raw_ocr_text,
                'format_type': result.format_type,
                'is_valid': result.is_valid,
                'ocr_engine': result.ocr_engine,
                'ocr_confidence': result.confidence,
                'detection_confidence': result.detection_confidence,
                'confidence_band': confidence_band,
            },
            'scenario_tags': list(result.scenario_tags),
            'warnings': list(result.warnings),
            'quality': {
                'score': result.quality_score,
                'metrics': result.quality_metrics,
                'assessment': result.quality_assessment,
            },
            'validation': result.validation_details,
            'llm_validation': result.llm_validation,
            'forensic': result.forensic_analysis,
            'detector': result.detector_metadata,
            'alternatives': list(result.alternative_plates),
            'timing': {
                'processing_time_ms': result.processing_time_ms,
                'steps': result.pipeline_steps_time,
            },
            'artifacts': {
                'artifact_dir': result.artifact_dir,
                'files': result.artifact_files,
            },
            'bbox': list(result.bbox),
            'char_confidences': [[char, score] for char, score in result.char_confidences],
        }

    def _write_payload(self, payload: Dict[str, Any], result: LocalPlateResult) -> str:
        preferred_dir = Path(result.artifact_dir) if result.artifact_dir else None
        if self.prefer_artifact_dir and preferred_dir is not None:
            output_dir = preferred_dir
        else:
            output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = payload.get('generated_at', '').replace(':', '').replace('-', '').replace('+00:00', 'Z')
        label = normalize_plate_text(result.plate_text or result.raw_ocr_text) or 'unknown'
        report_path = output_dir / f'{timestamp}_{label}_report.json'
        report_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')
        return str(report_path)