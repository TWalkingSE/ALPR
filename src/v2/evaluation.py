"""Fixture-based evaluation and threshold calibration utilities for ALPR v2."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

from src.v2.models import LocalPlateResult, normalize_plate_text


@dataclass(frozen=True)
class FixtureEntry:
    """Expected reference for a labeled media sample."""

    fixture_id: str
    path: str
    media_type: str = 'image'
    expected_plate: str = ''
    expected_format: str = ''
    scenario_tags: List[str] = field(default_factory=list)
    notes: str = ''


@dataclass(frozen=True)
class PredictionRecord:
    """Normalized prediction row used by evaluation and calibration."""

    fixture_id: str
    expected_plate: str
    predicted_plate: str
    confidence: float
    detection_confidence: float = 0.0
    processing_time_ms: float = 0.0
    scenario_tags: List[str] = field(default_factory=list)
    exact_match: bool = False
    char_accuracy: float = 0.0
    false_positive: bool = False


@dataclass
class EvaluationSummary:
    """Aggregate metrics for a labeled fixture run."""

    fixture_count: int
    exact_match_rate: float
    char_accuracy: float
    false_positive_rate: float
    avg_confidence: float
    avg_detection_confidence: float
    avg_processing_time_ms: float
    scenario_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)
    rows: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ThresholdCandidate:
    """Threshold triplet evaluated during calibration."""

    detector_confidence: float
    ocr_confidence_threshold: float
    fallback_confidence_threshold: float


@dataclass
class CalibrationResult:
    """Best threshold combination and leaderboard of tested candidates."""

    best_candidate: ThresholdCandidate
    best_score: float
    leaderboard: List[Dict[str, Any]] = field(default_factory=list)


def load_fixture_manifest(manifest_path: str | Path) -> List[FixtureEntry]:
    """Load a JSON fixture manifest into strongly typed entries."""
    manifest_file = Path(manifest_path)
    payload = json.loads(manifest_file.read_text(encoding='utf-8'))
    fixtures = payload.get('fixtures', [])

    return [
        FixtureEntry(
            fixture_id=str(item.get('id') or item.get('fixture_id') or Path(item['path']).stem),
            path=str(item['path']),
            media_type=str(item.get('media_type', 'image')),
            expected_plate=normalize_plate_text(str(item.get('expected_plate', ''))),
            expected_format=str(item.get('expected_format', '')),
            scenario_tags=[str(tag) for tag in item.get('scenario_tags', [])],
            notes=str(item.get('notes', '')),
        )
        for item in fixtures
    ]


def build_prediction_record(
    fixture: FixtureEntry,
    result: LocalPlateResult | None,
) -> PredictionRecord:
    """Normalize a single pipeline result into an evaluation row."""
    predicted_plate = normalize_plate_text(
        result.normalized_text if result is not None and result.normalized_text else (result.plate_text if result is not None else '')
    )
    expected_plate = normalize_plate_text(fixture.expected_plate)
    exact_match = bool(expected_plate and expected_plate == predicted_plate)
    false_positive = bool(not expected_plate and predicted_plate)
    char_accuracy = _compute_char_accuracy(expected_plate, predicted_plate)

    return PredictionRecord(
        fixture_id=fixture.fixture_id,
        expected_plate=expected_plate,
        predicted_plate=predicted_plate,
        confidence=float(result.confidence if result is not None else 0.0),
        detection_confidence=float(result.detection_confidence if result is not None else 0.0),
        processing_time_ms=float(result.processing_time_ms if result is not None else 0.0),
        scenario_tags=list(fixture.scenario_tags),
        exact_match=exact_match,
        char_accuracy=char_accuracy,
        false_positive=false_positive,
    )


def evaluate_prediction_records(records: Sequence[PredictionRecord]) -> EvaluationSummary:
    """Aggregate fixture-level predictions into baseline metrics."""
    rows = [asdict(record) for record in records]
    if not records:
        return EvaluationSummary(
            fixture_count=0,
            exact_match_rate=0.0,
            char_accuracy=0.0,
            false_positive_rate=0.0,
            avg_confidence=0.0,
            avg_detection_confidence=0.0,
            avg_processing_time_ms=0.0,
            scenario_breakdown={},
            rows=rows,
        )

    fixture_count = len(records)
    exact_match_rate = sum(1 for record in records if record.exact_match) / fixture_count
    char_accuracy = sum(record.char_accuracy for record in records) / fixture_count
    false_positive_rate = sum(1 for record in records if record.false_positive) / fixture_count
    avg_confidence = sum(record.confidence for record in records) / fixture_count
    avg_detection_confidence = sum(record.detection_confidence for record in records) / fixture_count
    avg_processing_time_ms = sum(record.processing_time_ms for record in records) / fixture_count

    scenario_breakdown: Dict[str, Dict[str, float]] = {}
    for tag in sorted({tag for record in records for tag in record.scenario_tags}):
        tagged = [record for record in records if tag in record.scenario_tags]
        if not tagged:
            continue

        count = len(tagged)
        scenario_breakdown[tag] = {
            'count': float(count),
            'exact_match_rate': sum(1 for record in tagged if record.exact_match) / count,
            'char_accuracy': sum(record.char_accuracy for record in tagged) / count,
            'avg_confidence': sum(record.confidence for record in tagged) / count,
            'avg_processing_time_ms': sum(record.processing_time_ms for record in tagged) / count,
        }

    return EvaluationSummary(
        fixture_count=fixture_count,
        exact_match_rate=exact_match_rate,
        char_accuracy=char_accuracy,
        false_positive_rate=false_positive_rate,
        avg_confidence=avg_confidence,
        avg_detection_confidence=avg_detection_confidence,
        avg_processing_time_ms=avg_processing_time_ms,
        scenario_breakdown=scenario_breakdown,
        rows=rows,
    )


def write_evaluation_report(
    summary: EvaluationSummary,
    output_dir: str | Path,
    report_name: str = 'baseline',
) -> Dict[str, str]:
    """Persist JSON and CSV reports for a baseline run."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / f'{report_name}.json'
    csv_path = output_path / f'{report_name}.csv'

    json_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=True, indent=2),
        encoding='utf-8',
    )

    fieldnames = [
        'fixture_id',
        'expected_plate',
        'predicted_plate',
        'confidence',
        'detection_confidence',
        'processing_time_ms',
        'exact_match',
        'char_accuracy',
        'false_positive',
        'scenario_tags',
    ]
    with csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary.rows:
            csv_row = dict(row)
            csv_row['scenario_tags'] = ','.join(csv_row.get('scenario_tags', []))
            writer.writerow(csv_row)

    return {
        'json': str(json_path),
        'csv': str(csv_path),
    }


def calibrate_thresholds(
    fixtures: Sequence[FixtureEntry],
    evaluate_candidate: Callable[[ThresholdCandidate, Sequence[FixtureEntry]], Sequence[PredictionRecord]],
    detector_thresholds: Iterable[float],
    ocr_thresholds: Iterable[float],
    fallback_thresholds: Iterable[float],
) -> CalibrationResult:
    """Grid-search threshold combinations using labeled fixtures."""
    leaderboard: List[Dict[str, Any]] = []

    for detector_confidence, ocr_threshold, fallback_threshold in product(
        detector_thresholds,
        ocr_thresholds,
        fallback_thresholds,
    ):
        candidate = ThresholdCandidate(
            detector_confidence=float(detector_confidence),
            ocr_confidence_threshold=float(ocr_threshold),
            fallback_confidence_threshold=float(fallback_threshold),
        )
        records = list(evaluate_candidate(candidate, fixtures))
        summary = evaluate_prediction_records(records)
        score = _calibration_score(summary)

        leaderboard.append({
            'candidate': asdict(candidate),
            'score': score,
            'exact_match_rate': summary.exact_match_rate,
            'char_accuracy': summary.char_accuracy,
            'false_positive_rate': summary.false_positive_rate,
            'avg_processing_time_ms': summary.avg_processing_time_ms,
        })

    leaderboard.sort(key=lambda item: item['score'], reverse=True)
    if not leaderboard:
        raise ValueError('Nenhum candidato de calibracao foi avaliado.')

    best = leaderboard[0]['candidate']
    return CalibrationResult(
        best_candidate=ThresholdCandidate(**best),
        best_score=float(leaderboard[0]['score']),
        leaderboard=leaderboard,
    )


def _compute_char_accuracy(expected_plate: str, predicted_plate: str) -> float:
    expected = normalize_plate_text(expected_plate)
    predicted = normalize_plate_text(predicted_plate)
    if not expected and not predicted:
        return 1.0

    max_len = max(len(expected), len(predicted))
    if max_len == 0:
        return 0.0

    matches = sum(
        1
        for index in range(min(len(expected), len(predicted)))
        if expected[index] == predicted[index]
    )
    return matches / max_len


def _calibration_score(summary: EvaluationSummary) -> float:
    precision = 1.0 - summary.false_positive_rate
    latency_score = max(0.0, 1.0 - min(summary.avg_processing_time_ms, 1500.0) / 1500.0)
    return (
        0.55 * summary.exact_match_rate
        + 0.20 * summary.char_accuracy
        + 0.15 * precision
        + 0.10 * latency_score
    )