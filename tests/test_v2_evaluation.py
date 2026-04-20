from __future__ import annotations

import json

from src.v2.config import AppConfig
from src.v2.evaluation import (
    FixtureEntry,
    build_prediction_record,
    calibrate_thresholds,
    evaluate_prediction_records,
    load_fixture_manifest,
    write_evaluation_report,
)
from src.v2.models import LocalPlateResult


def _result(plate_text: str, confidence: float = 0.9) -> LocalPlateResult:
    return LocalPlateResult(
        plate_text=plate_text,
        confidence=confidence,
        detection_confidence=0.85,
        format_type='mercosul',
        is_valid=True,
        original_crop=None,
        bbox=(0, 0, 1, 1),
        normalized_text=plate_text,
        processing_time_ms=45.0,
    )


def test_load_fixture_manifest_reads_entries(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(
        json.dumps({
            'version': 1,
            'fixtures': [
                {
                    'id': 'sample-1',
                    'path': 'images/sample-1.jpg',
                    'expected_plate': 'ABC1D23',
                    'scenario_tags': ['low_light'],
                }
            ],
        }),
        encoding='utf-8',
    )

    fixtures = load_fixture_manifest(manifest)

    assert len(fixtures) == 1
    assert fixtures[0].fixture_id == 'sample-1'
    assert fixtures[0].expected_plate == 'ABC1D23'


def test_evaluate_prediction_records_and_write_report(tmp_path):
    fixture = FixtureEntry(
        fixture_id='sample-1',
        path='images/sample-1.jpg',
        expected_plate='ABC1D23',
        scenario_tags=['low_light'],
    )
    record = build_prediction_record(fixture, _result('ABC1D23'))

    summary = evaluate_prediction_records([record])
    report_paths = write_evaluation_report(summary, tmp_path, report_name='smoke')

    assert summary.exact_match_rate == 1.0
    assert summary.char_accuracy == 1.0
    assert 'low_light' in summary.scenario_breakdown
    assert tmp_path.joinpath('smoke.json').exists()
    assert tmp_path.joinpath('smoke.csv').exists()
    assert set(report_paths) == {'json', 'csv'}


def test_calibrate_thresholds_selects_best_candidate():
    fixtures = [FixtureEntry(fixture_id='sample-1', path='images/sample-1.jpg', expected_plate='ABC1D23')]

    def _evaluate(candidate, _fixtures):
        confidence = 0.95 if candidate.detector_confidence == 0.2 else 0.60
        predicted = 'ABC1D23' if candidate.detector_confidence == 0.2 else 'ABC1023'
        return [
            build_prediction_record(
                fixtures[0],
                _result(predicted, confidence=confidence),
            )
        ]

    result = calibrate_thresholds(
        fixtures,
        _evaluate,
        detector_thresholds=[0.2, 0.3],
        ocr_thresholds=[0.55],
        fallback_thresholds=[0.8],
    )

    assert result.best_candidate.detector_confidence == 0.2
    assert result.best_score > 0.0