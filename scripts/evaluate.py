"""Baseline de acurácia do pipeline local sobre fixtures rotuladas.

Executa o pipeline real (YOLO + PaddleOCR + validação) contra o manifesto de
fixtures e grava relatórios JSON/CSV com exact match, acurácia por caractere,
taxa de falso positivo e latência — inclusive quebrados por cenário.

Uso:
    python scripts/evaluate.py --manifest data/fixtures/manifest.json
    python scripts/evaluate.py --report-name antes-do-ajuste
    python scripts/evaluate.py --compare data/results/evaluation/baseline.json
    python scripts/evaluate.py --fail-under 0.90        # para uso em CI
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401  (ajusta o sys.path)
from _fixtures import run_fixtures

from src.config_manager import load_config
from src.v2.config import build_v2_config
from src.v2.evaluation import (
    EvaluationSummary,
    evaluate_prediction_records,
    load_fixture_manifest,
    write_evaluation_report,
)
from src.v2.pipeline import LocalAnalysisPipeline

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Gera a baseline de acurácia do ALPR sobre fixtures rotuladas.',
    )
    parser.add_argument(
        '--manifest',
        default=None,
        help='Manifesto de fixtures (padrão: evaluation.manifest_path do config.yaml).',
    )
    parser.add_argument(
        '--report-name',
        default='baseline',
        help='Nome base dos relatórios gerados (padrão: baseline).',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Diretório dos relatórios (padrão: evaluation.reports_dir do config.yaml).',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=0,
        help='Processar no máximo N fixtures (0 = todas).',
    )
    parser.add_argument(
        '--compare',
        default=None,
        help='Relatório JSON anterior para exibir o delta de cada métrica.',
    )
    parser.add_argument(
        '--fail-under',
        type=float,
        default=None,
        help='Sai com código 1 se o exact match ficar abaixo deste valor (0-1).',
    )
    parser.add_argument(
        '--model',
        default=None,
        help='Caminho do peso YOLO (padrão: models.detector.dir + default do config.yaml).',
    )
    parser.add_argument('--verbose', action='store_true', help='Log em nível DEBUG.')
    return parser.parse_args(argv)


def _format_delta(current: float, previous: float | None, as_percent: bool = True) -> str:
    if previous is None:
        return ''
    delta = current - previous
    unit = 'pp' if as_percent else ''
    scale = 100.0 if as_percent else 1.0
    sign = '+' if delta >= 0 else ''
    return f'  ({sign}{delta * scale:.1f}{unit} vs anterior)'


def print_summary(summary: EvaluationSummary, previous: dict[str, Any] | None = None) -> None:
    prev = previous or {}
    print()
    print('=' * 62)
    print(f'  Baseline — {summary.fixture_count} fixtures')
    print('=' * 62)
    print(
        f'  Exact match .......... {summary.exact_match_rate:6.1%}'
        + _format_delta(summary.exact_match_rate, prev.get('exact_match_rate'))
    )
    print(
        f'  Acurácia por char .... {summary.char_accuracy:6.1%}'
        + _format_delta(summary.char_accuracy, prev.get('char_accuracy'))
    )
    print(
        f'  Falso positivo ....... {summary.false_positive_rate:6.1%}'
        + _format_delta(summary.false_positive_rate, prev.get('false_positive_rate'))
    )
    print(f'  Confiança OCR média .. {summary.avg_confidence:6.1%}')
    print(f'  Confiança det. média . {summary.avg_detection_confidence:6.1%}')
    print(
        f'  Latência média ....... {summary.avg_processing_time_ms:6.1f} ms'
        + _format_delta(
            summary.avg_processing_time_ms,
            prev.get('avg_processing_time_ms'),
            as_percent=False,
        )
    )

    if summary.scenario_breakdown:
        print()
        print('  Por cenário:')
        for tag, metrics in sorted(summary.scenario_breakdown.items()):
            print(
                f'    {tag:<16} n={int(metrics["count"]):<4} '
                f'exact={metrics["exact_match_rate"]:5.1%}  '
                f'char={metrics["char_accuracy"]:5.1%}  '
                f'{metrics["avg_processing_time_ms"]:6.1f} ms'
            )

    misses = [
        row for row in summary.rows
        if not row.get('exact_match') and row.get('expected_plate')
    ]
    if misses:
        print()
        print(f'  Divergências ({len(misses)}):')
        for row in misses[:20]:
            print(
                f'    {row["fixture_id"]:<28} esperado={row["expected_plate"]:<9} '
                f'lido={row["predicted_plate"] or "-":<9} conf={row["confidence"]:.0%}'
            )
        if len(misses) > 20:
            print(f'    ... e mais {len(misses) - 20}')
    print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format='%(levelname)s %(name)s: %(message)s',
    )

    config = build_v2_config(load_config())

    manifest_path = Path(args.manifest or config.evaluation.manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_DIR / manifest_path
    if not manifest_path.exists():
        print(f'ERRO: manifesto não encontrado: {manifest_path}', file=sys.stderr)
        print(
            'Crie um a partir de data/fixtures/manifest.template.json '
            '(ver data/fixtures/README.md).',
            file=sys.stderr,
        )
        return 2

    fixtures = load_fixture_manifest(manifest_path)
    if args.limit > 0:
        fixtures = fixtures[: args.limit]
    if not fixtures:
        print(f'ERRO: nenhum fixture em {manifest_path}', file=sys.stderr)
        return 2

    model_path = args.model or str(
        PROJECT_DIR / config.detector.models_dir / config.detector.model_name
    )
    print(f'Modelo YOLO: {model_path}')
    print(f'Fixtures:    {len(fixtures)} de {manifest_path}')

    try:
        pipeline = LocalAnalysisPipeline.from_settings(config, PROJECT_DIR, model_path=model_path)
    except Exception as exc:
        print(f'ERRO ao inicializar o pipeline: {exc}', file=sys.stderr)
        return 2

    def _progress(index: int, total: int, fixture) -> None:
        print(f'  [{index}/{total}] {fixture.fixture_id}', flush=True)

    records = run_fixtures(pipeline, config, fixtures, manifest_path, on_progress=_progress)
    summary = evaluate_prediction_records(records)

    previous = None
    if args.compare:
        compare_path = Path(args.compare)
        if compare_path.exists():
            previous = json.loads(compare_path.read_text(encoding='utf-8'))
        else:
            print(f'AVISO: relatório de comparação não encontrado: {compare_path}', file=sys.stderr)

    print_summary(summary, previous)

    output_dir = Path(args.output_dir or config.evaluation.reports_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    paths = write_evaluation_report(summary, output_dir, report_name=args.report_name)
    print(f'Relatórios: {paths["json"]}')
    print(f'            {paths["csv"]}')

    if args.fail_under is not None and summary.exact_match_rate < args.fail_under:
        print(
            f'FALHA: exact match {summary.exact_match_rate:.1%} '
            f'abaixo do mínimo exigido {args.fail_under:.1%}',
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
