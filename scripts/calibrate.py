"""Grid-search dos thresholds do pipeline sobre fixtures rotuladas.

Varre as combinações declaradas em ``calibration:`` no config.yaml (confiança
do detector × limiar de OCR × limiar de fallback), pontua cada uma com as
mesmas métricas da baseline e imprime o leaderboard.

O pipeline é construído UMA vez: entre combinações apenas os thresholds são
reaplicados via ``LocalAnalysisPipeline.apply_runtime_config``, sem recarregar
o YOLO nem o PaddleOCR — recarregar por combinação tornaria a varredura
inviável.

Uso:
    python scripts/calibrate.py
    python scripts/calibrate.py --manifest data/fixtures/manifest.json --apply
"""

from __future__ import annotations

import argparse
import copy
import logging
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import _bootstrap  # noqa: F401  (ajusta o sys.path)
import yaml
from _fixtures import run_fixtures

from src.config_manager import load_config
from src.v2.config import build_v2_config
from src.v2.evaluation import (
    FixtureEntry,
    PredictionRecord,
    ThresholdCandidate,
    calibrate_thresholds,
    load_fixture_manifest,
)
from src.v2.pipeline import LocalAnalysisPipeline

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_DIR / 'config.yaml'


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Calibra os thresholds do ALPR por grid-search sobre fixtures.',
    )
    parser.add_argument('--manifest', default=None, help='Manifesto de fixtures.')
    parser.add_argument(
        '--limit', type=int, default=0, help='Processar no máximo N fixtures (0 = todas).'
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Grava a melhor combinação de volta no config.yaml.',
    )
    parser.add_argument(
        '--top', type=int, default=10, help='Quantas linhas do leaderboard exibir.'
    )
    parser.add_argument('--model', default=None, help='Caminho do peso YOLO.')
    parser.add_argument('--verbose', action='store_true', help='Log em nível DEBUG.')
    return parser.parse_args(argv)


def replace_scalar(text: str, key: str, value: float, section: str | None = None) -> tuple[str, bool]:
    """Substitui o valor de `key` no YAML preservando o resto do arquivo.

    Edição pontual por linha em vez de reserializar via `yaml.safe_dump`, que
    descartaria todos os comentários do `config.yaml` — e eles documentam o
    porquê de cada parâmetro.

    Quando `section` é informado, só a chave dentro daquela seção de topo é
    considerada, evitando casar com uma chave homônima em outra seção.

    Returns:
        (texto_atualizado, substituiu)
    """
    lines = text.splitlines(keepends=True)
    key_pattern = re.compile(rf'^(\s*){re.escape(key)}:(\s*)([^\s#]+)(.*)$')
    section_pattern = re.compile(rf'^{re.escape(section)}:\s*$') if section else None
    top_level_pattern = re.compile(r'^[A-Za-z_]')

    inside_section = section is None
    for index, line in enumerate(lines):
        if section_pattern is not None:
            if section_pattern.match(line):
                inside_section = True
                continue
            # Sair da seção ao encontrar a próxima chave de topo.
            if inside_section and top_level_pattern.match(line):
                inside_section = False

        if not inside_section:
            continue

        match = key_pattern.match(line)
        if match:
            indent, spacing, _old, trailing = match.groups()
            lines[index] = f'{indent}{key}:{spacing}{value}{trailing}\n'
            return ''.join(lines), True

    return text, False


def apply_to_config_yaml(candidate: ThresholdCandidate, config_path: Path) -> None:
    """Escreve a melhor combinação no config.yaml, preservando comentários."""
    original = config_path.read_text(encoding='utf-8')
    text = original

    updates = [
        ('confidence', round(candidate.detector_confidence, 4), 'models'),
        ('ocr_confidence_threshold', round(candidate.ocr_confidence_threshold, 4), 'pipeline'),
        (
            'fallback_confidence_threshold',
            round(candidate.fallback_confidence_threshold, 4),
            'pipeline',
        ),
    ]

    aplicados = []
    for key, value, section in updates:
        text, replaced = replace_scalar(text, key, value, section=section)
        if replaced:
            aplicados.append(f'{section}.{key}={value}')
        else:
            print(f'AVISO: chave {section}.{key} não encontrada no config.yaml', file=sys.stderr)

    # Só grava se o resultado ainda for YAML válido e com os valores esperados.
    try:
        reparsed = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        print(f'ERRO: a edição produziu YAML inválido, nada foi gravado: {exc}', file=sys.stderr)
        return

    if reparsed.get('pipeline', {}).get('ocr_confidence_threshold') != round(
        candidate.ocr_confidence_threshold, 4
    ):
        print('ERRO: verificação pós-edição falhou, nada foi gravado.', file=sys.stderr)
        return

    backup = config_path.with_suffix('.yaml.bak')
    backup.write_text(original, encoding='utf-8')
    config_path.write_text(text, encoding='utf-8')
    print(f'config.yaml atualizado ({", ".join(aplicados)}); backup em {backup.name}.')


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format='%(levelname)s %(name)s: %(message)s',
    )

    raw_config = load_config()
    config = build_v2_config(raw_config)

    manifest_path = Path(args.manifest or config.evaluation.manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_DIR / manifest_path
    if not manifest_path.exists():
        print(f'ERRO: manifesto não encontrado: {manifest_path}', file=sys.stderr)
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
    try:
        pipeline = LocalAnalysisPipeline.from_settings(config, PROJECT_DIR, model_path=model_path)
    except Exception as exc:
        print(f'ERRO ao inicializar o pipeline: {exc}', file=sys.stderr)
        return 2

    total = (
        len(config.calibration.detector_thresholds)
        * len(config.calibration.ocr_thresholds)
        * len(config.calibration.fallback_thresholds)
    )
    print(f'Fixtures: {len(fixtures)} | combinações: {total}')
    progress = {'done': 0}

    def evaluate_candidate(
        candidate: ThresholdCandidate,
        candidate_fixtures: Sequence[FixtureEntry],
    ) -> Sequence[PredictionRecord]:
        progress['done'] += 1
        print(
            f'  [{progress["done"]}/{total}] det={candidate.detector_confidence:.2f} '
            f'ocr={candidate.ocr_confidence_threshold:.2f} '
            f'fallback={candidate.fallback_confidence_threshold:.2f}',
            flush=True,
        )

        # Reaplica apenas os thresholds; os modelos carregados permanecem.
        trial_config = copy.deepcopy(config)
        trial_config.detector.confidence = candidate.detector_confidence
        trial_config.ocr.confidence_threshold = candidate.ocr_confidence_threshold
        trial_config.ocr.fallback_threshold = candidate.fallback_confidence_threshold
        pipeline.apply_runtime_config(trial_config)

        return run_fixtures(pipeline, trial_config, candidate_fixtures, manifest_path)

    result = calibrate_thresholds(
        fixtures,
        evaluate_candidate,
        config.calibration.detector_thresholds,
        config.calibration.ocr_thresholds,
        config.calibration.fallback_thresholds,
    )

    print()
    print('=' * 78)
    print('  Leaderboard')
    print('=' * 78)
    print(f'  {"det":>5} {"ocr":>5} {"fbk":>5} {"score":>7} {"exact":>7} {"char":>7} {"ms":>8}')
    for entry in result.leaderboard[: args.top]:
        candidate = entry['candidate']
        print(
            f'  {candidate["detector_confidence"]:5.2f} '
            f'{candidate["ocr_confidence_threshold"]:5.2f} '
            f'{candidate["fallback_confidence_threshold"]:5.2f} '
            f'{entry["score"]:7.3f} {entry["exact_match_rate"]:7.1%} '
            f'{entry["char_accuracy"]:7.1%} {entry["avg_processing_time_ms"]:8.1f}'
        )

    best = result.best_candidate
    print()
    print(f'  Melhor: det={best.detector_confidence:.2f} '
          f'ocr={best.ocr_confidence_threshold:.2f} '
          f'fallback={best.fallback_confidence_threshold:.2f} '
          f'(score {result.best_score:.3f})')
    print()

    if args.apply:
        apply_to_config_yaml(best, CONFIG_PATH)
    else:
        print('Use --apply para gravar esses valores no config.yaml.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
