"""Execução de fixtures rotuladas contra o pipeline local.

Compartilhado por ``scripts/evaluate.py`` (baseline) e ``scripts/calibrate.py``
(grid-search de thresholds), para que ambos meçam exatamente a mesma coisa.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import _bootstrap  # noqa: F401  (ajusta o sys.path)
import cv2
import numpy as np

from src.v2.application import build_video_processor
from src.v2.config import AppConfig
from src.v2.evaluation import FixtureEntry, PredictionRecord, build_prediction_record
from src.v2.models import LocalPlateResult
from src.v2.pipeline import LocalAnalysisPipeline

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, FixtureEntry], None]


def resolve_fixture_path(fixture: FixtureEntry, manifest_path: Path, fixtures_dir: Path) -> Path:
    """Resolve o caminho de uma fixture.

    Tenta, nesta ordem: caminho absoluto, relativo ao diretório do manifesto,
    relativo a ``evaluation.fixtures_dir`` e relativo ao diretório atual.
    """
    candidate = Path(fixture.path)
    if candidate.is_absolute():
        return candidate

    for base in (manifest_path.parent, fixtures_dir, Path.cwd()):
        resolved = base / candidate
        if resolved.exists():
            return resolved
    return manifest_path.parent / candidate


def run_image_fixture(
    pipeline: LocalAnalysisPipeline,
    config: AppConfig,
    media_path: Path,
) -> LocalPlateResult | None:
    """Roda o pipeline local em uma imagem e devolve a melhor leitura."""
    image = cv2.imread(str(media_path))
    if image is None:
        logger.warning('Não foi possível ler a imagem: %s', media_path)
        return None

    results = pipeline.process_image(
        image,
        detector_confidence=config.detector.confidence,
        input_file_path=str(media_path),
    )
    return results[0] if results else None


def run_video_fixture(
    pipeline: LocalAnalysisPipeline,
    config: AppConfig,
    media_path: Path,
) -> LocalPlateResult | None:
    """Roda o pipeline de vídeo e converte a placa melhor ranqueada em resultado.

    A avaliação compara uma leitura por fixture, então a consolidação temporal
    do vídeo é reduzida à placa de maior score composto.
    """
    processor = build_video_processor(config)
    video_result = processor.process_video(
        str(media_path),
        pipeline=pipeline,
        detector_confidence=config.detector.confidence,
    )

    ranked = processor.rank_unique_plates(video_result.unique_plates)
    if not ranked:
        return None

    info = next(iter(ranked.values()))
    plate_text = str(info.get('plate_text', ''))
    return LocalPlateResult(
        plate_text=plate_text,
        confidence=float(info.get('best_confidence', 0.0)),
        detection_confidence=float(info.get('best_confidence', 0.0)),
        format_type='unknown',
        is_valid=bool(plate_text),
        # A avaliação só consome campos escalares; um recorte real não é
        # necessário e carregá-lo custaria memória à toa.
        original_crop=np.zeros((1, 1, 3), dtype=np.uint8),
        bbox=tuple(info.get('best_bbox', (0, 0, 0, 0))),
        normalized_text=plate_text,
        processing_time_ms=float(video_result.avg_processing_time_per_frame),
    )


def run_fixtures(
    pipeline: LocalAnalysisPipeline,
    config: AppConfig,
    fixtures: Sequence[FixtureEntry],
    manifest_path: Path,
    on_progress: ProgressCallback | None = None,
) -> list[PredictionRecord]:
    """Executa todas as fixtures e devolve as linhas de predição."""
    fixtures_dir = Path(config.evaluation.fixtures_dir)
    records: list[PredictionRecord] = []

    for index, fixture in enumerate(fixtures, start=1):
        if on_progress is not None:
            on_progress(index, len(fixtures), fixture)

        media_path = resolve_fixture_path(fixture, manifest_path, fixtures_dir)
        if not media_path.exists():
            logger.warning('Fixture ausente no disco: %s (%s)', fixture.fixture_id, media_path)
            records.append(build_prediction_record(fixture, None))
            continue

        try:
            if fixture.media_type == 'video':
                result = run_video_fixture(pipeline, config, media_path)
            else:
                result = run_image_fixture(pipeline, config, media_path)
        except Exception:
            logger.exception('Falha ao processar a fixture %s', fixture.fixture_id)
            result = None

        records.append(build_prediction_record(fixture, result))

    return records
