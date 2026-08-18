"""Leitura de placas em lote, sem Streamlit.

Processa uma imagem, um vídeo ou um diretório inteiro usando o mesmo pipeline
local da interface web, e emite o resultado em texto, JSON Lines ou CSV.

Uso:
    python scripts/alpr_cli.py imagem.jpg
    python scripts/alpr_cli.py ./pasta --recursive --out resultados.csv
    python scripts/alpr_cli.py video.mp4 --out resultado.jsonl
    python scripts/alpr_cli.py ./pasta --format json > leituras.jsonl

Código de saída:
    0  ao menos uma placa válida foi lida
    1  nada válido foi lido (útil em scripting)
    2  erro de configuração/inicialização
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401  (ajusta o sys.path)
import cv2

from src.config_manager import load_config
from src.v2.application import build_video_processor
from src.v2.config import AppConfig, build_v2_config
from src.v2.models import LocalPlateResult, normalize_plate_text
from src.v2.pipeline import LocalAnalysisPipeline
from src.video_processor import SUPPORTED_VIDEO_EXTENSIONS

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
VIDEO_EXTENSIONS = set(SUPPORTED_VIDEO_EXTENSIONS)

CSV_FIELDS = [
    'arquivo',
    'tipo',
    'placa',
    'formato',
    'valida',
    'confianca_ocr',
    'confianca_deteccao',
    'qualidade',
    'cenarios',
    'avisos',
    'tempo_ms',
    'laudo',
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Leitura de placas em lote (headless).',
    )
    parser.add_argument('entrada', help='Imagem, vídeo ou diretório a processar.')
    parser.add_argument(
        '--recursive', '-r', action='store_true',
        help='Percorrer subdiretórios quando a entrada for um diretório.',
    )
    parser.add_argument(
        '--format', '-f', choices=('text', 'json', 'csv'), default='text',
        help='Formato de saída (padrão: text). O formato é inferido pela extensão de --out.',
    )
    parser.add_argument(
        '--out', '-o', default=None,
        help='Arquivo de saída. Sem isto, escreve no stdout.',
    )
    parser.add_argument(
        '--all-plates', action='store_true',
        help='Emitir todas as placas de cada imagem, não apenas a melhor.',
    )
    parser.add_argument('--model', default=None, help='Caminho do peso YOLO.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Log em nível INFO.')
    return parser.parse_args(argv)


def iter_input_files(entrada: Path, recursive: bool) -> Iterator[Path]:
    """Enumera os arquivos de mídia suportados a partir da entrada."""
    if entrada.is_file():
        yield entrada
        return

    pattern = '**/*' if recursive else '*'
    supported = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    for path in sorted(entrada.glob(pattern)):
        if path.is_file() and path.suffix.lower() in supported:
            yield path


def _row_from_result(path: Path, result: LocalPlateResult, tipo: str) -> dict[str, Any]:
    return {
        'arquivo': str(path),
        'tipo': tipo,
        'placa': result.plate_text,
        'formato': result.format_type,
        'valida': result.is_valid,
        'confianca_ocr': round(result.confidence, 4),
        'confianca_deteccao': round(result.detection_confidence, 4),
        'qualidade': round(result.quality_score, 4),
        'cenarios': ','.join(result.scenario_tags),
        'avisos': ','.join(result.warnings),
        'tempo_ms': round(result.processing_time_ms, 2),
        'laudo': result.report_path,
    }


def process_image_file(
    pipeline: LocalAnalysisPipeline,
    config: AppConfig,
    path: Path,
    all_plates: bool,
) -> list[dict[str, Any]]:
    image = cv2.imread(str(path))
    if image is None:
        logger.warning('Não foi possível ler a imagem: %s', path)
        return []

    results = pipeline.process_image(
        image,
        detector_confidence=config.detector.confidence,
        image_bytes=path.read_bytes(),
        input_file_path=str(path),
    )
    if not results:
        return []
    selected = results if all_plates else results[:1]
    return [_row_from_result(path, result, 'imagem') for result in selected]


def process_video_file(
    pipeline: LocalAnalysisPipeline,
    config: AppConfig,
    path: Path,
) -> list[dict[str, Any]]:
    processor = build_video_processor(config)
    video_result = processor.process_video(
        str(path),
        pipeline=pipeline,
        detector_confidence=config.detector.confidence,
    )
    ranked = processor.rank_unique_plates(video_result.unique_plates)

    return [
        {
            'arquivo': str(path),
            'tipo': 'video',
            'placa': info.get('plate_text', ''),
            'formato': 'unknown',
            'valida': bool(normalize_plate_text(str(info.get('plate_text', '')))),
            'confianca_ocr': round(float(info.get('best_confidence', 0.0)), 4),
            'confianca_deteccao': round(float(info.get('best_confidence', 0.0)), 4),
            'qualidade': round(float(info.get('best_quality_score', 0.0)), 4),
            'cenarios': ','.join(info.get('scenario_counts', {})),
            'avisos': f"deteccoes={info.get('total_detections', 0)}",
            'tempo_ms': round(video_result.avg_processing_time_per_frame, 2),
            'laudo': processor.build_confirmed_reading(info),
        }
        for info in ranked.values()
    ]


def write_output(rows: list[dict[str, Any]], output_format: str, out_path: Path | None) -> None:
    if output_format == 'csv':
        handle = out_path.open('w', encoding='utf-8', newline='') if out_path else sys.stdout
        try:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        finally:
            if out_path:
                handle.close()
        return

    if output_format == 'json':
        lines = '\n'.join(json.dumps(row, ensure_ascii=False) for row in rows)
        if out_path:
            out_path.write_text(lines + '\n', encoding='utf-8')
        else:
            print(lines)
        return

    # text
    lines = []
    for row in rows:
        marca = 'OK  ' if row['valida'] else 'REV '
        lines.append(
            f"{marca} {row['placa'] or '-':<10} {row['confianca_ocr']:>6.1%}  "
            f"{row['formato']:<9} {row['arquivo']}"
        )
    texto = '\n'.join(lines)
    if out_path:
        out_path.write_text(texto + '\n', encoding='utf-8')
    else:
        print(texto)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(levelname)s %(name)s: %(message)s',
    )

    entrada = Path(args.entrada)
    if not entrada.exists():
        print(f'ERRO: entrada não encontrada: {entrada}', file=sys.stderr)
        return 2

    files = list(iter_input_files(entrada, args.recursive))
    if not files:
        print(f'ERRO: nenhum arquivo de mídia suportado em {entrada}', file=sys.stderr)
        return 2

    out_path = Path(args.out) if args.out else None
    output_format = args.format
    if out_path is not None and args.format == 'text':
        # Deixar a extensão do arquivo decidir quando o formato não foi pedido.
        suffix = out_path.suffix.lower()
        if suffix == '.csv':
            output_format = 'csv'
        elif suffix in ('.json', '.jsonl'):
            output_format = 'json'

    config = build_v2_config(load_config())
    model_path = args.model or str(
        PROJECT_DIR / config.detector.models_dir / config.detector.model_name
    )

    try:
        pipeline = LocalAnalysisPipeline.from_settings(config, PROJECT_DIR, model_path=model_path)
    except Exception as exc:
        print(f'ERRO ao inicializar o pipeline: {exc}', file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    for index, path in enumerate(files, start=1):
        if args.verbose or len(files) > 1:
            print(f'[{index}/{len(files)}] {path.name}', file=sys.stderr, flush=True)
        try:
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                rows.extend(process_video_file(pipeline, config, path))
            else:
                rows.extend(process_image_file(pipeline, config, path, args.all_plates))
        except Exception:
            logger.exception('Falha ao processar %s', path)

    try:
        write_output(rows, output_format, out_path)
    except OSError as exc:
        # Não faz sentido perder o processamento por causa do arquivo de saída:
        # relata o erro e cai para o stdout.
        print(f'ERRO ao escrever em {out_path}: {exc}', file=sys.stderr)
        print('Emitindo o resultado no stdout:', file=sys.stderr)
        write_output(rows, output_format, None)
        return 2

    if out_path:
        print(f'{len(rows)} leitura(s) em {out_path}', file=sys.stderr)

    return 0 if any(row['valida'] for row in rows) else 1


if __name__ == '__main__':
    sys.exit(main())
