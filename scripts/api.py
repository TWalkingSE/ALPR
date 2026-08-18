"""API HTTP de leitura de placas (FastAPI).

Expõe o mesmo pipeline local da interface Streamlit para integração com outros
sistemas. O pipeline é construído UMA vez no lifespan da aplicação — nunca por
requisição, que custaria o recarregamento do YOLO e do PaddleOCR.

Requer o extra opcional:
    pip install -e ".[api]"

Execução:
    uvicorn scripts.api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /v1/health           status do pipeline
    POST /v1/plates           uma imagem (multipart) -> laudo estruturado
    POST /v1/plates/batch     várias imagens
    GET  /v1/plates/{placa}   histórico de leituras (exige storage.enabled)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401  (ajusta o sys.path)
import cv2
import numpy as np

try:
    from fastapi import FastAPI, File, HTTPException, Query, UploadFile
except ImportError as exc:  # pragma: no cover - depende do extra opcional
    raise SystemExit(
        'FastAPI não está instalado. Rode: pip install -e ".[api]"'
    ) from exc

from src.config_manager import load_config
from src.v2.config import build_v2_config
from src.v2.models import LocalPlateResult, normalize_plate_text
from src.v2.pipeline import LocalAnalysisPipeline

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Preenchido no lifespan; nenhuma requisição constrói pipeline.
_services: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = build_v2_config(load_config())
    model_path = str(PROJECT_DIR / config.detector.models_dir / config.detector.model_name)

    logger.info('Inicializando pipeline ALPR (modelo: %s)', model_path)
    _services['config'] = config
    _services['pipeline'] = LocalAnalysisPipeline.from_settings(
        config, PROJECT_DIR, model_path=model_path
    )
    logger.info('Pipeline pronto.')

    yield

    store = getattr(_services.get('pipeline'), 'reading_store', None)
    if store is not None:
        store.close()
    _services.clear()


app = FastAPI(
    title='ALPR 2.0',
    version='2.0.0',
    description='Leitura de placas brasileiras — pipeline local offline.',
    lifespan=lifespan,
)


def _get_pipeline() -> LocalAnalysisPipeline:
    pipeline = _services.get('pipeline')
    if pipeline is None:
        raise HTTPException(status_code=503, detail='Pipeline ainda não inicializado.')
    return pipeline


def _decode_upload(payload: bytes) -> np.ndarray:
    if not payload:
        raise HTTPException(status_code=400, detail='Arquivo vazio.')
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail='Não foi possível decodificar a imagem.')
    return image


def _serialize(result: LocalPlateResult) -> dict[str, Any]:
    """Serializa uma leitura.

    Reaproveita o `report_payload` do ReportBuilder quando disponível — é o
    mesmo laudo que o fluxo Streamlit entrega — e cai para um resumo mínimo
    quando os laudos estão desligados.
    """
    if result.report_payload:
        return result.report_payload
    return {
        'recognition': {
            'plate_text': result.plate_text,
            'normalized_text': result.normalized_text,
            'raw_ocr_text': result.raw_ocr_text,
            'format_type': result.format_type,
            'is_valid': result.is_valid,
            'ocr_engine': result.ocr_engine,
            'ocr_confidence': result.confidence,
            'detection_confidence': result.detection_confidence,
        },
        'scenario_tags': list(result.scenario_tags),
        'warnings': list(result.warnings),
        'quality': {'score': result.quality_score, 'metrics': result.quality_metrics},
        'alternatives': list(result.alternative_plates),
        'bbox': list(result.bbox),
        'timing': {'processing_time_ms': result.processing_time_ms},
    }


def _analyze(payload: bytes, filename: str) -> dict[str, Any]:
    pipeline = _get_pipeline()
    config = _services['config']
    image = _decode_upload(payload)

    results = pipeline.process_image(
        image,
        detector_confidence=config.detector.confidence,
        image_bytes=payload,
        input_file_path=filename,
    )
    return {
        'arquivo': filename,
        'total': len(results),
        'placas': [_serialize(result) for result in results],
    }


@app.get('/v1/health')
def health() -> dict[str, Any]:
    pipeline = _services.get('pipeline')
    if pipeline is None:
        return {'status': 'starting'}
    return {'status': 'ok', 'pipeline': pipeline.get_pipeline_info()}


@app.post('/v1/plates')
async def read_plate(upload: UploadFile = File(..., alias='file')) -> dict[str, Any]:
    return _analyze(await upload.read(), upload.filename or 'upload')


@app.post('/v1/plates/batch')
async def read_plates_batch(
    uploads: list[UploadFile] = File(..., alias='files'),
) -> dict[str, Any]:
    resultados = [_analyze(await item.read(), item.filename or 'upload') for item in uploads]
    return {'total_arquivos': len(resultados), 'resultados': resultados}


@app.get('/v1/plates/{placa}')
def plate_history(
    placa: str,
    limit: int = Query(default=50, ge=1, le=1000),
    only_valid: bool | None = Query(default=None),
) -> dict[str, Any]:
    store = getattr(_get_pipeline(), 'reading_store', None)
    if store is None or not store.enabled:
        raise HTTPException(
            status_code=501,
            detail='Historico desabilitado. Ative `storage.enabled` no config.yaml.',
        )

    rows = store.search(plate=placa, only_valid=only_valid, limit=limit)
    return {
        'placa': normalize_plate_text(placa),
        'total': len(rows),
        'leituras': [row.to_dict() for row in rows],
    }
