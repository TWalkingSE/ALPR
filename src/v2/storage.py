"""Histórico persistente de leituras de placa (SQLite).

Cada leitura já gera um laudo JSON solto em `data/results/` — bom para auditoria
pontual, inútil para perguntas agregadas: "essa placa já apareceu antes?",
"quantas leituras ficaram abaixo do limiar esta semana?", "este arquivo já foi
processado?". Este módulo dá um índice consultável sobre as mesmas leituras.

Usa apenas a stdlib (`sqlite3`). Desligado por padrão — ver a seção `storage:`
do config.yaml.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.v2.models import LocalPlateResult, normalize_plate_text

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT    NOT NULL,
    plate_normalized    TEXT    NOT NULL,
    plate_text          TEXT    NOT NULL,
    format_type         TEXT    NOT NULL DEFAULT 'unknown',
    is_valid            INTEGER NOT NULL DEFAULT 0,
    ocr_confidence      REAL    NOT NULL DEFAULT 0.0,
    detection_confidence REAL   NOT NULL DEFAULT 0.0,
    quality_score       REAL    NOT NULL DEFAULT 0.0,
    ocr_engine          TEXT    NOT NULL DEFAULT '',
    scenario_tags       TEXT    NOT NULL DEFAULT '[]',
    warnings            TEXT    NOT NULL DEFAULT '[]',
    report_path         TEXT    NOT NULL DEFAULT '',
    artifact_dir        TEXT    NOT NULL DEFAULT '',
    source_sha256       TEXT    NOT NULL DEFAULT '',
    source_path         TEXT    NOT NULL DEFAULT '',
    origin              TEXT    NOT NULL DEFAULT 'image'
);

CREATE INDEX IF NOT EXISTS idx_readings_plate   ON readings (plate_normalized);
CREATE INDEX IF NOT EXISTS idx_readings_created ON readings (created_at);
CREATE INDEX IF NOT EXISTS idx_readings_sha     ON readings (source_sha256);

CREATE TABLE IF NOT EXISTS video_runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        TEXT    NOT NULL,
    video_path        TEXT    NOT NULL,
    processed_frames  INTEGER NOT NULL DEFAULT 0,
    skipped_frames    INTEGER NOT NULL DEFAULT 0,
    unique_plates     INTEGER NOT NULL DEFAULT 0,
    duration_seconds  REAL    NOT NULL DEFAULT 0.0,
    output_video_path TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_video_runs_created ON video_runs (created_at);

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class ReadingRow:
    """Uma leitura registrada no histórico."""

    id: int
    created_at: str
    plate_normalized: str
    plate_text: str
    format_type: str
    is_valid: bool
    ocr_confidence: float
    detection_confidence: float
    quality_score: float
    ocr_engine: str
    scenario_tags: list[str]
    warnings: list[str]
    report_path: str
    artifact_dir: str
    source_sha256: str
    source_path: str
    origin: str

    @classmethod
    def from_sqlite(cls, row: sqlite3.Row) -> ReadingRow:
        return cls(
            id=int(row['id']),
            created_at=str(row['created_at']),
            plate_normalized=str(row['plate_normalized']),
            plate_text=str(row['plate_text']),
            format_type=str(row['format_type']),
            is_valid=bool(row['is_valid']),
            ocr_confidence=float(row['ocr_confidence']),
            detection_confidence=float(row['detection_confidence']),
            quality_score=float(row['quality_score']),
            ocr_engine=str(row['ocr_engine']),
            scenario_tags=json.loads(row['scenario_tags'] or '[]'),
            warnings=json.loads(row['warnings'] or '[]'),
            report_path=str(row['report_path']),
            artifact_dir=str(row['artifact_dir']),
            source_sha256=str(row['source_sha256']),
            source_path=str(row['source_path']),
            origin=str(row['origin']),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'created_at': self.created_at,
            'plate_normalized': self.plate_normalized,
            'plate_text': self.plate_text,
            'format_type': self.format_type,
            'is_valid': self.is_valid,
            'ocr_confidence': self.ocr_confidence,
            'detection_confidence': self.detection_confidence,
            'quality_score': self.quality_score,
            'ocr_engine': self.ocr_engine,
            'scenario_tags': list(self.scenario_tags),
            'warnings': list(self.warnings),
            'report_path': self.report_path,
            'artifact_dir': self.artifact_dir,
            'source_sha256': self.source_sha256,
            'source_path': self.source_path,
            'origin': self.origin,
        }


class ReadingStore:
    """Repositório SQLite das leituras.

    O Streamlit reexecuta o script em threads diferentes e a API atende
    requisições concorrentes, então a conexão usa `check_same_thread=False`
    protegida por um lock — o volume de escrita (uma linha por placa lida) não
    justifica um pool.
    """

    def __init__(self, db_path: str | Path, enabled: bool = True):
        self.enabled = bool(enabled)
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._connection: sqlite3.Connection | None = None

        if self.enabled:
            self._connect()

    @classmethod
    def from_settings(cls, settings) -> ReadingStore:
        return cls(db_path=settings.db_path, enabled=settings.enabled)

    def _connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.executescript(_SCHEMA)
            self._connection.execute(
                'INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)',
                ('schema_version', str(SCHEMA_VERSION)),
            )
            self._connection.commit()
        logger.info('Histórico de leituras em %s', self.db_path)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    # ------------------------------------------------------------------
    # Escrita
    # ------------------------------------------------------------------

    def record_result(
        self,
        result: LocalPlateResult,
        origin: str = 'image',
        source_path: str = '',
    ) -> int | None:
        """Registra uma leitura. Devolve o id, ou None se desabilitado.

        Nunca propaga exceção: falhar ao gravar o histórico não pode derrubar
        um reconhecimento que já foi concluído com sucesso.
        """
        if not self.enabled or self._connection is None:
            return None

        plate_normalized = normalize_plate_text(result.plate_text or result.raw_ocr_text)
        if not plate_normalized:
            return None

        source_sha256 = ''
        report_source = (result.report_payload or {}).get('source') or {}
        if isinstance(report_source, dict):
            source_sha256 = str(report_source.get('sha256', ''))
            source_path = source_path or str(report_source.get('input_file_path', ''))

        try:
            with self._lock:
                cursor = self._connection.execute(
                    """
                    INSERT INTO readings (
                        created_at, plate_normalized, plate_text, format_type, is_valid,
                        ocr_confidence, detection_confidence, quality_score, ocr_engine,
                        scenario_tags, warnings, report_path, artifact_dir,
                        source_sha256, source_path, origin
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now(UTC).isoformat(),
                        plate_normalized,
                        result.plate_text,
                        result.format_type,
                        int(bool(result.is_valid)),
                        float(result.confidence),
                        float(result.detection_confidence),
                        float(result.quality_score),
                        result.ocr_engine,
                        json.dumps(list(result.scenario_tags), ensure_ascii=False),
                        json.dumps(list(result.warnings), ensure_ascii=False),
                        result.report_path,
                        result.artifact_dir,
                        source_sha256,
                        source_path,
                        origin,
                    ),
                )
                self._connection.commit()
                return int(cursor.lastrowid) if cursor.lastrowid is not None else None
        except Exception:
            logger.exception('Falha ao gravar leitura no histórico')
            return None

    def record_video_run(self, video_result, video_path: str = '') -> int | None:
        """Registra o resumo de um processamento de vídeo."""
        if not self.enabled or self._connection is None:
            return None

        try:
            with self._lock:
                cursor = self._connection.execute(
                    """
                    INSERT INTO video_runs (
                        created_at, video_path, processed_frames, skipped_frames,
                        unique_plates, duration_seconds, output_video_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now(UTC).isoformat(),
                        video_path or getattr(video_result, 'video_path', ''),
                        int(getattr(video_result, 'processed_frames', 0)),
                        int(getattr(video_result, 'skipped_frames', 0)),
                        len(getattr(video_result, 'unique_plates', {}) or {}),
                        float(getattr(video_result, 'duration_seconds', 0.0)),
                        str(getattr(video_result, 'output_video_path', '') or ''),
                    ),
                )
                self._connection.commit()
                return int(cursor.lastrowid) if cursor.lastrowid is not None else None
        except Exception:
            logger.exception('Falha ao gravar execução de vídeo no histórico')
            return None

    # ------------------------------------------------------------------
    # Leitura
    # ------------------------------------------------------------------

    def search(
        self,
        plate: str = '',
        only_valid: bool | None = None,
        since: str | None = None,
        until: str | None = None,
        origin: str = '',
        limit: int = 200,
    ) -> list[ReadingRow]:
        """Busca leituras. `plate` casa parcialmente (normalizado)."""
        if not self.enabled or self._connection is None:
            return []

        clauses: list[str] = []
        params: list[Any] = []

        if plate:
            clauses.append('plate_normalized LIKE ?')
            params.append(f'%{normalize_plate_text(plate)}%')
        if only_valid is not None:
            clauses.append('is_valid = ?')
            params.append(int(only_valid))
        if since:
            clauses.append('created_at >= ?')
            params.append(since)
        if until:
            clauses.append('created_at <= ?')
            params.append(until)
        if origin:
            clauses.append('origin = ?')
            params.append(origin)

        where = f' WHERE {" AND ".join(clauses)}' if clauses else ''
        query = f'SELECT * FROM readings{where} ORDER BY created_at DESC, id DESC LIMIT ?'
        params.append(max(1, int(limit)))

        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return [ReadingRow.from_sqlite(row) for row in rows]

    def find_by_sha256(self, sha256: str, limit: int = 20) -> list[ReadingRow]:
        """Leituras anteriores do mesmo arquivo — detecta reprocessamento."""
        if not self.enabled or self._connection is None or not sha256:
            return []
        with self._lock:
            rows = self._connection.execute(
                'SELECT * FROM readings WHERE source_sha256 = ? '
                'ORDER BY created_at DESC LIMIT ?',
                (sha256, max(1, int(limit))),
            ).fetchall()
        return [ReadingRow.from_sqlite(row) for row in rows]

    def stats(self) -> dict[str, Any]:
        """Contagens agregadas para o painel de histórico."""
        if not self.enabled or self._connection is None:
            return {'total': 0, 'validas': 0, 'placas_distintas': 0, 'videos': 0}

        with self._lock:
            total, validas, distintas = self._connection.execute(
                'SELECT COUNT(*), COALESCE(SUM(is_valid), 0), '
                'COUNT(DISTINCT plate_normalized) FROM readings'
            ).fetchone()
            (videos,) = self._connection.execute('SELECT COUNT(*) FROM video_runs').fetchone()

        return {
            'total': int(total),
            'validas': int(validas),
            'placas_distintas': int(distintas),
            'videos': int(videos),
        }

    def top_plates(self, limit: int = 10) -> Sequence[tuple[str, int]]:
        """Placas mais lidas, da mais frequente para a menos."""
        if not self.enabled or self._connection is None:
            return []
        with self._lock:
            rows = self._connection.execute(
                'SELECT plate_normalized, COUNT(*) AS total FROM readings '
                'GROUP BY plate_normalized ORDER BY total DESC, plate_normalized LIMIT ?',
                (max(1, int(limit)),),
            ).fetchall()
        return [(str(row['plate_normalized']), int(row['total'])) for row in rows]
