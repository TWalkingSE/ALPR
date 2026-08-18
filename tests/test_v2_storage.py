# tests/test_v2_storage.py
"""Testes do histórico persistente de leituras (src/v2/storage.py)."""

from datetime import UTC

import numpy as np
import pytest

from src.v2.models import LocalPlateResult
from src.v2.storage import ReadingStore


def _result(plate='ABC1D23', valid=True, confidence=0.91, **kwargs):
    return LocalPlateResult(
        plate_text=plate,
        confidence=confidence,
        detection_confidence=0.88,
        format_type='mercosul',
        is_valid=valid,
        original_crop=np.zeros((4, 8, 3), dtype=np.uint8),
        bbox=(0, 0, 8, 4),
        ocr_engine='paddle_ocr',
        quality_score=0.77,
        **kwargs,
    )


@pytest.fixture
def store(tmp_path):
    instance = ReadingStore(db_path=tmp_path / 'alpr.db')
    yield instance
    instance.close()


class TestLifecycle:
    def test_creates_database_file(self, tmp_path):
        path = tmp_path / 'sub' / 'alpr.db'
        store = ReadingStore(db_path=path)
        assert path.exists()
        store.close()

    def test_disabled_store_does_not_touch_disk(self, tmp_path):
        path = tmp_path / 'alpr.db'
        store = ReadingStore(db_path=path, enabled=False)
        assert not path.exists()
        assert store.record_result(_result()) is None
        assert store.search() == []
        assert store.stats()['total'] == 0

    def test_reopening_preserves_rows(self, tmp_path):
        path = tmp_path / 'alpr.db'
        first = ReadingStore(db_path=path)
        first.record_result(_result())
        first.close()

        second = ReadingStore(db_path=path)
        assert second.stats()['total'] == 1
        second.close()


class TestRecording:
    def test_records_reading(self, store):
        row_id = store.record_result(_result())
        assert row_id is not None

        rows = store.search()
        assert len(rows) == 1
        assert rows[0].plate_normalized == 'ABC1D23'
        assert rows[0].is_valid is True
        assert rows[0].ocr_confidence == pytest.approx(0.91)

    def test_normalizes_plate_for_indexing(self, store):
        store.record_result(_result(plate='ABC-1234'))
        assert store.search()[0].plate_normalized == 'ABC1234'

    def test_skips_result_without_readable_text(self, store):
        assert store.record_result(_result(plate='')) is None
        assert store.stats()['total'] == 0

    def test_persists_tags_and_warnings_as_lists(self, store):
        store.record_result(
            _result(scenario_tags=['low_light', 'small_plate'], warnings=['below_ocr_threshold'])
        )
        row = store.search()[0]
        assert row.scenario_tags == ['low_light', 'small_plate']
        assert row.warnings == ['below_ocr_threshold']

    def test_extracts_sha256_from_report_payload(self, store):
        store.record_result(
            _result(report_payload={'source': {'sha256': 'deadbeef', 'input_file_path': 'a.jpg'}})
        )
        row = store.search()[0]
        assert row.source_sha256 == 'deadbeef'
        assert row.source_path == 'a.jpg'

    def test_records_video_run(self, store):
        video_result = type(
            'VR',
            (),
            {
                'video_path': 'entrada.mp4',
                'processed_frames': 40,
                'skipped_frames': 12,
                'unique_plates': {'ABC1D23': {}},
                'duration_seconds': 8.5,
                'output_video_path': 'saida.mp4',
            },
        )()
        assert store.record_video_run(video_result) is not None
        assert store.stats()['videos'] == 1


class TestSearch:
    @pytest.fixture
    def populated(self, store):
        store.record_result(_result(plate='ABC1D23', valid=True, confidence=0.95))
        store.record_result(_result(plate='ABC1D23', valid=True, confidence=0.80))
        store.record_result(_result(plate='XYZ9K88', valid=False, confidence=0.40))
        return store

    def test_partial_match(self, populated):
        assert len(populated.search(plate='ABC')) == 2
        assert len(populated.search(plate='1D23')) == 2
        assert len(populated.search(plate='XYZ')) == 1

    def test_partial_match_ignores_formatting(self, populated):
        assert len(populated.search(plate='abc-1d23')) == 2

    def test_filter_by_validity(self, populated):
        assert len(populated.search(only_valid=True)) == 2
        assert len(populated.search(only_valid=False)) == 1

    def test_limit(self, populated):
        assert len(populated.search(limit=1)) == 1

    def test_stats(self, populated):
        stats = populated.stats()
        assert stats['total'] == 3
        assert stats['validas'] == 2
        assert stats['placas_distintas'] == 2

    def test_top_plates_ordered_by_frequency(self, populated):
        top = populated.top_plates()
        assert top[0] == ('ABC1D23', 2)

    def test_find_by_sha256_detects_reprocessing(self, store):
        payload = {'source': {'sha256': 'abc123', 'input_file_path': 'foto.jpg'}}
        store.record_result(_result(report_payload=payload))
        store.record_result(_result(report_payload=payload))

        assert len(store.find_by_sha256('abc123')) == 2
        assert store.find_by_sha256('outro') == []
        assert store.find_by_sha256('') == []


class TestResilience:
    def test_write_failure_does_not_raise(self, tmp_path):
        """Falhar ao gravar o histórico não pode derrubar um reconhecimento."""
        store = ReadingStore(db_path=tmp_path / 'alpr.db')
        store.close()

        class _DeadConnection:
            def execute(self, *args, **kwargs):
                raise RuntimeError('db off')

            def close(self):
                pass

        store._connection = _DeadConnection()

        assert store.record_result(_result()) is None
        assert store.record_video_run(object()) is None
        store.close()


class TestPeriodFilter:
    """Filtro por período da aba Histórico (`src/v2/ui/history.py`)."""

    def test_resolve_period_options(self):
        from src.v2.ui.history import PERIOD_OPTIONS, _resolve_period

        assert _resolve_period('Todo o periodo') == (None, None)

        since, until = _resolve_period('Ultimas 24h')
        assert since is not None and until is None

        since_7, _ = _resolve_period('Ultimos 7 dias')
        since_30, _ = _resolve_period('Ultimos 30 dias')
        assert since_30 < since_7, '30 dias deve olhar mais para trás que 7'

        assert 'Personalizado' in PERIOD_OPTIONS

    def test_custom_range_covers_full_days(self):
        from datetime import date

        from src.v2.ui.history import _resolve_period

        since, until = _resolve_period('Personalizado', (date(2026, 8, 1), date(2026, 8, 10)))

        assert since.startswith('2026-08-01T00:00:00')
        assert until.startswith('2026-08-10T23:59:59')

    def test_incomplete_custom_range_does_not_filter(self):
        from datetime import date

        from src.v2.ui.history import _resolve_period

        # O date_input do Streamlit devolve 1 elemento enquanto o usuário
        # escolhe a segunda ponta; nesse estado não se deve filtrar nada.
        assert _resolve_period('Personalizado', (date(2026, 8, 1),)) == (None, None)
        assert _resolve_period('Personalizado', None) == (None, None)

    def test_store_honours_since_and_until(self, store):
        from datetime import datetime, timedelta, timezone

        store.record_result(_result(plate='ABC1D23'))
        rows = store.search()
        assert len(rows) == 1

        agora = datetime.now(UTC)
        futuro = (agora + timedelta(days=1)).isoformat()
        passado = (agora - timedelta(days=1)).isoformat()

        assert len(store.search(since=passado)) == 1
        assert len(store.search(since=futuro)) == 0
        assert len(store.search(until=futuro)) == 1
        assert len(store.search(until=passado)) == 0
        assert len(store.search(since=passado, until=futuro)) == 1
