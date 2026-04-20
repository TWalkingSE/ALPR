"""Testes unitários para src/premium_alpr.py — PremiumALPRProvider."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.premium_alpr import PremiumALPRProvider, PremiumALPRResult


def _dummy_image(h=400, w=600):
    return np.full((h, w, 3), 128, dtype=np.uint8)


class TestPremiumALPRResult:
    def test_is_valid_requires_success_and_text(self):
        r = PremiumALPRResult(success=False, plate_text='ABC1234')
        assert r.is_valid is False

        r2 = PremiumALPRResult(success=True, plate_text='ABC1234')
        assert r2.is_valid is True

        r3 = PremiumALPRResult(success=True, plate_text='ABC')
        assert r3.is_valid is False, 'plate_text muito curto é inválido'


class TestAvailability:
    def test_disabled_when_enabled_false(self):
        p = PremiumALPRProvider(enabled=False, api_key='xxxxxxxxxx')
        assert p.available is False

    def test_disabled_when_key_missing(self):
        p = PremiumALPRProvider(enabled=True, api_key='')
        assert p.available is False

    def test_disabled_when_key_placeholder(self):
        p = PremiumALPRProvider(enabled=True, api_key='sua_chave_aqui')
        assert p.available is False

    def test_unknown_provider(self):
        p = PremiumALPRProvider(provider='unknown', api_key='x' * 20)
        assert p.available is False


class TestAnalyzeFullImage:
    @patch('src.premium_alpr.httpx')
    def test_success_valid_plate(self, mock_httpx):
        """Resposta válida da API retorna PremiumALPRResult preenchido."""
        # Mock check_availability
        mock_check = MagicMock()
        mock_check.status_code = 200
        mock_check.json.return_value = {'usage': {'calls': 10, 'max_calls': 2500}}
        mock_httpx.get.return_value = mock_check

        p = PremiumALPRProvider(enabled=True, api_key='x' * 20, regions=['br'])
        assert p.available is True

        # Mock POST (análise)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            'results': [{
                'plate': 'ABC1D23',
                'score': 0.92,
                'box': {'xmin': 10, 'ymin': 20, 'xmax': 100, 'ymax': 60},
                'region': {'code': 'br'},
                'vehicle': {'type': 'car'},
                'candidates': [
                    {'plate': 'ABC1D23', 'score': 0.92},
                    {'plate': 'ABC1023', 'score': 0.35},
                ],
            }]
        }
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=None)
        mock_ctx.post.return_value = mock_resp
        mock_httpx.Client.return_value = mock_ctx

        result = p.analyze_full_image(_dummy_image())

        assert result.success is True
        assert result.plate_text == 'ABC1D23'
        assert result.format_type == 'mercosul'
        assert result.confidence == 0.92
        assert result.region == 'br'
        assert result.vehicle_type == 'car'
        assert result.bbox is not None
        assert len(result.alternates) == 2
        assert result.api_cost_calls == 1

    @patch('src.premium_alpr.httpx')
    def test_no_plates_detected(self, mock_httpx):
        mock_check = MagicMock()
        mock_check.status_code = 200
        mock_check.json.return_value = {'usage': {}}
        mock_httpx.get.return_value = mock_check

        p = PremiumALPRProvider(enabled=True, api_key='x' * 20)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {'results': []}
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=None)
        mock_ctx.post.return_value = mock_resp
        mock_httpx.Client.return_value = mock_ctx

        result = p.analyze_full_image(_dummy_image())
        assert result.success is True
        assert result.plate_text == ''
        assert 'Nenhuma' in (result.error or '')

    @patch('src.premium_alpr.httpx')
    def test_rate_limit_429(self, mock_httpx):
        mock_check = MagicMock()
        mock_check.status_code = 200
        mock_check.json.return_value = {'usage': {}}
        mock_httpx.get.return_value = mock_check

        p = PremiumALPRProvider(enabled=True, api_key='x' * 20)

        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=None)
        mock_ctx.post.return_value = mock_resp
        mock_httpx.Client.return_value = mock_ctx

        result = p.analyze_full_image(_dummy_image())
        assert result.success is False
        assert '429' in result.error

    def test_disabled_returns_error(self):
        p = PremiumALPRProvider(enabled=False)
        result = p.analyze_full_image(_dummy_image())
        assert result.success is False
        assert 'desabilitado' in result.error.lower()

    def test_empty_image(self):
        p = PremiumALPRProvider(enabled=True, api_key='x' * 20)
        # Mesmo com provider "available" via mock, imagem vazia nunca é enviada
        result = p.analyze_full_image(np.array([]))
        assert result.success is False


class TestStructuredLoggingHook:
    @patch('src.premium_alpr.httpx')
    def test_logs_to_structured_logger(self, mock_httpx):
        """Se structured_logger presente, cada chamada é registrada."""
        mock_check = MagicMock()
        mock_check.status_code = 200
        mock_check.json.return_value = {'usage': {}}
        mock_httpx.get.return_value = mock_check

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            'results': [{'plate': 'ABC1234', 'score': 0.9,
                         'region': {'code': 'br'}, 'vehicle': {'type': 'car'}}]
        }
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=None)
        mock_ctx.post.return_value = mock_resp
        mock_httpx.Client.return_value = mock_ctx

        slog = MagicMock()
        p = PremiumALPRProvider(
            enabled=True, api_key='x' * 20,
            structured_logger=slog, log_all_calls=True,
        )
        p.analyze_full_image(_dummy_image())

        slog._write_event.assert_called_once()
        event = slog._write_event.call_args[0][0]
        assert event['event_type'] == 'premium_api_call'
        assert event['success'] is True
        assert event['plate_text'] == 'ABC1234'
