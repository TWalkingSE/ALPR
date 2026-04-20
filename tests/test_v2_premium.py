from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from src.premium_alpr import PremiumALPRResult
from src.v2.config import PremiumConfig
from src.v2.premium import PremiumAnalysisService


def test_premium_service_returns_disabled_result_without_provider():
    service = PremiumAnalysisService.from_settings(
        PremiumConfig(enabled=True, api_key='')
    )

    result = service.analyze_full_image(np.zeros((8, 8, 3), dtype=np.uint8))

    assert service.available is False
    assert result.success is False
    assert 'desabilitada' in (result.error or '').lower() or 'sem chave' in (result.error or '').lower()


@patch('src.v2.premium.PremiumALPRProvider')
def test_premium_service_builds_provider_from_settings(mock_provider_cls):
    provider = MagicMock()
    provider.available = True
    provider.provider = 'platerecognizer'
    mock_provider_cls.return_value = provider

    settings = PremiumConfig(
        enabled=True,
        provider='platerecognizer',
        api_key='x' * 20,
        regions=['br'],
        min_confidence=0.7,
        timeout=45,
        log_all_calls=False,
    )

    service = PremiumAnalysisService.from_settings(settings)

    mock_provider_cls.assert_called_once_with(
        provider='platerecognizer',
        api_key='x' * 20,
        regions=['br'],
        min_confidence=0.7,
        timeout=45,
        enabled=True,
        log_all_calls=False,
    )
    assert service.available is True
    assert service.provider == 'platerecognizer'


def test_premium_service_delegates_to_provider():
    expected = PremiumALPRResult(
        success=True,
        plate_text='ABC1D23',
        provider='platerecognizer',
    )
    client = MagicMock()
    client.available = True
    client.provider = 'platerecognizer'
    client.analyze_full_image.return_value = expected
    service = PremiumAnalysisService(
        client=client,
        settings=PremiumConfig(enabled=True, api_key='x' * 20),
    )
    image = np.zeros((12, 24, 3), dtype=np.uint8)

    result = service.analyze_full_image(image)

    client.analyze_full_image.assert_called_once_with(image)
    assert result is expected