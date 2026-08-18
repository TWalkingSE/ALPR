"""Premium analysis adapter for ALPR v2."""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.premium_alpr import PremiumALPRProvider, PremiumALPRResult
from src.v2.config import PremiumConfig


class PremiumAnalysisService:
    """Thin wrapper around the stable Plate Recognizer integration."""

    def __init__(self, client: Optional[PremiumALPRProvider], settings: PremiumConfig):
        self.client = client
        self.settings = settings

    @classmethod
    def from_settings(cls, settings: PremiumConfig) -> 'PremiumAnalysisService':
        client = None
        if settings.enabled and settings.api_key:
            client = PremiumALPRProvider(
                provider=settings.provider,
                api_key=settings.api_key,
                regions=settings.regions,
                min_confidence=settings.min_confidence,
                timeout=settings.timeout,
                enabled=True,
                log_all_calls=settings.log_all_calls,
            )
        return cls(client=client, settings=settings)

    def apply_runtime_config(self, settings: PremiumConfig) -> None:
        """Atualiza os ajustes que não trocam a identidade do cliente.

        `enabled`, `provider` e a presença da chave definem se existe cliente e
        qual é (ver `AppConfig.signature`); limiar, regiões e timeout podem ser
        alterados no cliente já construído, sem recriá-lo.
        """
        self.settings = settings
        if self.client is None:
            return
        self.client.regions = list(settings.regions) or ['br']
        self.client.min_confidence = settings.min_confidence
        self.client.timeout = settings.timeout
        self.client.log_all_calls = settings.log_all_calls

    @property
    def available(self) -> bool:
        return self.client is not None and self.client.available

    @property
    def provider(self) -> str:
        if self.client is not None:
            return self.client.provider
        return self.settings.provider

    def analyze_full_image(self, image: np.ndarray) -> PremiumALPRResult:
        if self.client is None:
            return PremiumALPRResult(
                success=False,
                provider=self.settings.provider,
                error='Premium API desabilitada ou sem chave configurada',
            )
        return self.client.analyze_full_image(image)