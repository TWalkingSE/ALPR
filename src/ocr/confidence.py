# src/ocr/confidence.py
"""
Aderência ao formato de placa brasileira como sinal de confiança.

Usado pelo `PaddleOCREngine` para combinar a confiança nativa de leitura do
engine com a plausibilidade do formato lido (ver `_combine_confidence`).

Nota histórica: este módulo também abrigava um protocolo de *self-consistency*
(N chamadas com temperatura > 0, voto majoritário) que existia para estimar
confiança de engines OCR baseados em LLM via Ollama. Esses engines foram
removidos do projeto — o ALPR 2.0 usa exclusivamente o PaddleOCR, que expõe
confiança nativa por caractere — e o protocolo foi removido junto.
"""

from __future__ import annotations

from src.constants import RE_MERCOSUL, RE_OLD


def format_aderence_confidence(text: str) -> float:
    """
    Confiança baseada APENAS em aderência ao formato brasileiro.

    Args:
        text: Texto já normalizado (maiúsculas, apenas alfanuméricos).

    Returns:
        Score entre 0.0 e 1.0.
    """
    if not text:
        return 0.0
    if RE_MERCOSUL.match(text):
        return 0.88
    if RE_OLD.match(text):
        return 0.85
    if len(text) == 7:
        return 0.55
    if len(text) in (6, 8):
        return 0.35
    return 0.20
