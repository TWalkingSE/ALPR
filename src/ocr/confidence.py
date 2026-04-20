# src/ocr/confidence.py
"""
Utilitários para estimar confiança OCR via self-consistency.

Modelos Ollama (GLM OCR, OLMoOCR2) não expõem logprobs, então derivamos
confiança comparando múltiplas rodadas do mesmo prompt com temperatura > 0.
A concordância entre as rodadas é um proxy melhor para "quão certo o modelo
está" do que a mera aderência ao formato.

Protocolo:
  1. Rodar o modelo N vezes (tipicamente N=3) com temperatura moderada (0.2).
  2. Escolher o texto que aparece mais vezes (voto majoritário).
  3. Confiança = fração de rodadas que concordaram com o vencedor, modulada
     pela aderência ao formato brasileiro.

Tradeoffs:
  - +N× de latência (chamadas extras ao Ollama).
  - Requer temperatura > 0 para ter variação.
  - Se todas as rodadas concordam → alta confiança (0.90+).
  - Se rodadas discordam → baixa confiança (~0.30-0.50) mesmo em formato válido.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, List, Tuple

_RE_OLD = re.compile(r'^[A-Z]{3}[0-9]{4}$')
_RE_MERCOSUL = re.compile(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$')


def format_aderence_confidence(text: str) -> float:
    """
    Confiança baseada APENAS em aderência ao formato brasileiro.
    (Fallback quando self-consistency não é aplicável.)
    """
    if not text:
        return 0.0
    if _RE_MERCOSUL.match(text):
        return 0.88
    if _RE_OLD.match(text):
        return 0.85
    if len(text) == 7:
        return 0.55
    if len(text) in (6, 8):
        return 0.35
    return 0.20


def self_consistency_confidence(
    samples: List[str],
    floor: float = 0.15,
) -> Tuple[str, float]:
    """
    Calcula confiança por self-consistency entre várias leituras.

    Args:
        samples: Lista de leituras produzidas por chamadas independentes ao
            mesmo modelo (com temperatura > 0).
        floor: Confiança mínima mesmo quando todas as leituras forem vazias.

    Returns:
        (texto_vencedor, confidence) onde:
          - texto_vencedor é a leitura mais frequente não-vazia;
          - confidence combina taxa de concordância com aderência ao formato.
    """
    # Filtrar vazios / 'UNREADABLE' / inválidos óbvios
    valid = [s for s in samples if s and s != 'UNREADABLE']

    if not valid:
        return '', floor

    # Voto majoritário
    counter = Counter(valid)
    winner, votes = counter.most_common(1)[0]
    total = len(samples)  # mantém divisor incluindo vazios (penaliza falhas)

    agreement = votes / total if total > 0 else 0.0

    # Combinar com aderência ao formato (0..1):
    #   conf = 0.6 * agreement + 0.4 * format_score
    # Isso garante:
    #   - unânime + formato válido ⇒ ~0.95
    #   - unânime + formato inválido ⇒ ~0.78
    #   - 2/3 concordância + formato válido ⇒ ~0.75
    #   - sem concordância + formato válido ⇒ ~0.54
    format_score = format_aderence_confidence(winner)
    confidence = 0.6 * agreement + 0.4 * format_score

    # Bonus leve se unanimidade exata (>= 3 amostras)
    if total >= 3 and votes == total:
        confidence = min(1.0, confidence + 0.05)

    return winner, max(floor, min(1.0, confidence))


def run_self_consistency(
    call_fn: Callable[[], str],
    num_samples: int = 3,
    clean_fn: Callable[[str], str] | None = None,
) -> Tuple[str, float, List[str]]:
    """
    Executa N chamadas independentes e consolida.

    Args:
        call_fn: Callable sem argumentos que retorna o texto bruto de 1 rodada.
            (Deve usar temperatura > 0 para gerar variação.)
        num_samples: Número de rodadas (3 é o padrão razoável; >5 custa muito).
        clean_fn: Função para limpar a saída bruta antes da comparação
            (ex: GLMOCREngine._clean_output).

    Returns:
        (winner_text, confidence, raw_samples) — samples é a lista completa
        de saídas (já limpas) para logging/debug.
    """
    samples: List[str] = []
    for _ in range(max(1, num_samples)):
        try:
            raw = call_fn() or ''
        except Exception:
            raw = ''
        cleaned = clean_fn(raw) if clean_fn else raw
        samples.append(cleaned)

    winner, conf = self_consistency_confidence(samples)
    return winner, conf, samples
