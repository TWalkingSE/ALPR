from typing import Any, Mapping, Sequence, TypedDict, cast

OCRCharConfidence = tuple[str, float]
OCRBBox = list[Any]


class OCRResult(TypedDict, total=False):
    text: str
    confidence: float
    bbox: OCRBBox
    engine: str
    char_confidences: list[OCRCharConfidence]
    merge_sources: list[str]
    used_fallback: bool
    used_original: bool
    variant_idx: int
    ngram_score: float
    region: str
    vehicle_type: str
    substituted_pos4: str
    votes: int
    avg_confidence: float
    engines_voted: list[str]
    # Auditoria de self-consistency: amostras brutas quando N>1 chamadas foram
    # feitas pelo engine (GLM/OLMoOCR2) para medir concordância.
    self_consistency_samples: list[str]


def normalize_ocr_text(text: str) -> str:
    if not text:
        return ''
    return text.strip().upper().replace(' ', '').replace('-', '')


def normalize_char_confidences(
    char_confidences: Sequence[tuple[str, float]] | None,
) -> list[OCRCharConfidence]:
    if not char_confidences:
        return []

    normalized: list[OCRCharConfidence] = []
    for item in char_confidences:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        char, confidence = item
        normalized.append((str(char).upper(), float(confidence)))
    return normalized


def create_ocr_result(
    *,
    text: str,
    confidence: float,
    engine: str,
    bbox: Sequence[Any] | None = None,
    char_confidences: Sequence[tuple[str, float]] | None = None,
    **extra: Any,
) -> OCRResult:
    result: OCRResult = {
        'text': normalize_ocr_text(text),
        'confidence': max(0.0, min(1.0, float(confidence))),
        'bbox': list(bbox) if bbox is not None else [],
        'engine': engine,
        'char_confidences': normalize_char_confidences(char_confidences),
    }
    result.update(extra)
    return result


def clone_ocr_result(result: Mapping[str, Any], **updates: Any) -> OCRResult:
    merged = dict(result)
    merged.update(updates)
    return create_ocr_result(
        text=str(merged.get('text', '')),
        confidence=float(merged.get('confidence', 0.0)),
        engine=str(merged.get('engine', 'unknown')),
        bbox=cast(Sequence[Any] | None, merged.get('bbox')),
        char_confidences=cast(Sequence[tuple[str, float]] | None, merged.get('char_confidences')),
        **{
            key: value
            for key, value in merged.items()
            if key not in {'text', 'confidence', 'engine', 'bbox', 'char_confidences'}
        },
    )
