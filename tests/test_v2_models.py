import numpy as np

from src.v2.models import LocalPlateResult, normalize_plate_text


def test_normalize_plate_text_strips_non_alnum():
    assert normalize_plate_text('ABC-1D23') == 'ABC1D23'
    assert normalize_plate_text(' abc 1234 ') == 'ABC1234'


def test_local_plate_result_keeps_v2_metadata():
    image = np.zeros((40, 120, 3), dtype=np.uint8)
    result = LocalPlateResult(
        plate_text='ABC1D23',
        confidence=0.91,
        detection_confidence=0.87,
        format_type='mercosul',
        is_valid=True,
        original_crop=image,
        bbox=(1, 2, 3, 4),
        normalized_crop=image,
        preprocessed_image=image[:, :, 0],
        ocr_engine='paddle_ocr',
        char_confidences=[('A', 0.9)],
        alternative_plates=[{'text': 'ABC1023', 'probability': 0.2, 'changes': '5:D->0'}],
        raw_ocr_text='ABC1D23',
        normalized_text='ABC1D23',
        warnings=['candidate_ranking'],
    )

    assert result.plate_text == 'ABC1D23'
    assert result.confidence == 0.91
    assert result.bbox == (1, 2, 3, 4)
    assert result.ocr_engine == 'paddle_ocr'
    assert result.alternative_plates[0]['text'] == 'ABC1023'
    assert result.normalized_text == 'ABC1D23'
    assert result.warnings == ['candidate_ranking']