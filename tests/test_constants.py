# tests/test_constants.py
"""
Testes unitários para src/constants.py.
Verifica integridade dos mapas de caracteres e padrões de placa.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.constants import (
    SIMILAR_CHARS,
    OCR_CHAR_MAP,
    PLATE_PATTERN_OLD,
    PLATE_PATTERN_MERCOSUL,
    RE_OLD,
    RE_MERCOSUL,
    PLATE_POSITIONS_OLD,
    PLATE_POSITIONS_MERCOSUL,
    CONFIDENCE_LEVELS,
    FORENSIC_TERMS,
    MAX_UNIQUE_PLATES,
    CHAR_CONFIDENCE_THRESHOLD,
)


class TestSimilarChars:
    def test_is_dict(self):
        assert isinstance(SIMILAR_CHARS, dict)

    def test_values_are_lists(self):
        for k, v in SIMILAR_CHARS.items():
            assert isinstance(v, list), f"SIMILAR_CHARS['{k}'] should be a list"

    def test_zero_maps_to_O(self):
        assert 'O' in SIMILAR_CHARS['0']

    def test_O_maps_to_zero(self):
        assert '0' in SIMILAR_CHARS['O']

    def test_bidirectional(self):
        """Se A está nos similares de B, B deveria estar nos similares de A."""
        for char, similar_list in SIMILAR_CHARS.items():
            for s in similar_list:
                assert s in SIMILAR_CHARS, f"'{s}' is in SIMILAR_CHARS['{char}'] but has no entry"
                assert char in SIMILAR_CHARS[s], (
                    f"'{char}' -> '{s}' não é bidirecional"
                )


class TestOCRCharMap:
    def test_is_dict(self):
        assert isinstance(OCR_CHAR_MAP, dict)

    def test_simple_values(self):
        for k, v in OCR_CHAR_MAP.items():
            assert isinstance(v, str) and len(v) == 1, (
                f"OCR_CHAR_MAP['{k}'] should be single char, got '{v}'"
            )

    def test_zero_maps_to_O(self):
        assert OCR_CHAR_MAP['0'] == 'O'

    def test_O_maps_to_zero(self):
        assert OCR_CHAR_MAP['O'] == '0'


class TestPlatePatterns:
    @pytest.mark.parametrize("plate", ["ABC1234", "XYZ9876", "AAA0000"])
    def test_old_format_matches(self, plate):
        assert RE_OLD.match(plate), f"'{plate}' should match old format"

    @pytest.mark.parametrize("plate", ["ABC1D23", "XYZ9A00", "AAA0B99"])
    def test_mercosul_format_matches(self, plate):
        assert RE_MERCOSUL.match(plate), f"'{plate}' should match Mercosul format"

    @pytest.mark.parametrize("plate", ["AB12345", "ABCD123", "123ABCD", "abc1234"])
    def test_invalid_old_no_match(self, plate):
        assert not RE_OLD.match(plate), f"'{plate}' should NOT match old format"

    @pytest.mark.parametrize("plate", ["ABC12D3", "1BC1D23", "ABCxD23"])
    def test_invalid_mercosul_no_match(self, plate):
        assert not RE_MERCOSUL.match(plate), f"'{plate}' should NOT match Mercosul"


class TestPlatePositions:
    def test_old_length(self):
        assert len(PLATE_POSITIONS_OLD) == 7

    def test_mercosul_length(self):
        assert len(PLATE_POSITIONS_MERCOSUL) == 7

    def test_old_pattern(self):
        assert PLATE_POSITIONS_OLD == ['L', 'L', 'L', 'N', 'N', 'N', 'N']

    def test_mercosul_pattern(self):
        assert PLATE_POSITIONS_MERCOSUL == ['L', 'L', 'L', 'N', 'L', 'N', 'N']


class TestConfidenceLevels:
    def test_required_levels(self):
        for level in ('alta', 'media', 'baixa', 'insuficiente'):
            assert level in CONFIDENCE_LEVELS

    def test_min_values_descending(self):
        mins = [CONFIDENCE_LEVELS[lvl]['min'] for lvl in ('alta', 'media', 'baixa', 'insuficiente')]
        assert mins == sorted(mins, reverse=True)


class TestForensicTerms:
    def test_required_keys(self):
        for key in ('positive', 'probable', 'possible', 'uncertain', 'negative', 'avoid'):
            assert key in FORENSIC_TERMS

    def test_avoid_is_list(self):
        assert isinstance(FORENSIC_TERMS['avoid'], list)


class TestMaxUniquePlates:
    def test_is_int(self):
        assert isinstance(MAX_UNIQUE_PLATES, int)

    def test_positive(self):
        assert MAX_UNIQUE_PLATES > 0

    def test_default_value(self):
        assert MAX_UNIQUE_PLATES == 10


class TestCharConfidenceThreshold:
    def test_is_float(self):
        assert isinstance(CHAR_CONFIDENCE_THRESHOLD, float)

    def test_in_range(self):
        assert 0.0 < CHAR_CONFIDENCE_THRESHOLD <= 1.0

    def test_default_value(self):
        assert CHAR_CONFIDENCE_THRESHOLD == 0.70
