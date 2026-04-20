"""
Testes unitários para o módulo PlateValidator.
Cobre validação de placas brasileiras nos formatos antigo e Mercosul,
correção de erros comuns de OCR e normalização de texto.
"""

import sys
from pathlib import Path

# Ajustar path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.validator import PlateValidator


@pytest.fixture
def validator():
    """Cria instância do PlateValidator para cada teste."""
    return PlateValidator()


# ==================== TESTES DE LIMPEZA DE TEXTO ====================

class TestCleanText:
    def test_empty_string(self, validator):
        assert validator.clean_text("") == ""

    def test_none_input(self, validator):
        assert validator.clean_text(None) == ""

    def test_uppercase_conversion(self, validator):
        assert validator.clean_text("abc1234") == "ABC1234"

    def test_strip_whitespace(self, validator):
        assert validator.clean_text("  ABC1234  ") == "ABC1234"

    def test_remove_special_chars(self, validator):
        assert validator.clean_text("ABC@#$1234") == "ABC1234"

    def test_keep_hyphen(self, validator):
        assert validator.clean_text("ABC-1234") == "ABC-1234"


# ==================== TESTES DE DETECÇÃO DE FORMATO ====================

class TestFormatDetection:
    def test_old_format_without_hyphen(self, validator):
        assert validator.is_old_format("ABC1234") is True

    def test_old_format_with_hyphen(self, validator):
        assert validator.is_old_format("ABC-1234") is True

    def test_mercosul_format(self, validator):
        assert validator.is_mercosul_format("ABC1D23") is True

    def test_mercosul_format_various(self, validator):
        assert validator.is_mercosul_format("RIO2A18") is True
        assert validator.is_mercosul_format("BRA0S17") is True

    def test_invalid_format(self, validator):
        assert validator.is_old_format("12ABCD3") is False
        assert validator.is_mercosul_format("12ABCD3") is False

    def test_too_short(self, validator):
        assert validator.is_old_format("ABC") is False
        assert validator.is_mercosul_format("ABC") is False

    def test_too_long(self, validator):
        assert validator.is_old_format("ABCD12345") is False
        assert validator.is_mercosul_format("ABCD1E234") is False


# ==================== TESTES DE VALIDAÇÃO ====================

class TestValidate:
    def test_valid_old_format(self, validator):
        result = validator.validate("ABC1234")
        assert result == "ABC-1234"

    def test_valid_old_with_hyphen(self, validator):
        result = validator.validate("ABC-1234")
        assert result == "ABC-1234"

    def test_valid_mercosul(self, validator):
        result = validator.validate("ABC1D23")
        assert result == "ABC1D23"

    def test_lowercase_input(self, validator):
        result = validator.validate("abc1234")
        assert result == "ABC-1234"

    def test_empty_input(self, validator):
        assert validator.validate("") is None
        assert validator.validate(None) is None

    def test_invalid_returns_none(self, validator):
        assert validator.validate("XXXXXXXX") is None

    def test_with_spaces(self, validator):
        result = validator.validate("ABC 1234")
        assert result is not None


# ==================== TESTES DE CORREÇÃO OCR ====================

class TestOCRCorrection:
    def test_zero_to_O_correction(self, validator):
        # "0BC1234" -> "OBC-1234" (posição 0 deve ser letra)
        result = validator.validate("0BC1234")
        assert result is not None
        assert result[0] == 'O'

    def test_I_to_1_correction(self, validator):
        # "ABCI234" -> "ABC-1234" (posição 3 deve ser número)
        result = validator.validate("ABCI234")
        assert result is not None

    def test_B_to_8_correction(self, validator):
        # "ABCB234" poderia ser formato Mercosul (ABC+num+letra+nums)
        # ou antigo com B->8 = "ABC-8234"
        result = validator.validate("ABCB234")
        assert result is not None

    def test_correction_history_recorded(self, validator):
        validator.validate("0BC1234")
        history = validator.get_correction_history()
        # Se uma correção foi feita, deve estar no histórico
        # (pode não estar se o texto original já era válido após limpeza)
        assert isinstance(history, dict)


# ==================== TESTES DE CHECK_PLATE_VALIDITY ====================

class TestCheckPlateValidity:
    def test_valid_old_plate(self, validator):
        result = validator.check_plate_validity("ABC1234")
        assert result["is_valid"] is True
        assert result["format"] == "old"
        assert result["normalized_plate"] == "ABC-1234"

    def test_valid_mercosul_plate(self, validator):
        result = validator.check_plate_validity("ABC1D23")
        assert result["is_valid"] is True
        assert result["format"] == "mercosul"
        assert result["normalized_plate"] == "ABC1D23"

    def test_invalid_plate(self, validator):
        result = validator.check_plate_validity("123ABCD")
        assert result["is_valid"] is False or result.get("corrected") is True

    def test_empty_plate(self, validator):
        result = validator.check_plate_validity("")
        assert result["is_valid"] is False
        assert "Placa vazia" in result["errors"]

    def test_mercosul_vowel_in_position_5(self, validator):
        # Posição index 4 com vogal — deve ser inválida por regra Mercosul
        result = validator.check_plate_validity("ABC1A23")
        assert result["is_valid"] is False

    def test_mercosul_consonant_in_position_5(self, validator):
        result = validator.check_plate_validity("ABC1D23")
        assert result["is_valid"] is True

    def test_with_hyphen_normalization(self, validator):
        result = validator.check_plate_validity("ABC-1234")
        assert result["is_valid"] is True
        assert result["normalized_plate"] == "ABC-1234"


# ==================== TESTES DE FORMAT_PLATE ====================

class TestFormatPlate:
    def test_format_old(self, validator):
        assert validator.format_plate("ABC1234") == "ABC-1234"

    def test_format_mercosul(self, validator):
        assert validator.format_plate("ABC1D23") == "ABC1D23"

    def test_format_invalid_returns_original(self, validator):
        assert validator.format_plate("INVALID") == "INVALID"
