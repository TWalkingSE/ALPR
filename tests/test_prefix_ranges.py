# tests/test_prefix_ranges.py
"""
Invariantes da tabela única de faixas de prefixo por estado.

Esta tabela vivia duplicada em `src/validator.py` e `src/plate_patterns.py`,
com faixas divergentes entre as duas cópias — e ambas alimentam o mesmo score
composto de ranking de candidatos. Os testes abaixo travam a consolidação.
"""

import pytest

from src.constants import BRAZILIAN_PREFIX_RANGES, BRAZILIAN_UFS
from src.plate_patterns import PlateNgramModel
from src.validator import BRAZILIAN_STATE_PREFIXES, is_plausible_plate_prefix


class TestTableInvariants:
    def test_covers_all_27_ufs(self):
        assert len(BRAZILIAN_PREFIX_RANGES) == 27
        assert frozenset(BRAZILIAN_PREFIX_RANGES) == BRAZILIAN_UFS

    def test_uf_codes_are_two_uppercase_letters(self):
        for uf in BRAZILIAN_PREFIX_RANGES:
            assert len(uf) == 2 and uf.isupper() and uf.isalpha(), f'UF inválida: {uf}'

    @pytest.mark.parametrize('uf', sorted(BRAZILIAN_PREFIX_RANGES))
    def test_ranges_are_well_formed(self, uf):
        ranges = BRAZILIAN_PREFIX_RANGES[uf]
        assert ranges, f'{uf} sem nenhuma faixa'
        for start, end in ranges:
            assert len(start) == 3 and start.isalpha() and start.isupper()
            assert len(end) == 3 and end.isalpha() and end.isupper()
            assert start <= end, f'{uf}: faixa invertida {start}-{end}'


class TestSingleSourceOfTruth:
    def test_validator_alias_points_to_shared_table(self):
        assert BRAZILIAN_STATE_PREFIXES is BRAZILIAN_PREFIX_RANGES

    def test_ngram_model_enumerates_from_shared_table(self):
        model = PlateNgramModel()
        # Um prefixo de cada extremo de faixa deve estar enumerado.
        for uf, ranges in BRAZILIAN_PREFIX_RANGES.items():
            start, end = ranges[0]
            assert start in model._valid_prefixes, f'{uf}: {start} ausente'
            assert end in model._valid_prefixes, f'{uf}: {end} ausente'

    def test_both_scorers_agree_on_previously_divergent_prefixes(self):
        """Prefixos que as duas tabelas classificavam de forma oposta.

        Ex.: 'OBA' existia só em plate_patterns (AP/PI) e 'KMA' só lá (PE);
        o validador os tratava como desconhecidos, rebaixando leituras válidas.
        """
        model = PlateNgramModel()
        for prefix in ('OBA', 'KMA', 'NOA', 'OCA', 'JSA', 'NCA'):
            assert is_plausible_plate_prefix(prefix + '1B23') == 1.0, prefix
            assert prefix in model._valid_prefixes, prefix
