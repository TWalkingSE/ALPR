# src/validator.py
"""
Módulo para validação de placas brasileiras nos padrões antigo (AAA-1234)
e Mercosul (AAA1B23) com suporte a regras específicas de validação.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .constants import (
    MERCOSUL_FORMAT_BIAS,
    OCR_CHAR_MAP,
    PLATE_PATTERN_MERCOSUL,
    PLATE_PATTERN_OLD,
    PLATE_PATTERN_OLD_HYPHEN,
    RE_MERCOSUL,
    RE_OLD,
    RE_OLD_HYPHEN,
    SIMILAR_CHARS,
    get_confusion_weight,
)

logger = logging.getLogger(__name__)


# Prefixos de placa por estado brasileiro (para ranking de sugestões)
# Fonte: sistema de registro veicular brasileiro (DENATRAN/SENATRAN)
BRAZILIAN_STATE_PREFIXES = {
    # Acre
    'AC': [('QTA', 'QTZ'), ('NAA', 'NBZ')],
    # Alagoas
    'AL': [('QRA', 'QSZ')],
    # Amapá
    'AP': [('QOA', 'QPZ')],
    # Amazonas
    'AM': [('QLA', 'QNZ')],
    # Bahia
    'BA': [('JAA', 'JZZ'), ('NUA', 'NZZ'), ('OAA', 'OAZ'), ('QCA', 'QDZ')],
    # Ceará
    'CE': [('HTA', 'HZZ'), ('QFA', 'QHZ')],
    # Distrito Federal
    'DF': [('JKA', 'JZZ'), ('OLA', 'OMZ'), ('QGA', 'QGZ')],
    # Espírito Santo
    'ES': [('MTA', 'MZZ'), ('PPA', 'PPZ'), ('QBA', 'QBZ')],
    # Goiás
    'GO': [('OAA', 'OZZ'), ('QEA', 'QEZ')],
    # Maranhão
    'MA': [('HSA', 'HSZ'), ('QIA', 'QIZ')],
    # Mato Grosso
    'MT': [('NGA', 'NTZ'), ('QJA', 'QKZ')],
    # Mato Grosso do Sul
    'MS': [('HRA', 'HRZ'), ('NCA', 'NFZ'), ('QQA', 'QQZ')],
    # Minas Gerais
    'MG': [('GKJ', 'HOK'), ('HAA', 'HQZ'), ('OUA', 'OZZ'), ('QVA', 'QZZ')],
    # Pará
    'PA': [('QOA', 'QPZ'), ('NNA', 'NNZ')],
    # Paraíba
    'PB': [('QOA', 'QOZ')],
    # Paraná
    'PR': [('AAA', 'BEZ'), ('AXA', 'BFZ'), ('QMA', 'QMZ')],
    # Pernambuco
    'PE': [('QPA', 'QPZ')],
    # Piauí
    'PI': [('QNA', 'QNZ')],
    # Rio de Janeiro
    'RJ': [('KMF', 'LVE'), ('LAA', 'LZZ'), ('QQA', 'QSZ')],
    # Rio Grande do Norte
    'RN': [('QOA', 'QOZ')],
    # Rio Grande do Sul
    'RS': [('IAA', 'JZZ'), ('ICA', 'IJZ'), ('QUA', 'QUZ')],
    # Rondônia
    'RO': [('NBA', 'NBZ'), ('QTA', 'QTZ')],
    # Roraima
    'RR': [('QOA', 'QOZ')],
    # Santa Catarina
    'SC': [('MAA', 'MSZ'), ('QJA', 'QJZ')],
    # São Paulo
    'SP': [('BFA', 'GKI'), ('CPA', 'GKZ'), ('QWA', 'QZZ')],
    # Sergipe
    'SE': [('QOA', 'QOZ')],
    # Tocantins
    'TO': [('QOA', 'QOZ')],
}


def is_plausible_plate_prefix(text: str) -> float:
    """
    Retorna um score de plausibilidade (0.0-1.0) baseado no prefixo.
    Prefixos que existem no sistema brasileiro recebem score mais alto.
    
    Args:
        text: Texto da placa (mínimo 3 chars)
    
    Returns:
        Score entre 0.0 e 1.0
    """
    if len(text) < 3:
        return 0.5  # Neutro

    prefix = text[:3].upper()

    # Verificar se o prefixo cai em alguma faixa de estado
    for _state, ranges in BRAZILIAN_STATE_PREFIXES.items():
        for (start, end) in ranges:
            if start <= prefix <= end:
                return 1.0  # Prefixo válido de estado

    # Prefixo não reconhecido mas pode ser válido (faixas não listadas)
    return 0.7

class PlateValidator:
    """Classe para validação de placas de veículos brasileiras."""

    # Padrões de placa — ponteiros para as constantes compartilhadas em
    # src/constants.py (fonte única de verdade). Expostos como atributos
    # de classe para retrocompatibilidade com testes e código externo.
    PATTERN_OLD = PLATE_PATTERN_OLD
    PATTERN_MERCOSUL = PLATE_PATTERN_MERCOSUL
    PATTERN_OLD_WITH_HYPHEN = PLATE_PATTERN_OLD_HYPHEN

    # Aliases para mapas de caracteres OCR (importados de constants)
    COMMON_OCR_ERRORS = OCR_CHAR_MAP
    MULTI_ALTERNATIVES = SIMILAR_CHARS

    def __init__(self):
        """Inicializa o validador de placas."""
        # Reutilizar os regex já compilados em constants.py (sem re-compilar)
        self.re_old = RE_OLD
        self.re_mercosul = RE_MERCOSUL
        self.re_old_with_hyphen = RE_OLD_HYPHEN

        # Histórico de correções para aprendizado
        self.correction_history: Dict[str, str] = {}

    def clean_text(self, text: str) -> str:
        """
        Limpa o texto removendo caracteres indesejados.
        
        Args:
            text: Texto a ser limpo
            
        Returns:
            Texto limpo
        """
        if not text:
            return ""

        # Remover espaços em branco e converter para maiúsculas
        text = text.strip().upper()

        # Remover caracteres especiais, exceto hífen para placas antigas
        text = re.sub(r'[^A-Z0-9\-]', '', text)

        return text

    def is_old_format(self, text: str) -> bool:
        """
        Verifica se o texto corresponde ao formato antigo de placas.
        
        Args:
            text: Texto a ser verificado
            
        Returns:
            True se for no formato antigo, False caso contrário
        """
        return bool(self.re_old.match(text) or self.re_old_with_hyphen.match(text))

    def is_mercosul_format(self, text: str) -> bool:
        """
        Verifica se o texto corresponde ao formato Mercosul de placas.
        
        Args:
            text: Texto a ser verificado
            
        Returns:
            True se for no formato Mercosul, False caso contrário
        """
        return bool(self.re_mercosul.match(text))

    def format_plate(self, text: str) -> str:
        """
        Formata o texto para o padrão de placa correspondente.
        
        Args:
            text: Texto a ser formatado
            
        Returns:
            Texto formatado como placa
        """
        # Remover todos os caracteres não alfanuméricos
        clean = re.sub(r'[^A-Z0-9]', '', text)

        # Formatar de acordo com o padrão
        if self.is_old_format(clean):
            return f"{clean[:3]}-{clean[3:]}"
        elif self.is_mercosul_format(clean):
            return clean

        return text  # Retorna o original se não corresponder a nenhum padrão

    def validate(self, text: str, format_hint: Optional[str] = None) -> Optional[str]:
        """
        Valida e formata uma placa brasileira.
        Testa AMBOS os formatos (old e mercosul) e escolhe o melhor com base
        em scoring ponderado por pesos de confusão.
        
        Args:
            text: Texto a ser validado
            format_hint: 'old', 'mercosul' ou None. Se fornecido (ex: via
                         detecção visual da faixa azul), dá preferência
                         forte a esse formato.
            
        Returns:
            Placa formatada ou None se inválida
        """
        if not text:
            return None

        # Limpar o texto
        clean_text = self.clean_text(text)

        # Armazenar o texto original para comparação
        original = clean_text

        # Normalizar: remover hífen/espaços para análise uniforme
        normalized = re.sub(r'[-\s]', '', clean_text)

        # === MATCH EXATO: já é válido em algum formato ===
        is_exact_old = bool(self.re_old.match(normalized))
        is_exact_merc = bool(self.re_mercosul.match(normalized)) and normalized[4] not in 'AEIOU'

        # Se o format_hint contradiz o match exato, tentar correção para o
        # formato hintado ANTES de aceitar o match exato.
        # Ex: OCR leu "ABC1023" (match old), mas hint='mercosul' → tentar 0→D → ABC1D23
        if format_hint and not (
            (format_hint == 'old' and is_exact_old) or
            (format_hint == 'mercosul' and is_exact_merc)
        ):
            hinted_correction = self._try_correction(normalized, format_hint=format_hint)
            if hinted_correction:
                self.correction_history[original] = hinted_correction
                return hinted_correction

        if is_exact_old and is_exact_merc:
            if format_hint == 'mercosul':
                return normalized
            elif format_hint == 'old':
                return f"{normalized[:3]}-{normalized[3:]}"
            else:
                return normalized  # Bias Mercosul
        elif is_exact_merc:
            return normalized
        elif is_exact_old:
            return f"{normalized[:3]}-{normalized[3:]}"

        # === CORREÇÃO: tentar ambos os formatos e escolher o melhor ===
        corrected = self._try_correction(normalized, format_hint=format_hint)
        if corrected:
            self.correction_history[original] = corrected
            return corrected

        return None

    def _try_correction(self, text: str, format_hint: Optional[str] = None) -> Optional[str]:
        """
        Tenta corrigir erros comuns em OCR de placas usando SIMILAR_CHARS
        (multi-alternativa) para maior cobertura.
        
        Sempre testa AMBOS os formatos e pontua cada correção usando
        CONFUSION_WEIGHTS para escolher a mais provável.
        
        Args:
            text: Texto a ser corrigido
            format_hint: 'old', 'mercosul' ou None
            
        Returns:
            Texto corrigido ou None se não for possível corrigir
        """
        if not text:
            return None

        # Verificar se o comprimento está próximo do esperado
        if len(text) >= 6 and len(text) <= 8:
            # Normalizar para 7 caracteres antes de tentar correção
            candidates_7 = self._normalize_to_7_chars(text)

            for candidate in candidates_7:
                # Tentar corrigir para AMBOS os formatos
                corrected_old, score_old = self._correct_to_format_scored(candidate, 'old')
                corrected_merc, score_merc = self._correct_to_format_scored(candidate, 'mercosul')

                # Aplicar bias Mercosul (maioria das placas desde 2018)
                if score_merc > 0:
                    score_merc *= MERCOSUL_FORMAT_BIAS

                # Aplicar format_hint como multiplicador forte
                if format_hint == 'mercosul' and score_merc > 0:
                    score_merc *= 1.5
                elif format_hint == 'old' and score_old > 0:
                    score_old *= 1.5

                # Aplicar plausibilidade do prefixo
                if corrected_old:
                    score_old *= (0.7 + 0.3 * is_plausible_plate_prefix(corrected_old))
                if corrected_merc:
                    score_merc *= (0.7 + 0.3 * is_plausible_plate_prefix(corrected_merc))

                # Escolher o melhor
                if corrected_merc and corrected_old:
                    if score_merc >= score_old:
                        return corrected_merc
                    else:
                        return f"{corrected_old[:3]}-{corrected_old[3:]}"
                elif corrected_merc:
                    return corrected_merc
                elif corrected_old:
                    return f"{corrected_old[:3]}-{corrected_old[3:]}"

                # Tentar substituições múltiplas com brute-force
                multi = self._try_multi_substitutions(candidate)
                if multi:
                    return multi

        return None

    def _correct_to_format(self, text: str, format_type: str) -> Optional[str]:
        """
        Tenta corrigir cada posição para o tipo esperado pelo formato.
        Usa SIMILAR_CHARS para encontrar a melhor substituição.
        
        Args:
            text: Texto com 7 caracteres
            format_type: 'old' ou 'mercosul'
            
        Returns:
            Texto corrigido ou None
        """
        result, _ = self._correct_to_format_scored(text, format_type)
        return result

    def _correct_to_format_scored(self, text: str, format_type: str) -> Tuple[Optional[str], float]:
        """
        Tenta corrigir cada posição para o tipo esperado pelo formato.
        Retorna o texto corrigido E um score de qualidade baseado nos pesos
        de confusão de cada substituição feita.
        
        Args:
            text: Texto com 7 caracteres
            format_type: 'old' ou 'mercosul'
            
        Returns:
            (texto_corrigido, score) ou (None, 0.0)
        """
        if format_type == 'old':
            expected = ['L', 'L', 'L', 'N', 'N', 'N', 'N']  # AAA-1234
        else:
            expected = ['L', 'L', 'L', 'N', 'L', 'N', 'N']  # AAA1A23

        corrected = []
        total_score = 1.0  # Score multiplicativo

        for i, (char, exp_type) in enumerate(zip(text, expected, strict=False)):
            if exp_type == 'L' and char.isalpha():
                # Para Mercosul posição 4: rejeitar vogais mesmo que já seja letra
                if format_type == 'mercosul' and i == 4 and char in 'AEIOU':
                    replacement, weight = self._find_best_replacement_weighted(
                        char, exp_type, exclude_vowels=True
                    )
                    if replacement:
                        corrected.append(replacement)
                        total_score *= weight
                    else:
                        return None, 0.0
                else:
                    corrected.append(char)
            elif exp_type == 'N' and char.isdigit():
                corrected.append(char)
            else:
                # Caractere não bate com tipo esperado — buscar substituição
                # Para Mercosul posição 4: excluir vogais das alternativas
                exclude_vowels = (format_type == 'mercosul' and i == 4 and exp_type == 'L')
                replacement, weight = self._find_best_replacement_weighted(
                    char, exp_type, exclude_vowels=exclude_vowels
                )
                if replacement:
                    corrected.append(replacement)
                    total_score *= weight  # Penalizar pela confusão
                else:
                    return None, 0.0  # Impossível corrigir esta posição

        result = ''.join(corrected)

        # Validar resultado
        if format_type == 'old' and self.is_old_format(result):
            return result, total_score
        elif format_type == 'mercosul' and self.is_mercosul_format(result):
            if result[4] not in 'AEIOU':
                return result, total_score

        return None, 0.0

    def _find_best_replacement_weighted(self, char: str, expected_type: str, exclude_vowels: bool = False) -> Tuple[Optional[str], float]:
        """
        Encontra a melhor substituição para um caractere, rankeada por peso
        de confusão OCR. Retorna o par mais provável em vez do primeiro encontrado.
        
        Args:
            char: Caractere original
            expected_type: 'L' para letra, 'N' para dígito
            exclude_vowels: Excluir vogais das alternativas
            
        Returns:
            (caractere_substituto, peso_confusao) ou (None, 0.0)
        """
        candidates = []  # [(replacement_char, confusion_weight)]

        # Coletar TODAS as alternativas válidas com seus pesos
        alternatives = self.MULTI_ALTERNATIVES.get(char, [])
        for alt in alternatives:
            if (expected_type == 'L' and alt.isalpha() and (not exclude_vowels or alt not in 'AEIOU')) or (expected_type == 'N' and alt.isdigit()):
                weight = get_confusion_weight(char, alt)
                candidates.append((alt, weight))

        # Também verificar OCR_CHAR_MAP como fonte adicional
        direct = self.COMMON_OCR_ERRORS.get(char)
        if direct:
            if (expected_type == 'L' and direct.isalpha() and (not exclude_vowels or direct not in 'AEIOU')) or (expected_type == 'N' and direct.isdigit()):
                weight = get_confusion_weight(char, direct)
                if not any(c == direct for c, _ in candidates):
                    candidates.append((direct, weight))

        if not candidates:
            return None, 0.0

        # Ordenar por peso de confusão decrescente (mais provável primeiro)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0]

    def _normalize_to_7_chars(self, text: str) -> List[str]:
        """
        Normaliza texto de 6 ou 8 caracteres para candidatos de 7 caracteres.
        
        Para 6 chars: tenta inserir um caractere plausível em cada posição.
        Para 8 chars: tenta remover um caractere de cada posição.
        Para 7 chars: retorna o próprio texto.
        
        Args:
            text: Texto com 6-8 caracteres
            
        Returns:
            Lista de candidatos com 7 caracteres, ordenados por plausibilidade
        """
        if len(text) == 7:
            return [text]

        candidates = []

        if len(text) == 8:
            # Tentar remover cada posição e verificar se vira placa válida
            for i in range(8):
                candidate = text[:i] + text[i+1:]
                score = self._score_plate_candidate(candidate)
                candidates.append((candidate, score))

        elif len(text) == 6:
            # Posições esperadas nos dois formatos
            formats = [
                ['L', 'L', 'L', 'N', 'N', 'N', 'N'],  # Antigo
                ['L', 'L', 'L', 'N', 'L', 'N', 'N'],  # Mercosul
            ]

            for expected in formats:
                for insert_pos in range(7):
                    # Determinar qual tipo inserir nesta posição
                    exp_type = expected[insert_pos]

                    # Inferir o caractere mais provável baseado em vizinhos
                    inferred = self._infer_missing_char(text, insert_pos, exp_type)
                    if inferred:
                        candidate = text[:insert_pos] + inferred + text[insert_pos:]
                        if len(candidate) == 7:
                            score = self._score_plate_candidate(candidate)
                            candidates.append((candidate, score))

        if not candidates:
            return []

        # Ordenar por score decrescente e remover duplicatas
        candidates.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        unique = []
        for c, _score in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        # Retornar top 5 candidatos mais plausíveis
        return unique[:5]

    def _score_plate_candidate(self, text: str) -> float:
        """
        Pontua um candidato de 7 caracteres pela plausibilidade como placa.
        
        Args:
            text: Texto com 7 caracteres
            
        Returns:
            Score entre 0.0 e 1.0
        """
        if len(text) != 7:
            return 0.0

        score = 0.0

        # Formato válido perfeito
        if self.is_old_format(text):
            score = 1.0
        elif self.is_mercosul_format(text):
            score = 1.0 if text[4] not in 'AEIOU' else 0.8
        else:
            # Verificar quantas posições batem com cada formato
            old_match = sum(1 for i, c in enumerate(text) if
                          (i < 3 and c.isalpha()) or (i >= 3 and c.isdigit()))
            merc_match = sum(1 for i, c in enumerate(text) if
                           (i in (0,1,2,4) and c.isalpha()) or (i in (3,5,6) and c.isdigit()))
            score = max(old_match, merc_match) / 7.0 * 0.7

        # Bonus por prefixo plausível
        prefix_score = is_plausible_plate_prefix(text)
        score = score * 0.8 + prefix_score * 0.2

        return score

    def _infer_missing_char(self, text: str, insert_pos: int, expected_type: str) -> Optional[str]:
        """
        Infere um caractere faltante baseado no contexto e tipo esperado.
        
        Args:
            text: Texto com 6 caracteres
            insert_pos: Posição onde inserir (0-6)
            expected_type: 'L' para letra, 'N' para dígito
            
        Returns:
            Caractere inferido ou None
        """
        # Verificar se o caractere antes/depois da posição de inserção
        # poderia estar "duplicado" ou ser um similar

        # Se a posição de inserção é entre dois caracteres,
        # tentar duplicar o vizinho se ele é do tipo correto
        if insert_pos > 0 and insert_pos <= len(text):
            prev_char = text[insert_pos - 1]
            if (expected_type == 'L' and prev_char.isalpha()) or (expected_type == 'N' and prev_char.isdigit()):
                return prev_char

        if insert_pos < len(text):
            next_char = text[insert_pos]
            if (expected_type == 'L' and next_char.isalpha()) or (expected_type == 'N' and next_char.isdigit()):
                return next_char

        # Fallback: retornar caractere neutro
        if expected_type == 'L':
            return 'A'
        return None

    def _find_best_replacement(self, char: str, expected_type: str) -> Optional[str]:
        """
        Encontra a melhor substituição para um caractere usando SIMILAR_CHARS.
        Wrapper de compatibilidade para _find_best_replacement_weighted.
        """
        replacement, _ = self._find_best_replacement_weighted(char, expected_type)
        return replacement

    def _try_multi_substitutions(self, text: str) -> Optional[str]:
        """
        Tenta múltiplas combinações usando SIMILAR_CHARS (multi-alternativa).
        Usa brute-force controlado com até 4 posições ambíguas.
        
        Args:
            text: Texto a ser corrigido
            
        Returns:
            Texto corrigido ou None
        """
        if len(text) < 7:
            return None

        # Encontrar posições ambíguas usando SIMILAR_CHARS
        ambiguous_positions = []
        for i, char in enumerate(text):
            if char in self.MULTI_ALTERNATIVES:
                ambiguous_positions.append((i, char))

        # Limitar a 4 posições (mais abrangente que o antigo limite de 3)
        if len(ambiguous_positions) > 4:
            ambiguous_positions = ambiguous_positions[:4]

        if not ambiguous_positions:
            return None

        # Gerar combinações e verificar validade
        return self._generate_multi_combinations(text, ambiguous_positions)

    def _generate_multi_combinations(self, text: str, ambiguous_positions: List[Tuple[int, str]]) -> Optional[str]:
        """
        Gera combinações usando SIMILAR_CHARS e verifica se alguma é válida.
        Prioriza resultados por plausibilidade do prefixo.
        
        Args:
            text: Texto original
            ambiguous_positions: Lista de posições ambíguas e caracteres
            
        Returns:
            Primeira placa válida mais plausível encontrada ou None
        """
        # Caso base: sem posições ambíguas
        if not ambiguous_positions:
            if self.is_old_format(text):
                return f"{text[:3]}-{text[3:]}"
            elif self.is_mercosul_format(text):
                # Verificar regra de vogal na posição 5
                if text[4] not in 'AEIOU':
                    return text
            return None

        # Pegar a primeira posição ambígua
        pos, char = ambiguous_positions[0]
        remaining = ambiguous_positions[1:]

        # Testar todas as alternativas (original + similares)
        alternatives = [char]
        if char in self.MULTI_ALTERNATIVES:
            alternatives.extend(self.MULTI_ALTERNATIVES[char])
        # Remover duplicatas
        alternatives = list(dict.fromkeys(alternatives))

        # Coletar todos os resultados válidos para ranking
        valid_results = []

        for alt in alternatives:
            text_alt = text[:pos] + alt + text[pos+1:]
            result = self._generate_multi_combinations(text_alt, remaining)
            if result:
                valid_results.append(result)

        if not valid_results:
            return None

        # Rankear por plausibilidade do prefixo
        if len(valid_results) > 1:
            scored = [(r, is_plausible_plate_prefix(r.replace('-', ''))) for r in valid_results]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[0][0]

        return valid_results[0]

    def get_correction_history(self) -> Dict[str, str]:
        """
        Retorna o histórico de correções.
        
        Returns:
            Dicionário com textos originais e corrigidos
        """
        return self.correction_history

    def _format_fit_score(self, text: str, format_type: str) -> float:
        clean = re.sub(r'[-\s]', '', self.clean_text(text))
        if not clean:
            return 0.0

        expected = ['L', 'L', 'L', 'N', 'N', 'N', 'N']
        if format_type == 'mercosul':
            expected = ['L', 'L', 'L', 'N', 'L', 'N', 'N']

        comparisons = min(len(clean), len(expected))
        matches = 0
        for index in range(comparisons):
            char = clean[index]
            if expected[index] == 'L' and char.isalpha():
                matches += 1
            elif expected[index] == 'N' and char.isdigit():
                matches += 1

        score = matches / len(expected)
        if format_type == 'mercosul' and len(clean) > 4 and clean[4] in 'AEIOU':
            score *= 0.75
        return float(score)

    def describe_validation(self, plate: str, format_hint: Optional[str] = None) -> Dict[str, Any]:
        clean = self.clean_text(plate)
        normalized = re.sub(r'[-\s]', '', clean)
        suggested_plate = self.validate(plate, format_hint=format_hint)
        validity = self.check_plate_validity(plate, format_hint=format_hint)
        normalized_suggested = re.sub(r'[-\s]', '', suggested_plate or '')
        exact_old = bool(self.re_old.match(normalized) or self.re_old_with_hyphen.match(clean))
        exact_mercosul = bool(self.re_mercosul.match(normalized)) and (len(normalized) <= 4 or normalized[4] not in 'AEIOU')

        issues = list(validity.get('errors', []))
        if len(normalized) != 7:
            issues.append('length_outside_expected')
        if suggested_plate and normalized_suggested != normalized:
            issues.append('corrected_from_ocr_noise')
        if not suggested_plate:
            issues.append('unresolved_plate')

        deduped_issues = []
        seen = set()
        for issue in issues:
            if issue not in seen:
                deduped_issues.append(issue)
                seen.add(issue)

        return {
            'input': plate,
            'cleaned': clean,
            'normalized_input': normalized,
            'suggested_plate': suggested_plate or '',
            'normalized_plate': validity.get('normalized_plate') or suggested_plate or '',
            'is_valid': bool(validity.get('is_valid', False) and suggested_plate),
            'format': validity.get('format', 'unknown'),
            'format_hint': format_hint or '',
            'exact_match_old': exact_old,
            'exact_match_mercosul': exact_mercosul,
            'correction_applied': bool(suggested_plate and normalized_suggested != normalized),
            'prefix_score': is_plausible_plate_prefix(normalized),
            'old_format_score': self._format_fit_score(normalized, 'old'),
            'mercosul_format_score': self._format_fit_score(normalized, 'mercosul'),
            'issues': deduped_issues,
        }

    def check_plate_validity(self, plate: str, format_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Verifica a validade de uma placa, incluindo regras específicas.
        Testa Mercosul ANTES de old format para evitar classificação incorreta.
        
        Args:
            plate: Texto da placa a ser verificada
            format_hint: 'old', 'mercosul' ou None (detecção visual)
            
        Returns:
            Dicionário com informações de validade
        """
        result = {
            "is_valid": False,
            "format": "unknown",
            "errors": [],
            "normalized_plate": None
        }

        if not plate:
            result["errors"].append("Placa vazia")
            return result

        # Limpar o texto
        clean = self.clean_text(plate)
        normalized = re.sub(r'[-\s]', '', clean)

        # Verificar formato: testar Mercosul PRIMEIRO
        # (evita que placas Mercosul sejam classificadas como antigas)
        is_merc = self.is_mercosul_format(normalized)
        is_old = self.is_old_format(clean) or self.re_old_with_hyphen.match(clean)

        if is_merc and is_old:
            # Ambigüidade: usar hint ou bias Mercosul
            if format_hint == 'old':
                chosen_format = 'old'
            else:
                chosen_format = 'mercosul'  # Bias Mercosul
        elif is_merc:
            chosen_format = 'mercosul'
        elif is_old:
            chosen_format = 'old'
        else:
            chosen_format = None

        if chosen_format == 'mercosul':
            result["format"] = "mercosul"
            result["normalized_plate"] = normalized
            if normalized[4] in "AEIOU":
                result["errors"].append("No formato Mercosul, a letra na 5ª posição não pode ser vogal")
                result["is_valid"] = False
            else:
                result["is_valid"] = True
        elif chosen_format == 'old':
            norm_clean = re.sub(r'[-\s]', '', clean)
            result["format"] = "old"
            result["normalized_plate"] = f"{norm_clean[:3]}-{norm_clean[3:]}"
            result["is_valid"] = True
        else:
            # Tentar corrigir antes de invalidar (com hint)
            corrected = self._try_correction(normalized, format_hint=format_hint)
            if corrected:
                if "-" in corrected:
                    result["format"] = "old"
                else:
                    result["format"] = "mercosul"
                result["normalized_plate"] = corrected
                result["is_valid"] = True
                result["corrected"] = True
                result["original"] = plate
            else:
                result["errors"].append("Formato inválido de placa")

        return result
