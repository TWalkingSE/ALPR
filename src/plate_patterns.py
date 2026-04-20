# src/plate_patterns.py
"""
Modelo estatístico de n-gram para placas brasileiras.

Calcula a probabilidade de uma sequência de caracteres ser uma placa real
baseado em padrões posicionais aprendidos das faixas de placas brasileiras.

Features:
  - Probabilidade posicional: P(char | posição)
  - Bi-grams posicionais: P(char_i | char_{i-1}, posição)
  - Validação de prefixo por estado
  - Score composto para ranking de candidatos

Não requer dados externos: as distribuições são derivadas das faixas
oficiais de prefixos dos estados brasileiros (DENATRAN/SENATRAN).
"""

import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ==================== FAIXAS DE PLACAS POR ESTADO ====================
# Fonte: Registro Nacional de Veículos Automotores (RENAVAM)
# Formato: (inicio, fim) — faixa de 3 letras de prefixo

BRAZILIAN_PREFIX_RANGES = {
    'AC': [('NAA', 'NBZ'), ('QTA', 'QTZ')],
    'AL': [('QRA', 'QSZ')],
    'AP': [('OBA', 'OBZ')],
    'AM': [('JWA', 'JZZ'), ('NOA', 'NQZ')],
    'BA': [('JAA', 'JVZ'), ('NUA', 'NZZ'), ('OAA', 'OAZ'), ('QCA', 'QDZ')],
    'CE': [('HTA', 'HZZ'), ('NCA', 'NFZ'), ('QFA', 'QHZ')],
    'DF': [('JKA', 'JMZ'), ('OLA', 'OMZ'), ('QGA', 'QGZ')],
    'ES': [('MTA', 'MZZ'), ('PPA', 'PPZ'), ('QBA', 'QBZ')],
    'GO': [('NNA', 'NTZ'), ('OCA', 'OKZ'), ('QEA', 'QEZ')],
    'MA': [('HSA', 'HSZ'), ('NGA', 'NMZ'), ('QIA', 'QIZ')],
    'MT': [('JWA', 'JZZ'), ('NGA', 'NTZ'), ('QJA', 'QKZ')],
    'MS': [('HRA', 'HRZ'), ('NCA', 'NFZ'), ('QQA', 'QQZ')],
    'MG': [('GKJ', 'HOK'), ('HAA', 'HQZ'), ('OUA', 'OZZ'), ('QVA', 'QZZ')],
    'PA': [('NNA', 'NNZ'), ('QOA', 'QPZ')],
    'PB': [('OCA', 'OKZ'), ('QOA', 'QOZ')],
    'PR': [('AAA', 'BEZ'), ('AXA', 'BFZ'), ('QMA', 'QMZ')],
    'PE': [('KMA', 'LVE'), ('QPA', 'QPZ')],
    'PI': [('OBA', 'OBZ'), ('QNA', 'QNZ')],
    'RJ': [('KMF', 'LVE'), ('LAA', 'LZZ'), ('QQA', 'QSZ')],
    'RN': [('NNA', 'NNZ'), ('QOA', 'QOZ')],
    'RS': [('IAA', 'JJZ'), ('ICA', 'IJZ'), ('QUA', 'QUZ')],
    'RO': [('NBA', 'NBZ'), ('QTA', 'QTZ')],
    'RR': [('NCA', 'NCZ'), ('QOA', 'QOZ')],
    'SC': [('MAA', 'MSZ'), ('QJA', 'QJZ')],
    'SP': [('BFA', 'GKI'), ('CPA', 'GKZ'), ('QWA', 'QZZ')],
    'SE': [('JSA', 'JTZ'), ('QOA', 'QOZ')],
    'TO': [('NBA', 'NBZ'), ('QOA', 'QOZ')],
}


class PlateNgramModel:
    """
    Modelo estatístico de n-gram para avaliação de placas brasileiras.
    
    Calcula a plausibilidade de um texto ser uma placa real brasileira
    baseado em padrões posicionais derivados das faixas oficiais.
    
    Uso:
        model = PlateNgramModel()
        score = model.score_plate("DRO1J05")  # → 0.82
        scores = model.rank_candidates(["DRO1J05", "DRO1305", "DRO1B05"])
    """

    def __init__(self, enabled: bool = True):
        """
        Inicializa o modelo com distribuições derivadas das faixas oficiais.
        """
        self.enabled = enabled

        # Distribuição posicional: P(char | posição)
        self._positional_dist: Dict[int, Dict[str, float]] = {}

        # Bi-gram posicional: P(char_i | char_{i-1}, posição)
        self._bigram_dist: Dict[int, Dict[str, Dict[str, float]]] = {}

        # Prefixos válidos (3 letras)
        self._valid_prefixes: set = set()

        # Estado por prefixo
        self._prefix_to_state: Dict[str, str] = {}

        if enabled:
            self._build_model()
            logger.info(
                f"PlateNgramModel inicializado: "
                f"{len(self._valid_prefixes)} prefixos, "
                f"{len(self._positional_dist)} distribuições posicionais"
            )
        else:
            logger.info("PlateNgramModel: desabilitado")

    def _build_model(self):
        """Constrói as distribuições a partir das faixas de prefixos."""
        # 1. Enumerar todos os prefixos válidos
        self._enumerate_prefixes()

        # 2. Construir distribuição posicional para posições 0-2
        self._build_positional_distributions()

        # 3. Construir bi-grams para posições 0-2
        self._build_bigram_distributions()

        # 4. Para posições 3-6, usar distribuição uniforme (dígitos) / consoantes (pos 4)
        self._build_digit_distributions()

    def _enumerate_prefixes(self):
        """Enumera todos os prefixos de 3 letras válidos."""
        for state, ranges in BRAZILIAN_PREFIX_RANGES.items():
            for (start, end) in ranges:
                # Gerar todos os prefixos na faixa
                for a in range(ord(start[0]), ord(end[0]) + 1):
                    for b in range(ord(start[1]) if chr(a) == start[0] else ord('A'),
                                   (ord(end[1]) if chr(a) == end[0] else ord('Z')) + 1):
                        for c in range(ord(start[2]) if (chr(a) == start[0] and chr(b) == start[1]) else ord('A'),
                                       (ord(end[2]) if (chr(a) == end[0] and chr(b) == end[1]) else ord('Z')) + 1):
                            prefix = chr(a) + chr(b) + chr(c)
                            if start <= prefix <= end:
                                self._valid_prefixes.add(prefix)
                                self._prefix_to_state[prefix] = state

    def _build_positional_distributions(self):
        """Constrói P(char | posição) para posições 0, 1, 2."""
        for pos in range(3):
            char_counts = defaultdict(int)
            total = 0

            for prefix in self._valid_prefixes:
                char_counts[prefix[pos]] += 1
                total += 1

            # Suavização de Laplace
            dist = {}
            for ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                dist[ch] = (char_counts[ch] + 1) / (total + 26)

            self._positional_dist[pos] = dist

    def _build_bigram_distributions(self):
        """Constrói P(char_i | char_{i-1}, posição) para posições 1 e 2."""
        for pos in [1, 2]:
            bigram_counts = defaultdict(lambda: defaultdict(int))
            context_totals = defaultdict(int)

            for prefix in self._valid_prefixes:
                prev_char = prefix[pos - 1]
                curr_char = prefix[pos]
                bigram_counts[prev_char][curr_char] += 1
                context_totals[prev_char] += 1

            dist = {}
            for prev in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                dist[prev] = {}
                total = context_totals[prev] + 26  # Laplace
                for curr in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    dist[prev][curr] = (bigram_counts[prev][curr] + 1) / total

            self._bigram_dist[pos] = dist

    def _build_digit_distributions(self):
        """Distribuições para posições de dígitos (3, 5, 6) e letra Mercosul (4)."""
        # Dígitos: distribuição uniforme (qualquer dígito 0-9 é válido)
        uniform_digit = {str(d): 0.1 for d in range(10)}

        self._positional_dist[3] = uniform_digit.copy()
        self._positional_dist[5] = uniform_digit.copy()
        self._positional_dist[6] = uniform_digit.copy()

        # Posição 4: depende do formato
        # Para old format: dígito
        # Para Mercosul: consoante (excluir vogais)
        # Usar distribuição mista (ponderada 60% Mercosul, 40% old)
        pos4_dist = {}

        # Consoantes (Mercosul)
        consonants = set('BCDFGHJKLMNPQRSTVWXYZ')
        total_consonants = len(consonants)

        for ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            if ch in consonants:
                pos4_dist[ch] = 0.6 / total_consonants
            else:
                pos4_dist[ch] = 0.001  # Vogais muito improváveis

        for d in range(10):
            pos4_dist[str(d)] = 0.4 / 10  # Dígitos (old format)

        self._positional_dist[4] = pos4_dist

    def score_plate(self, text: str) -> float:
        """
        Calcula score de plausibilidade para um texto de placa.
        
        Combina:
        - Score posicional (P(char | posição))
        - Score bi-gram (P(char_i | char_{i-1}))
        - Bonus por prefixo válido
        - Bonus por formato válido (regex)
        
        Args:
            text: Texto da placa (7 caracteres, sem hífen)
            
        Returns:
            Score entre 0.0 e 1.0
        """
        if not self.enabled:
            return 0.5

        clean = text.replace('-', '').upper()
        if len(clean) != 7:
            return 0.0

        # 1. Score posicional unigram
        log_prob = 0.0
        for pos in range(7):
            ch = clean[pos]
            dist = self._positional_dist.get(pos, {})
            prob = dist.get(ch, 1e-6)
            log_prob += math.log(prob + 1e-10)

        # 2. Score bi-gram (posições 1 e 2)
        for pos in [1, 2]:
            prev = clean[pos - 1]
            curr = clean[pos]
            bigram = self._bigram_dist.get(pos, {}).get(prev, {})
            prob = bigram.get(curr, 1e-6)
            log_prob += math.log(prob + 1e-10)

        # Normalizar log_prob para 0-1
        # Score máximo teórico ~= -5 (todas probabilidades altas)
        # Score mínimo teórico ~= -60 (todas probabilidades muito baixas)
        max_log = -3.0
        min_log = -50.0
        normalized = (log_prob - min_log) / (max_log - min_log)
        normalized = max(0.0, min(1.0, normalized))

        # 3. Bonus por prefixo válido
        prefix = clean[:3]
        if prefix in self._valid_prefixes:
            normalized = min(1.0, normalized + 0.15)

        # 4. Bonus por formato válido
        import re
        if re.match(r'^[A-Z]{3}[0-9]{4}$', clean):
            normalized = min(1.0, normalized + 0.05)
        elif re.match(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$', clean):
            if clean[4] not in 'AEIOU':
                normalized = min(1.0, normalized + 0.07)  # Mercosul ligeiramente preferido
            else:
                normalized -= 0.05  # Vogal na posição 4 penaliza

        return normalized

    def rank_candidates(
        self,
        candidates: List[str],
        ocr_confidences: Optional[List[float]] = None,
        ocr_weight: float = 0.3,
    ) -> List[Tuple[str, float]]:
        """
        Classifica candidatos de placa por plausibilidade.
        
        Args:
            candidates: Lista de textos candidatos
            ocr_confidences: Confiança OCR de cada candidato (opcional)
            ocr_weight: Peso da confiança OCR no score final
            
        Returns:
            Lista de (texto, score_combinado) ordenada por score decrescente
        """
        if not self.enabled:
            return [(c, 0.5) for c in candidates]

        scored = []
        for i, text in enumerate(candidates):
            ngram_score = self.score_plate(text)

            if ocr_confidences and i < len(ocr_confidences):
                ocr_conf = ocr_confidences[i]
                combined = (1 - ocr_weight) * ngram_score + ocr_weight * ocr_conf
            else:
                combined = ngram_score

            scored.append((text, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def get_prefix_state(self, text: str) -> Optional[str]:
        """
        Retorna o estado brasileiro associado ao prefixo da placa.
        
        Args:
            text: Texto da placa (mínimo 3 caracteres)
            
        Returns:
            Sigla do estado ou None se prefixo não reconhecido
        """
        if len(text) < 3:
            return None
        prefix = text[:3].upper()
        return self._prefix_to_state.get(prefix)

    def is_valid_prefix(self, text: str) -> bool:
        """Verifica se o prefixo pertence a uma faixa válida."""
        if len(text) < 3:
            return False
        return text[:3].upper() in self._valid_prefixes

    def get_likely_alternatives(
        self,
        text: str,
        position: int,
        top_n: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Para uma posição específica, retorna as alternativas mais prováveis.
        
        Args:
            text: Texto da placa
            position: Posição a avaliar (0-6)
            top_n: Número de alternativas a retornar
            
        Returns:
            Lista de (caractere, probabilidade) ordenada
        """
        if position < 0 or position > 6:
            return []

        dist = self._positional_dist.get(position, {})

        # Se posição tem bi-gram e temos contexto
        if position in self._bigram_dist and position > 0 and len(text) > position - 1:
            prev_char = text[position - 1].upper()
            bigram = self._bigram_dist[position].get(prev_char, dist)
            dist = bigram

        sorted_chars = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        return sorted_chars[:top_n]

    def get_status(self) -> Dict:
        """Retorna status do modelo para UI."""
        return {
            'enabled': self.enabled,
            'valid_prefixes': len(self._valid_prefixes),
            'positional_distributions': len(self._positional_dist),
            'bigram_distributions': len(self._bigram_dist),
        }
