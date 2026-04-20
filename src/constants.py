# src/constants.py
"""
Constantes centralizadas do projeto ALPR.
Mapas de caracteres similares, formatos de placa e definições compartilhadas.
"""

import re


# ==================== MODELOS PADRÃO ====================

DEFAULT_GLM_OCR_MODEL = 'glm-ocr:bf16'
DEFAULT_OLMOCR_MODEL = 'richardyoung/olmocr2:7b-q8'

# ==================== CARACTERES SIMILARES (OCR) ====================

# Mapeamento bidirecional de caracteres visualmente similares.
# Usado pelos componentes determinísticos da v2 para validação e ranqueamento.
SIMILAR_CHARS = {
    '0': ['O', 'D', 'Q', 'C'],
    'O': ['0', 'Q', 'D', 'C'],
    '1': ['I', 'L', '7', 'J', '3'],
    'I': ['1', 'L', '7', 'J'],
    'L': ['1', 'I'],
    '2': ['Z'],
    'Z': ['2'],
    '4': ['A'],
    'A': ['4'],
    '5': ['S'],
    'S': ['5'],
    '8': ['B', '3'],
    'B': ['8', '3'],
    '6': ['G'],
    'G': ['6', 'C'],
    '9': ['P'],
    'P': ['9', 'F'],
    'Q': ['O', '0'],
    'D': ['O', '0'],
    '3': ['B', '8', 'J', '1'],
    '7': ['1', 'I', 'T'],
    'T': ['7'],
    # Pares adicionais de confusão OCR (câmeras de vigilância)
    'C': ['G', '0', 'O'],
    'E': ['F'],
    'F': ['E', 'P'],
    'R': ['K'],
    'K': ['R', 'X'],
    'M': ['N', 'W'],
    'N': ['M', 'H'],
    'H': ['N'],
    'V': ['U', 'Y'],
    'U': ['V'],
    'J': ['1', 'I', '3'],
    'W': ['M'],
    'Y': ['V'],
    'X': ['K'],
}

# Mapeamento simples char->char para substituição direta OCR.
# Usado pelo validador e por utilitários de normalização da v2.
OCR_CHAR_MAP = {
    '0': 'O', 'O': '0',
    '1': 'I', 'I': '1',
    'L': '1',
    'B': '8', '8': 'B',
    'S': '5', '5': 'S',
    'Z': '2', '2': 'Z',
    'G': '6', '6': 'G',
    'Q': 'O',
    'D': '0',
    'T': '7', '7': 'T',
    # Pares adicionais de confusão
    'C': 'G',
    'J': '3',
    'E': 'F', 'F': 'E',
    'H': 'N', 'N': 'H',
    'V': 'U', 'U': 'V',
    'K': 'R', 'R': 'K',
    'M': 'N',
    'W': 'M',
}


# ==================== PESOS DE CONFUSÃO OCR ====================

# Probabilidade empírica de cada confusão OCR (0.0 a 1.0).
# Quanto maior, mais provável que o OCR confunda esses dois caracteres.
# Usado pelo validator e pelo ranqueamento de candidatos na v2.
CONFUSION_WEIGHTS = {
    # Confusões extremamente comuns (>0.85)
    ('0', 'O'): 0.95, ('O', '0'): 0.95,
    ('0', 'D'): 0.70, ('D', '0'): 0.70,
    ('1', 'I'): 0.90, ('I', '1'): 0.90,
    ('1', 'L'): 0.75, ('L', '1'): 0.75,
    ('8', 'B'): 0.90, ('B', '8'): 0.90,
    ('5', 'S'): 0.85, ('S', '5'): 0.85,
    ('2', 'Z'): 0.80, ('Z', '2'): 0.80,
    ('6', 'G'): 0.75, ('G', '6'): 0.75,
    ('7', 'T'): 0.70, ('T', '7'): 0.70,
    ('7', '1'): 0.60, ('1', '7'): 0.60,

    # Confusões comuns (0.50-0.70)
    ('0', 'Q'): 0.55, ('Q', '0'): 0.55,
    ('0', 'C'): 0.30, ('C', '0'): 0.30,
    ('Q', 'O'): 0.50, ('O', 'Q'): 0.50,
    ('3', 'B'): 0.50, ('B', '3'): 0.50,
    ('3', '8'): 0.45, ('8', '3'): 0.45,
    ('4', 'A'): 0.55, ('A', '4'): 0.55,
    ('9', 'P'): 0.50, ('P', '9'): 0.50,
    ('1', 'J'): 0.40, ('J', '1'): 0.40,
    ('3', 'J'): 0.45, ('J', '3'): 0.45,
    ('1', '3'): 0.30, ('3', '1'): 0.30,

    # Confusões menos comuns (<0.40)
    ('C', 'G'): 0.35, ('G', 'C'): 0.35,
    ('E', 'F'): 0.30, ('F', 'E'): 0.30,
    ('P', 'F'): 0.25, ('F', 'P'): 0.25,
    ('R', 'K'): 0.20, ('K', 'R'): 0.20,
    ('K', 'X'): 0.20, ('X', 'K'): 0.20,
    ('M', 'N'): 0.35, ('N', 'M'): 0.35,
    ('M', 'W'): 0.25, ('W', 'M'): 0.25,
    ('N', 'H'): 0.30, ('H', 'N'): 0.30,
    ('V', 'U'): 0.35, ('U', 'V'): 0.35,
    ('V', 'Y'): 0.20, ('Y', 'V'): 0.20,
    ('I', 'L'): 0.50, ('L', 'I'): 0.50,
    ('I', 'J'): 0.35, ('J', 'I'): 0.35,
    ('D', 'O'): 0.60, ('O', 'D'): 0.60,
}


def get_confusion_weight(char_from: str, char_to: str) -> float:
    """
    Retorna o peso de confusão entre dois caracteres.
    
    Args:
        char_from: Caractere original (lido pelo OCR)
        char_to: Caractere substituto proposto
        
    Returns:
        Peso entre 0.0 e 1.0 (0.1 default para pares não catalogados)
    """
    return CONFUSION_WEIGHTS.get((char_from, char_to), 0.10)


# Bias para Mercosul: desde 2018, a grande maioria das placas novas é Mercosul.
# Em caso de ambiguidade entre old/mercosul, este multiplicador favorece Mercosul.
MERCOSUL_FORMAT_BIAS = 1.15


# ==================== LIMITES DE RESULTADO ====================

# Máximo de placas únicas a exibir nos resultados de vídeo
# Apenas as combinações mais prováveis (melhor score composto) são mantidas
MAX_UNIQUE_PLATES = 10

# Confiança mínima por caractere para considerá-lo "confirmado"
# Caracteres abaixo deste limiar aparecem como * na leitura confirmada
CHAR_CONFIDENCE_THRESHOLD = 0.70


# ==================== FORMATOS DE PLACAS BRASILEIRAS ====================

# Regex patterns
PLATE_PATTERN_OLD = r'^[A-Z]{3}[0-9]{4}$'
PLATE_PATTERN_OLD_HYPHEN = r'^[A-Z]{3}[-\s]?[0-9]{4}$'
PLATE_PATTERN_MERCOSUL = r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$'

# Compiled regex (reutilizáveis)
RE_OLD = re.compile(PLATE_PATTERN_OLD)
RE_OLD_HYPHEN = re.compile(PLATE_PATTERN_OLD_HYPHEN)
RE_MERCOSUL = re.compile(PLATE_PATTERN_MERCOSUL)

# Definição posicional: L=letra, N=número
PLATE_POSITIONS_OLD = ['L', 'L', 'L', 'N', 'N', 'N', 'N']  # AAA-1234
PLATE_POSITIONS_MERCOSUL = ['L', 'L', 'L', 'N', 'L', 'N', 'N']  # AAA1A23


# ==================== TERMOS FORENSES ====================

FORENSIC_TERMS = {
    'positive': 'evidências indicam',
    'probable': 'compatível com',
    'possible': 'indicativo de',
    'uncertain': 'não é possível determinar com certeza',
    'negative': 'não foram encontradas evidências de',
    'avoid': [
        'provavelmente', 'talvez', 'acho que', 'parece que',
        'com certeza', 'definitivamente', 'sem dúvida'
    ],
}

CONFIDENCE_LEVELS = {
    'alta': {'min': 0.80, 'label': 'Alta', 'description': 'Identificação positiva e segura'},
    'media': {'min': 0.60, 'label': 'Média', 'description': 'Identificação provável'},
    'baixa': {'min': 0.40, 'label': 'Baixa', 'description': 'Indicativo, porém inconclusivo'},
    'insuficiente': {'min': 0.0, 'label': 'Insuficiente', 'description': 'Dados insuficientes para conclusão'},
}
