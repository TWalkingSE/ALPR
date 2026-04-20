"""Core package for the ALPR v2 codebase."""

# Importações leves (sem dependências pesadas como torch/ultralytics)
from .constants import (
    CONFIDENCE_LEVELS,
    OCR_CHAR_MAP,
    PLATE_PATTERN_MERCOSUL,
    PLATE_PATTERN_OLD,
    RE_MERCOSUL,
    RE_OLD,
    SIMILAR_CHARS,
)
from .validator import PlateValidator


# Importações pesadas são lazy (permite testes unitários sem torch/ultralytics)
def __getattr__(name):
    if name == 'PlateDetector':
        from .detector import PlateDetector
        return PlateDetector
    elif name == 'ImagePreprocessor':
        from .preprocessor import ImagePreprocessor
        return ImagePreprocessor
    elif name in ('PremiumALPRProvider', 'PremiumALPRResult'):
        from .premium_alpr import PremiumALPRProvider, PremiumALPRResult
        return PremiumALPRProvider if name == 'PremiumALPRProvider' else PremiumALPRResult
    elif name == 'VideoProcessor':
        from .video_processor import VideoProcessor
        return VideoProcessor
    raise AttributeError(f"module 'src' has no attribute {name!r}")
