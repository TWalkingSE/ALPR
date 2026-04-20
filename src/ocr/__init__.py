"""OCR package for ALPR v2.

The v2 runtime uses PaddleOCR through OCRManager.
"""

from src.ocr.base import OCREngine
from src.ocr.manager import OCRManager
from src.ocr.paddle_engine import PaddleOCREngine

__all__ = [
    'OCREngine',
    'OCRManager',
    'PaddleOCREngine',
]
