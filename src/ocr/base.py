# src/ocr/base.py
"""Classe abstrata base para engines OCR."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.ocr.types import OCRResult


class OCREngine(ABC):
    """Classe abstrata para engines OCR."""

    @abstractmethod
    def recognize(self, image: np.ndarray) -> List[OCRResult]:
        """
        Reconhece texto em uma imagem.
        
        Args:
            image: Imagem numpy array (BGR ou Grayscale)
            
        Returns:
            Lista de resultados [{text, confidence, char_confidences}]
        """
        pass
