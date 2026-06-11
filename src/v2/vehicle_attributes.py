"""Reconhecimento opcional de atributos do veículo (cor, marca, modelo).

Este módulo é complementar à leitura de placa e foi desenhado para a mesma
filosofia do projeto: offline-first, determinístico e com degradação graciosa.

O que é totalmente funcional offline, sem pesos externos:
- Detecção de cor dominante do veículo via análise HSV.

O que é plugável (e degrada para "desconhecido" quando ausente):
- Classificação de marca/modelo. Como esse reconhecimento exige um modelo
  treinado (dataset de marca/modelo), o módulo aceita um classificador injetável
  (`MakeModelClassifier`). Sem um classificador configurado, marca/modelo
  retornam vazios em vez de inventar um resultado.

A região do veículo é estimada de forma heurística a partir da bounding box da
placa quando o frame completo está disponível (a placa fica tipicamente na
porção inferior-central do veículo).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# Nomes de cor em PT-BR usados na saída.
COLOR_UNKNOWN = 'desconhecido'


@dataclass
class VehicleAttributes:
    """Atributos estimados de um veículo associado a uma placa."""

    color: str = COLOR_UNKNOWN
    color_confidence: float = 0.0
    make: str = ''
    model: str = ''
    make_model_confidence: float = 0.0
    vehicle_bbox: tuple[int, int, int, int] | None = None
    source: str = 'disabled'
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.vehicle_bbox is not None:
            data['vehicle_bbox'] = list(self.vehicle_bbox)
        return data


class MakeModelClassifier(Protocol):
    """Interface para um classificador de marca/modelo injetável.

    Implementações devem receber um crop BGR do veículo e retornar
    (marca, modelo, confiança). Retornar ('', '', 0.0) quando indeterminado.
    """

    def predict(self, vehicle_crop: np.ndarray) -> tuple[str, str, float]:
        ...


def detect_dominant_color(
    image_bgr: np.ndarray,
    sample_size: int = 64,
) -> tuple[str, float]:
    """Estima a cor dominante de um crop BGR.

    Returns:
        (nome_da_cor_pt_br, confiança) onde confiança é a fração de pixels que
        votaram na cor vencedora. Retorna (COLOR_UNKNOWN, 0.0) para entradas
        inválidas.
    """
    if image_bgr is None or getattr(image_bgr, 'size', 0) == 0:
        return COLOR_UNKNOWN, 0.0

    image = np.asarray(image_bgr)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        return COLOR_UNKNOWN, 0.0

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Reduzir para acelerar e suavizar ruído de pixel.
    side = max(8, int(sample_size))
    small = cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)

    votes: dict[str, int] = {}
    for h, s, v in pixels:
        color = _classify_hsv_pixel(int(h), int(s), int(v))
        votes[color] = votes.get(color, 0) + 1

    if not votes:
        return COLOR_UNKNOWN, 0.0

    total = float(sum(votes.values()))
    winner = max(votes.items(), key=lambda item: item[1])
    return winner[0], float(winner[1] / total) if total > 0 else 0.0


def _classify_hsv_pixel(h: int, s: int, v: int) -> str:
    """Classifica um único pixel HSV (OpenCV: H 0-179, S/V 0-255) em cor PT-BR."""
    # Acromático: saturação baixa → escala de cinza definida pelo brilho.
    if s < 40:
        if v < 50:
            return 'preto'
        if v < 110:
            return 'cinza'
        if v < 190:
            return 'prata'
        return 'branco'

    # Muito escuro mesmo com alguma saturação → preto.
    if v < 40:
        return 'preto'

    # Cromático: classificar pelo matiz.
    if h <= 10 or h >= 170:
        return 'vermelho'
    if 11 <= h <= 22:
        # Laranja escuro/dessaturado tende a ser percebido como marrom.
        if v < 130 or s < 110:
            return 'marrom'
        return 'laranja'
    if 23 <= h <= 33:
        return 'amarelo'
    if 34 <= h <= 85:
        return 'verde'
    if 86 <= h <= 125:
        return 'azul'
    if 126 <= h <= 169:
        return 'roxo'
    return COLOR_UNKNOWN


class VehicleAttributeAnalyzer:
    """Analisa atributos do veículo a partir do frame e da bbox da placa."""

    def __init__(
        self,
        enabled: bool = False,
        make_model_classifier: MakeModelClassifier | None = None,
        roi_width_scale: float = 3.0,
        roi_height_scale: float = 5.0,
    ):
        """
        Args:
            enabled: Liga/desliga a análise.
            make_model_classifier: Classificador opcional de marca/modelo.
            roi_width_scale: Largura da ROI do veículo em múltiplos da largura da placa.
            roi_height_scale: Altura da ROI do veículo em múltiplos da altura da placa.
        """
        self.enabled = enabled
        self.make_model_classifier = make_model_classifier
        self.roi_width_scale = max(1.0, float(roi_width_scale))
        self.roi_height_scale = max(1.0, float(roi_height_scale))

    def analyze(
        self,
        full_image: np.ndarray | None,
        plate_bbox: tuple[int, int, int, int] | None,
    ) -> VehicleAttributes:
        """Estima atributos do veículo ao redor da placa."""
        if not self.enabled:
            return VehicleAttributes(source='disabled')

        if full_image is None or getattr(full_image, 'size', 0) == 0 or not plate_bbox:
            return VehicleAttributes(source='unavailable')

        roi, roi_bbox = self._estimate_vehicle_roi(full_image, plate_bbox)
        if roi is None or roi.size == 0:
            return VehicleAttributes(source='unavailable')

        return self.analyze_crop(roi, vehicle_bbox=roi_bbox)

    def analyze_crop(
        self,
        vehicle_crop: np.ndarray,
        vehicle_bbox: tuple[int, int, int, int] | None = None,
    ) -> VehicleAttributes:
        """Analisa um crop de veículo já recortado."""
        if not self.enabled:
            return VehicleAttributes(source='disabled')
        if vehicle_crop is None or getattr(vehicle_crop, 'size', 0) == 0:
            return VehicleAttributes(source='unavailable')

        color, color_conf = detect_dominant_color(vehicle_crop)

        make, model, mm_conf = '', '', 0.0
        if self.make_model_classifier is not None:
            try:
                make, model, mm_conf = self.make_model_classifier.predict(vehicle_crop)
                make = str(make or '')
                model = str(model or '')
                mm_conf = float(mm_conf or 0.0)
            except Exception as exc:
                logger.debug("Classificador de marca/modelo falhou: %s", exc)
                make, model, mm_conf = '', '', 0.0

        return VehicleAttributes(
            color=color,
            color_confidence=color_conf,
            make=make,
            model=model,
            make_model_confidence=mm_conf,
            vehicle_bbox=vehicle_bbox,
            source='analyzed' if self.make_model_classifier is not None else 'color_only',
        )

    def _estimate_vehicle_roi(
        self,
        full_image: np.ndarray,
        plate_bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
        """Estima a ROI do veículo expandindo ao redor da placa.

        A placa costuma estar na metade inferior do veículo, então a ROI é
        expandida principalmente para cima e lateralmente.
        """
        height, width = full_image.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in plate_bbox)
        plate_w = max(1, x2 - x1)
        plate_h = max(1, y2 - y1)
        cx = (x1 + x2) // 2

        half_w = int(plate_w * self.roi_width_scale / 2)
        roi_x1 = max(0, cx - half_w)
        roi_x2 = min(width, cx + half_w)

        # Expandir mais para cima (carroceria) que para baixo.
        roi_y1 = max(0, int(y1 - plate_h * (self.roi_height_scale - 1)))
        roi_y2 = min(height, int(y2 + plate_h * 0.5))

        if roi_x1 >= roi_x2 or roi_y1 >= roi_y2:
            return None, None

        roi = full_image[roi_y1:roi_y2, roi_x1:roi_x2]
        return roi, (roi_x1, roi_y1, roi_x2, roi_y2)
