"""Heuristic tampering review for ALPR crops."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import cv2
import numpy as np


@dataclass
class ForensicAnalysis:
    tampering_score: float
    severity: str
    review_recommended: bool
    signals: List[str]
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ForensicAnalyzer:
    """Review-only tampering heuristics for suspicious crops."""

    def __init__(
        self,
        jpeg_quality: int = 90,
        review_threshold: float = 0.55,
        high_risk_threshold: float = 0.75,
    ):
        self.jpeg_quality = int(max(50, min(100, jpeg_quality)))
        self.review_threshold = float(review_threshold)
        self.high_risk_threshold = float(high_risk_threshold)

    def analyze(self, image: np.ndarray) -> ForensicAnalysis:
        gray, color = self._normalize_inputs(image)
        if gray.size == 0:
            return ForensicAnalysis(0.0, 'low', False, [], {})

        ela_score = self._ela_score(color)
        illumination_inconsistency = self._quadrant_spread(gray, reducer=np.mean, scale=45.0)
        edge_inconsistency = self._quadrant_spread(self._edge_density_map(gray), reducer=np.mean, scale=0.12)

        residual = gray.astype(np.float32) - cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
        noise_inconsistency = self._quadrant_spread(np.abs(residual), reducer=np.std, scale=12.0)
        border_inconsistency = self._border_center_delta(gray)

        tampering_score = float(
            0.30 * ela_score
            + 0.20 * illumination_inconsistency
            + 0.20 * noise_inconsistency
            + 0.15 * edge_inconsistency
            + 0.15 * border_inconsistency
        )

        signals: List[str] = []
        if ela_score >= 0.55:
            signals.append('recompression_residual_high')
        if illumination_inconsistency >= 0.55:
            signals.append('illumination_inconsistency')
        if noise_inconsistency >= 0.55:
            signals.append('noise_profile_inconsistency')
        if edge_inconsistency >= 0.55:
            signals.append('edge_density_inconsistency')
        if border_inconsistency >= 0.55:
            signals.append('border_center_inconsistency')

        severity = 'low'
        if tampering_score >= self.high_risk_threshold:
            severity = 'high'
        elif tampering_score >= self.review_threshold:
            severity = 'review'

        return ForensicAnalysis(
            tampering_score=tampering_score,
            severity=severity,
            review_recommended=tampering_score >= self.review_threshold,
            signals=signals,
            metrics={
                'ela_score': ela_score,
                'illumination_inconsistency': illumination_inconsistency,
                'noise_inconsistency': noise_inconsistency,
                'edge_inconsistency': edge_inconsistency,
                'border_inconsistency': border_inconsistency,
            },
        )

    def _ela_score(self, color: np.ndarray) -> float:
        success, encoded = cv2.imencode(
            '.jpg',
            color,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not success:
            return 0.0
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None or decoded.size == 0:
            return 0.0
        diff = cv2.absdiff(color, decoded)
        return float(min(1.0, np.mean(diff) / 32.0))

    @staticmethod
    def _normalize_inputs(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if image is None or image.size == 0:
            empty_gray = np.zeros((0, 0), dtype=np.uint8)
            empty_color = np.zeros((0, 0, 3), dtype=np.uint8)
            return empty_gray, empty_color
        if len(image.shape) == 2:
            gray = image.astype(np.uint8)
            color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return gray, color
        if image.shape[2] == 4:
            color = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            color = image.astype(np.uint8)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        return gray, color

    @staticmethod
    def _edge_density_map(gray: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(gray, 80, 180)
        return edges.astype(np.float32) / 255.0

    @staticmethod
    def _quadrant_spread(image: np.ndarray, reducer, scale: float) -> float:
        h, w = image.shape[:2]
        if h < 4 or w < 4:
            return 0.0
        quadrants = [
            image[: h // 2, : w // 2],
            image[: h // 2, w // 2 :],
            image[h // 2 :, : w // 2],
            image[h // 2 :, w // 2 :],
        ]
        values = [float(reducer(q)) for q in quadrants if q.size]
        if not values:
            return 0.0
        return float(min(1.0, np.std(values) / max(scale, 1e-6)))

    @staticmethod
    def _border_center_delta(gray: np.ndarray) -> float:
        h, w = gray.shape[:2]
        if h < 8 or w < 8:
            return 0.0
        border = max(2, min(h, w) // 8)
        border_mask = np.ones((h, w), dtype=bool)
        border_mask[border:h - border, border:w - border] = False
        center = gray[border:h - border, border:w - border]
        if center.size == 0:
            return 0.0
        border_values = gray[border_mask]
        delta = abs(float(np.mean(border_values)) - float(np.mean(center)))
        return float(min(1.0, delta / 70.0))