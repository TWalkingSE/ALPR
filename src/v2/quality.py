"""Objective image quality assessment for ALPR crops."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import cv2
import numpy as np


@dataclass
class QualityAssessment:
    brightness: float
    contrast: float
    sharpness: float
    snr: float
    motion_blur: float
    edge_density: float
    highlight_clipping: float
    shadow_clipping: float
    quality_score: float
    quality_band: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QualityAssessor:
    """Compute deterministic quality signals for plate crops."""

    def assess(self, image: np.ndarray) -> QualityAssessment:
        gray = self._to_gray(image)
        gray_float = gray.astype(np.float32)

        brightness = float(np.mean(gray_float)) if gray_float.size else 0.0
        contrast = float(np.std(gray_float)) if gray_float.size else 0.0
        laplacian = cv2.Laplacian(gray_float, cv2.CV_32F) if gray_float.size else gray_float
        sharpness = float(laplacian.var()) if laplacian.size else 0.0

        denoised = cv2.GaussianBlur(gray_float, (3, 3), 0) if gray_float.size else gray_float
        noise = gray_float - denoised
        noise_sigma = float(np.std(noise)) + 1e-6
        snr = float(20.0 * np.log10((brightness + 1.0) / noise_sigma)) if gray_float.size else 0.0

        sobel_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3) if gray_float.size else gray_float
        sobel_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3) if gray_float.size else gray_float
        mean_x = float(np.mean(np.abs(sobel_x))) if sobel_x.size else 0.0
        mean_y = float(np.mean(np.abs(sobel_y))) if sobel_y.size else 0.0
        anisotropy = abs(mean_x - mean_y) / (mean_x + mean_y + 1e-6)
        sharpness_penalty = 1.0 - self._clip(sharpness / 120.0)
        motion_blur = float(min(1.0, 0.65 * anisotropy + 0.35 * sharpness_penalty))

        edges = cv2.Canny(gray, 80, 180) if gray.size else gray
        edge_density = float(np.count_nonzero(edges) / edges.size) if edges.size else 0.0

        highlight_clipping = float(np.count_nonzero(gray >= 250) / gray.size) if gray.size else 0.0
        shadow_clipping = float(np.count_nonzero(gray <= 5) / gray.size) if gray.size else 0.0

        brightness_score = 1.0 - self._clip(abs(brightness - 145.0) / 145.0)
        contrast_score = self._clip(contrast / 55.0)
        sharpness_score = self._clip(np.log1p(sharpness) / 5.5) if sharpness > 0 else 0.0
        snr_score = self._clip((snr - 6.0) / 16.0)
        motion_score = 1.0 - self._clip(motion_blur / 0.55)
        edge_score = self._clip(edge_density / 0.18)
        clipping_score = 1.0 - self._clip((highlight_clipping + shadow_clipping) / 0.30)

        quality_score = float(
            0.18 * brightness_score
            + 0.16 * contrast_score
            + 0.20 * sharpness_score
            + 0.16 * snr_score
            + 0.12 * motion_score
            + 0.10 * edge_score
            + 0.08 * clipping_score
        )

        return QualityAssessment(
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
            snr=snr,
            motion_blur=motion_blur,
            edge_density=edge_density,
            highlight_clipping=highlight_clipping,
            shadow_clipping=shadow_clipping,
            quality_score=quality_score,
            quality_band=self._band_for_score(quality_score),
        )

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)
        if len(image.shape) == 2:
            gray = image
        elif image.shape[2] == 1:
            gray = image[:, :, 0]
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        return gray

    @staticmethod
    def _clip(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    @staticmethod
    def _band_for_score(score: float) -> str:
        if score >= 0.80:
            return 'excellent'
        if score >= 0.65:
            return 'good'
        if score >= 0.45:
            return 'review'
        return 'critical'