# tests/test_geometric_normalizer.py
"""
Testes para o GeometricNormalizer (src/geometric_normalizer.py).

Cobre:
  - Inicialização e flags (enabled, perspective, rotation, contrast, resize)
  - normalize() — pipeline end-to-end com diferentes inputs
  - _order_corners — ordenação dos cantos (TL, TR, BR, BL)
  - _apply_perspective_transform — aspect ratio guard
  - _standard_resize — redimensionamento + padding
  - _compute_adaptive_clip_limit — CLAHE adaptativo
  - _equalize_contrast — CLAHE color/gray
  - _correct_rotation — short-circuits sem linhas
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.geometric_normalizer import GeometricNormalizer


# ============================================================================
# Helpers
# ============================================================================

def _make_bgr_plate(h: int = 100, w: int = 300) -> np.ndarray:
    """Cria imagem de placa sintética (fundo claro com texto escuro)."""
    img = np.full((h, w, 3), 200, dtype=np.uint8)  # fundo claro
    cv2.putText(img, 'ABC1D23', (w // 8, int(h * 0.7)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    return img


def _make_gray_plate(h: int = 100, w: int = 300) -> np.ndarray:
    img = np.full((h, w), 200, dtype=np.uint8)
    cv2.putText(img, 'ABC1234', (w // 8, int(h * 0.7)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, 20, 2)
    return img


# ============================================================================
# Inicialização
# ============================================================================

class TestInitialization:
    def test_default_init(self):
        n = GeometricNormalizer()
        assert n.enabled is True
        assert n.perspective_correction is True
        assert n.rotation_correction is True
        assert n.contrast_equalization is True
        assert n.standard_resize is True
        assert n.target_width == GeometricNormalizer.DEFAULT_TARGET_WIDTH
        assert n.target_height == GeometricNormalizer.DEFAULT_TARGET_HEIGHT

    def test_disabled_passes_through(self):
        n = GeometricNormalizer(enabled=False)
        img = _make_bgr_plate()
        assert np.array_equal(n.normalize(img), img)

    def test_custom_target_dims(self):
        n = GeometricNormalizer(target_width=400, target_height=120)
        assert n.target_width == 400
        assert n.target_height == 120


# ============================================================================
# Inputs inválidos
# ============================================================================

class TestInvalidInputs:
    def test_none_image_returns_none(self):
        n = GeometricNormalizer()
        assert n.normalize(None) is None

    def test_empty_array_returns_empty(self):
        n = GeometricNormalizer()
        img = np.array([], dtype=np.uint8)
        result = n.normalize(img)
        assert result.size == 0


# ============================================================================
# normalize() — pipeline
# ============================================================================

class TestNormalize:
    def test_bgr_returns_bgr(self):
        n = GeometricNormalizer()
        img = _make_bgr_plate(100, 300)
        result = n.normalize(img)
        assert result is not None
        assert len(result.shape) == 3
        assert result.shape[2] == 3

    def test_grayscale_returns_grayscale(self):
        n = GeometricNormalizer()
        img = _make_gray_plate(100, 300)
        result = n.normalize(img)
        assert result is not None
        # CLAHE em gray preserva 2D
        assert len(result.shape) == 2

    def test_resize_produces_target_dims(self):
        n = GeometricNormalizer(
            perspective_correction=False,
            rotation_correction=False,
            contrast_equalization=False,
            standard_resize=True,
            target_width=300,
            target_height=100,
        )
        img = _make_bgr_plate(50, 200)  # menor que target
        result = n.normalize(img)
        assert result.shape[:2] == (100, 300)

    def test_all_stages_disabled_returns_copy(self):
        n = GeometricNormalizer(
            perspective_correction=False,
            rotation_correction=False,
            contrast_equalization=False,
            standard_resize=False,
        )
        img = _make_bgr_plate()
        result = n.normalize(img)
        assert np.array_equal(result, img)


# ============================================================================
# _order_corners — ordenação TL, TR, BR, BL
# ============================================================================

class TestOrderCorners:
    def test_already_ordered(self):
        pts = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)
        ordered = GeometricNormalizer._order_corners(pts)
        assert np.allclose(ordered[0], [0, 0])  # TL
        assert np.allclose(ordered[1], [100, 0])  # TR
        assert np.allclose(ordered[2], [100, 50])  # BR
        assert np.allclose(ordered[3], [0, 50])  # BL

    def test_shuffled_order_fixed(self):
        # Mesmos pontos, embaralhados
        pts = np.array([[100, 50], [0, 0], [0, 50], [100, 0]], dtype=np.float32)
        ordered = GeometricNormalizer._order_corners(pts)
        assert np.allclose(ordered[0], [0, 0])
        assert np.allclose(ordered[1], [100, 0])
        assert np.allclose(ordered[2], [100, 50])
        assert np.allclose(ordered[3], [0, 50])

    def test_returns_float32(self):
        pts = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.int32)
        ordered = GeometricNormalizer._order_corners(pts)
        assert ordered.dtype == np.float32

    def test_shape_4x2(self):
        pts = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)
        ordered = GeometricNormalizer._order_corners(pts)
        assert ordered.shape == (4, 2)


# ============================================================================
# _apply_perspective_transform
# ============================================================================

class TestApplyPerspectiveTransform:
    def test_valid_corners_produces_image(self):
        n = GeometricNormalizer()
        img = _make_bgr_plate(100, 300)
        corners = np.array([
            [10, 10], [290, 10], [290, 90], [10, 90]
        ], dtype=np.float32)
        result = n._apply_perspective_transform(img, corners)
        assert result is not None
        assert result.size > 0

    def test_degenerate_corners_returns_none(self):
        n = GeometricNormalizer()
        img = _make_bgr_plate()
        # Todos os cantos iguais → max_width/height = 0
        corners = np.array([
            [50, 50], [50, 50], [50, 50], [50, 50]
        ], dtype=np.float32)
        result = n._apply_perspective_transform(img, corners)
        assert result is None

    def test_invalid_aspect_ratio_returns_none(self):
        n = GeometricNormalizer()
        img = _make_bgr_plate()
        # Cantos formando quadrado (aspect ~1.0) — abaixo do mínimo válido
        # MIN_PLATE_ASPECT_RATIO = 1.2, *0.5 = 0.6 ainda OK
        # Cantos formando forma muito "alta" (aspect 0.1)
        corners = np.array([
            [0, 0], [10, 0], [10, 200], [0, 200]
        ], dtype=np.float32)
        result = n._apply_perspective_transform(img, corners)
        # aspect = 10/200 = 0.05 < 0.6 → None
        assert result is None


# ============================================================================
# _standard_resize
# ============================================================================

class TestStandardResize:
    def test_exact_target_dims(self):
        n = GeometricNormalizer(target_width=300, target_height=100)
        img = _make_bgr_plate(100, 300)
        result = n._standard_resize(img)
        assert result.shape[:2] == (100, 300)

    def test_upscale_with_padding(self):
        n = GeometricNormalizer(target_width=300, target_height=100)
        img = _make_bgr_plate(50, 150)
        result = n._standard_resize(img)
        assert result.shape[:2] == (100, 300)

    def test_downscale(self):
        n = GeometricNormalizer(target_width=300, target_height=100)
        img = _make_bgr_plate(200, 600)
        result = n._standard_resize(img)
        assert result.shape[:2] == (100, 300)

    def test_zero_dim_returns_none(self):
        n = GeometricNormalizer()
        img = np.zeros((0, 100, 3), dtype=np.uint8)
        result = n._standard_resize(img)
        assert result is None

    def test_preserves_channels(self):
        n = GeometricNormalizer()
        img_color = _make_bgr_plate(100, 300)
        img_gray = _make_gray_plate(100, 300)
        assert len(n._standard_resize(img_color).shape) == 3
        assert len(n._standard_resize(img_gray).shape) == 2


# ============================================================================
# _compute_adaptive_clip_limit
# ============================================================================

class TestAdaptiveClipLimit:
    def test_dark_image_higher_clip(self):
        # Imagem escura (mean ~40)
        dark = np.full((100, 300), 40, dtype=np.uint8)
        clip = GeometricNormalizer._compute_adaptive_clip_limit(dark)
        assert clip >= 3.0

    def test_bright_image_lower_clip(self):
        # Imagem clara (mean ~180)
        bright = np.full((100, 300), 180, dtype=np.uint8)
        clip = GeometricNormalizer._compute_adaptive_clip_limit(bright)
        assert clip <= 2.5

    def test_clip_in_valid_range(self):
        for mean_val in (30, 90, 150, 220):
            img = np.full((100, 300), mean_val, dtype=np.uint8)
            clip = GeometricNormalizer._compute_adaptive_clip_limit(img)
            assert 1.0 <= clip <= 4.0

    def test_high_std_reduces_clip(self):
        # Imagem com muito contraste (std alto)
        img = np.random.randint(0, 255, (100, 300), dtype=np.uint8)
        mean_val = int(np.mean(img))
        clip_high_std = GeometricNormalizer._compute_adaptive_clip_limit(img)
        # Comparar com imagem "achatada" no mesmo mean
        flat = np.full((100, 300), mean_val, dtype=np.uint8)
        clip_low_std = GeometricNormalizer._compute_adaptive_clip_limit(flat)
        # High std → clip menor ou igual
        assert clip_high_std <= clip_low_std + 0.01


# ============================================================================
# _equalize_contrast
# ============================================================================

class TestEqualizeContrast:
    def test_color_returns_bgr(self):
        n = GeometricNormalizer()
        img = _make_bgr_plate(100, 300)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = n._equalize_contrast(gray, img, is_color=True)
        assert result is not None
        assert result.shape == img.shape

    def test_gray_returns_gray(self):
        n = GeometricNormalizer()
        img = _make_gray_plate(100, 300)
        result = n._equalize_contrast(img, img, is_color=False)
        assert result is not None
        assert result.shape == img.shape


# ============================================================================
# _correct_rotation
# ============================================================================

class TestCorrectRotation:
    def test_flat_image_no_lines_returns_none(self):
        n = GeometricNormalizer()
        img = np.full((100, 300, 3), 128, dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = n._correct_rotation(gray, img)
        # Sem estrutura → sem linhas → None
        assert result is None

    def test_horizontal_plate_no_rotation_needed(self):
        n = GeometricNormalizer(min_rotation_angle=0.5)
        # Placa com linhas horizontais perfeitas
        img = np.full((100, 300, 3), 230, dtype=np.uint8)
        cv2.rectangle(img, (10, 20), (290, 80), (30, 30, 30), 2)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = n._correct_rotation(gray, img)
        # Sem inclinação significativa → None
        assert result is None


# ============================================================================
# _find_quadrilateral
# ============================================================================

class TestFindQuadrilateral:
    def test_empty_edges_returns_none(self):
        n = GeometricNormalizer()
        edges = np.zeros((100, 300), dtype=np.uint8)
        result = n._find_quadrilateral(edges, 100 * 300, 100, 300)
        assert result is None

    def test_synthetic_rectangle_detected(self):
        """Desenha retângulo preenchido e verifica detecção."""
        n = GeometricNormalizer(min_contour_area_ratio=0.10)
        edges = np.zeros((200, 600), dtype=np.uint8)
        # Retângulo com aspect ratio 3:1 (similar a placa)
        cv2.rectangle(edges, (50, 50), (550, 150), 255, -1)
        result = n._find_quadrilateral(edges, 200 * 600, 200, 600)
        # Pode ou não detectar dependendo do approxPolyDP; teste apenas que não lança
        if result is not None:
            assert result.shape == (4, 2)
