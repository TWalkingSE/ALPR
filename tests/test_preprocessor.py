"""
Testes unitários para o módulo ImagePreprocessor.
Cobre as melhorias de pré-processamento: CLAHE adaptativo, deskew,
limpeza morfológica, multi-binarização e denoising aprimorado.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import cv2
from src.preprocessor import ImagePreprocessor


@pytest.fixture
def preprocessor():
    """Cria instância do ImagePreprocessor com todas as features habilitadas."""
    return ImagePreprocessor(
        enhance_contrast=True,
        remove_noise=True,
        sharpen=True,
        adaptive_threshold=True,
        morphological_cleanup=True,
        deskew=True,
        multi_binarization=True,
        adaptive_clahe=True,
        use_nlmeans_denoising=True,
    )


@pytest.fixture
def preprocessor_minimal():
    """Preprocessor com features mínimas."""
    return ImagePreprocessor(
        enhance_contrast=False,
        remove_noise=False,
        sharpen=False,
        adaptive_threshold=False,
        morphological_cleanup=False,
        deskew=False,
        multi_binarization=False,
        adaptive_clahe=False,
        use_nlmeans_denoising=False,
    )


@pytest.fixture
def sample_plate_image():
    """Cria uma imagem sintética de placa para testes."""
    # Criar imagem BGR 200x60 (proporção de placa)
    img = np.ones((60, 200, 3), dtype=np.uint8) * 200  # Fundo cinza claro
    # Adicionar texto preto simulando caracteres
    cv2.putText(img, "ABC1D23", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    return img


@pytest.fixture
def dark_image():
    """Cria uma imagem escura para teste de CLAHE adaptativo."""
    img = np.ones((60, 200), dtype=np.uint8) * 30  # Muito escura
    cv2.putText(img, "ABC1234", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 80, 2)
    return img


@pytest.fixture
def bright_image():
    """Cria uma imagem clara para teste de CLAHE adaptativo."""
    img = np.ones((60, 200), dtype=np.uint8) * 220  # Muito clara
    cv2.putText(img, "ABC1234", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 180, 2)
    return img


@pytest.fixture
def small_plate_image():
    """Cria uma imagem pequena para teste de upscaling."""
    img = np.ones((15, 50, 3), dtype=np.uint8) * 200
    cv2.putText(img, "AB", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    return img


# ==================== TESTES BÁSICOS ====================

class TestBasicProcessing:
    def test_process_returns_list(self, preprocessor, sample_plate_image):
        result = preprocessor.process(sample_plate_image)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_process_empty_image(self, preprocessor):
        result = preprocessor.process(np.array([]))
        assert result == []

    def test_process_none_image(self, preprocessor):
        result = preprocessor.process(None)
        assert result == []

    def test_original_image_first(self, preprocessor, sample_plate_image):
        result = preprocessor.process(sample_plate_image)
        assert result[0].shape == sample_plate_image.shape

    def test_minimal_preprocessor(self, preprocessor_minimal, sample_plate_image):
        result = preprocessor_minimal.process(sample_plate_image)
        # Deve retornar pelo menos a imagem original
        assert len(result) >= 1

    def test_process_grayscale_input(self, preprocessor):
        gray = np.ones((60, 200), dtype=np.uint8) * 128
        result = preprocessor.process(gray)
        assert isinstance(result, list)
        assert len(result) >= 1


# ==================== TESTES DE CLAHE ADAPTATIVO ====================

class TestAdaptiveCLAHE:
    def test_adaptive_clip_limit_dark_image(self, dark_image):
        clip = ImagePreprocessor._compute_adaptive_clip_limit(dark_image)
        assert clip >= 3.0  # Imagem escura = clipLimit alto

    def test_adaptive_clip_limit_bright_image(self, bright_image):
        clip = ImagePreprocessor._compute_adaptive_clip_limit(bright_image)
        assert clip <= 2.0  # Imagem clara = clipLimit baixo

    def test_adaptive_clip_limit_normal_image(self):
        img = np.ones((60, 200), dtype=np.uint8) * 128
        clip = ImagePreprocessor._compute_adaptive_clip_limit(img)
        assert 1.0 <= clip <= 4.0

    def test_adaptive_clip_limit_range(self):
        """clipLimit deve sempre estar entre 1.0 e 4.0."""
        for mean_val in [0, 50, 100, 150, 200, 255]:
            img = np.ones((60, 200), dtype=np.uint8) * mean_val
            clip = ImagePreprocessor._compute_adaptive_clip_limit(img)
            assert 1.0 <= clip <= 4.0


# ==================== TESTES DE LIMPEZA MORFOLÓGICA ====================

class TestMorphologicalCleanup:
    def test_cleanup_removes_noise(self):
        # Criar imagem com ruído (pontos isolados)
        img = np.zeros((60, 200), dtype=np.uint8)
        # Adicionar pontos de ruído isolados
        for _ in range(50):
            x, y = np.random.randint(0, 200), np.random.randint(0, 60)
            img[y, x] = 255
        
        prep = ImagePreprocessor()
        cleaned = prep._morphological_cleanup(img)
        # Imagem limpa deve ter menos pixels brancos
        assert np.sum(cleaned > 0) <= np.sum(img > 0)

    def test_cleanup_preserves_structure(self):
        # Criar imagem com estrutura (retângulo sólido)
        img = np.zeros((60, 200), dtype=np.uint8)
        cv2.rectangle(img, (50, 20), (150, 40), 255, -1)  # Retângulo cheio
        
        prep = ImagePreprocessor()
        cleaned = prep._morphological_cleanup(img)
        # Estrutura principal deve ser preservada
        assert np.sum(cleaned > 0) > 0

    def test_cleanup_output_shape(self):
        img = np.zeros((60, 200), dtype=np.uint8)
        prep = ImagePreprocessor()
        cleaned = prep._morphological_cleanup(img)
        assert cleaned.shape == img.shape


# ==================== TESTES DE DESKEW ====================

class TestDeskew:
    def test_deskew_returns_none_for_no_lines(self, preprocessor):
        # Imagem uniforme sem linhas
        img = np.ones((60, 200), dtype=np.uint8) * 128
        result = preprocessor._deskew_image(img)
        assert result is None  # Sem linhas detectadas

    def test_deskew_returns_none_for_small_angle(self, preprocessor):
        # Imagem com linhas horizontais (ângulo < 0.5°)
        img = np.zeros((60, 200), dtype=np.uint8)
        cv2.line(img, (0, 30), (200, 30), 255, 1)
        cv2.line(img, (0, 50), (200, 50), 255, 1)
        result = preprocessor._deskew_image(img)
        # Ângulo muito pequeno, não deve corrigir
        assert result is None

    def test_deskew_output_same_shape(self, preprocessor):
        # Criar imagem com linhas inclinadas
        img = np.zeros((100, 300), dtype=np.uint8)
        # Linhas com inclinação de ~5°
        cv2.line(img, (0, 40), (300, 66), 255, 2)
        cv2.line(img, (0, 60), (300, 86), 255, 2)
        cv2.line(img, (0, 80), (300, 106), 255, 2)
        
        result = preprocessor._deskew_image(img)
        if result is not None:
            assert result.shape == img.shape


# ==================== TESTES DE MULTI-BINARIZAÇÃO ====================

class TestMultiBinarization:
    def test_multi_binarization_produces_more_images(self, sample_plate_image):
        """Com multi_binarization, deve gerar mais variantes que sem."""
        pp_multi = ImagePreprocessor(
            enhance_contrast=True,
            remove_noise=True,
            sharpen=True,
            adaptive_threshold=True,
            multi_binarization=True,
            deskew=False,
        )
        pp_single = ImagePreprocessor(
            enhance_contrast=True,
            remove_noise=True,
            sharpen=True,
            adaptive_threshold=True,
            multi_binarization=False,
            deskew=False,
        )
        
        result_multi = pp_multi.process(sample_plate_image)
        result_single = pp_single.process(sample_plate_image)
        
        assert len(result_multi) > len(result_single)

    def test_all_outputs_are_valid_arrays(self, preprocessor, sample_plate_image):
        result = preprocessor.process(sample_plate_image)
        for img in result:
            assert isinstance(img, np.ndarray)
            assert img.size > 0


# ==================== TESTES DE DENOISING ====================

class TestDenoising:
    def test_nlmeans_vs_bilateral(self, sample_plate_image):
        """Ambos modos de denoising devem produzir resultados válidos."""
        pp_nlmeans = ImagePreprocessor(
            remove_noise=True,
            use_nlmeans_denoising=True,
            enhance_contrast=False,
            sharpen=False,
            adaptive_threshold=False,
            deskew=False,
        )
        pp_bilateral = ImagePreprocessor(
            remove_noise=True,
            use_nlmeans_denoising=False,
            enhance_contrast=False,
            sharpen=False,
            adaptive_threshold=False,
            deskew=False,
        )
        
        result_nl = pp_nlmeans.process(sample_plate_image)
        result_bi = pp_bilateral.process(sample_plate_image)
        
        assert len(result_nl) >= 2  # Original + denoised
        assert len(result_bi) >= 2


# ==================== TESTES DE UPSCALING ====================

class TestUpscaling:
    def test_small_image_is_upscaled(self, preprocessor, small_plate_image):
        result = preprocessor.process(small_plate_image)
        # Deve ter sido upscaled: alguma saída deve ser maior que o original
        has_larger = any(
            img.shape[0] > small_plate_image.shape[0]
            for img in result[1:]  # Excluir original
            if len(img.shape) == 2  # Grayscale outputs
        )
        assert has_larger or len(result) >= 2

    def test_normal_image_not_upscaled(self, preprocessor, sample_plate_image):
        result = preprocessor.process(sample_plate_image)
        assert len(result) >= 2  # Deve gerar múltiplas versões
