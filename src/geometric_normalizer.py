# src/geometric_normalizer.py
"""
Módulo de Normalização Geométrica para placas veiculares.

Camada intermediária entre a detecção YOLO (crop da placa) e o OCR.
Aplica uma sequência de transformações geométricas para retificar a imagem
da placa, melhorando significativamente a acurácia do OCR subsequente.

Pipeline de normalização:
1. Detecção dos 4 cantos da placa (contornos OpenCV)
2. Transformação de perspectiva (homografia)
3. Correção de rotação (Hough Lines)
4. Equalização de contraste (CLAHE adaptativo)
5. Redimensionamento padronizado

Todas as operações usam OpenCV exclusivamente.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class GeometricNormalizer:
    """
    Normalização geométrica de crops de placas veiculares.

    Recebe o crop da placa (saída do detector YOLO) e aplica
    transformações para retificá-la antes do OCR.
    """

    # Aspect ratios válidos para placas veiculares brasileiras
    # Mercosul: 400×130mm → ratio ≈ 3.08
    # Antiga:   400×130mm → ratio ≈ 3.08
    # Moto:     190×130mm → ratio ≈ 1.46
    MIN_PLATE_ASPECT_RATIO = 1.2
    MAX_PLATE_ASPECT_RATIO = 5.0

    # Tamanho padrão de saída
    DEFAULT_TARGET_WIDTH = 300
    DEFAULT_TARGET_HEIGHT = 100

    def __init__(
        self,
        enabled: bool = True,
        perspective_correction: bool = True,
        rotation_correction: bool = True,
        contrast_equalization: bool = True,
        standard_resize: bool = True,
        target_width: int = DEFAULT_TARGET_WIDTH,
        target_height: int = DEFAULT_TARGET_HEIGHT,
        min_contour_area_ratio: float = 0.15,
        max_rotation_angle: float = 30.0,
        min_rotation_angle: float = 0.5,
        clahe_clip_limit: Optional[float] = None,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
    ):
        """
        Inicializa o normalizador geométrico.

        Args:
            enabled: Se a normalização está ativa (se False, retorna imagem original)
            perspective_correction: Se deve corrigir perspectiva via homografia
            rotation_correction: Se deve corrigir rotação via Hough Lines
            contrast_equalization: Se deve aplicar CLAHE adaptativo
            standard_resize: Se deve redimensionar para tamanho padrão
            target_width: Largura alvo após normalização
            target_height: Altura alvo após normalização
            min_contour_area_ratio: Área mínima do contorno como fração da imagem
            max_rotation_angle: Ângulo máximo para considerar rotação (graus)
            min_rotation_angle: Ângulo mínimo para aplicar correção (graus)
            clahe_clip_limit: clipLimit para CLAHE (None = adaptativo)
            clahe_tile_grid: tileGridSize para CLAHE
        """
        self.enabled = enabled
        self.perspective_correction = perspective_correction
        self.rotation_correction = rotation_correction
        self.contrast_equalization = contrast_equalization
        self.standard_resize = standard_resize
        self.target_width = target_width
        self.target_height = target_height
        self.min_contour_area_ratio = min_contour_area_ratio
        self.max_rotation_angle = max_rotation_angle
        self.min_rotation_angle = min_rotation_angle
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid

        logger.info(
            f"GeometricNormalizer inicializado: enabled={enabled}, "
            f"perspective={perspective_correction}, rotation={rotation_correction}, "
            f"contrast={contrast_equalization}, resize={standard_resize}, "
            f"target={target_width}×{target_height}"
        )

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica normalização geométrica completa ao crop da placa.

        Pipeline:
        1. Detecção dos 4 cantos → Transformação de perspectiva
        2. Correção de rotação
        3. Equalização de contraste
        4. Redimensionamento padronizado

        Se alguma etapa falhar, as etapas subsequentes continuam
        operando sobre o resultado parcial (graceful degradation).

        Args:
            image: Crop da placa (numpy array BGR ou Grayscale)

        Returns:
            Imagem normalizada (numpy array, mesmo formato da entrada)
        """
        if not self.enabled:
            return image

        if image is None or image.size == 0:
            logger.warning("GeometricNormalizer: Imagem vazia recebida")
            return image

        current = image.copy()
        is_color = len(current.shape) == 3 and current.shape[2] == 3

        # Obter versão grayscale para operações que precisam
        if is_color:
            gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        else:
            gray = current.copy()

        # ── Etapa 1: Detecção de cantos + Transformação de perspectiva ──
        if self.perspective_correction:
            try:
                corners = self._detect_plate_corners(gray, current)
                if corners is not None:
                    warped = self._apply_perspective_transform(current, corners)
                    if warped is not None and warped.size > 0:
                        current = warped
                        # Atualizar gray após transformação
                        if is_color:
                            gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
                        else:
                            gray = current.copy()
                        logger.debug("GeometricNormalizer: Perspectiva corrigida")
                    else:
                        logger.debug("GeometricNormalizer: Perspectiva — warp resultou em imagem vazia")
                else:
                    logger.debug("GeometricNormalizer: 4 cantos não detectados, pulando perspectiva")
            except Exception as e:
                logger.debug(f"GeometricNormalizer: Erro na correção de perspectiva: {e}")

        # ── Etapa 2: Correção de rotação ──
        if self.rotation_correction:
            try:
                rotated = self._correct_rotation(gray, current)
                if rotated is not None:
                    current = rotated
                    if is_color:
                        gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = current.copy()
                    logger.debug("GeometricNormalizer: Rotação corrigida")
            except Exception as e:
                logger.debug(f"GeometricNormalizer: Erro na correção de rotação: {e}")

        # ── Etapa 3: Equalização de contraste ──
        if self.contrast_equalization:
            try:
                equalized = self._equalize_contrast(gray, current, is_color)
                if equalized is not None:
                    current = equalized
                    logger.debug("GeometricNormalizer: Contraste equalizado")
            except Exception as e:
                logger.debug(f"GeometricNormalizer: Erro na equalização de contraste: {e}")

        # ── Etapa 4: Redimensionamento padronizado ──
        if self.standard_resize:
            try:
                resized = self._standard_resize(current)
                if resized is not None:
                    current = resized
                    logger.debug(
                        f"GeometricNormalizer: Redimensionado para {current.shape[1]}×{current.shape[0]}"
                    )
            except Exception as e:
                logger.debug(f"GeometricNormalizer: Erro no redimensionamento: {e}")

        return current

    # ──────────────────────────────────────────────────────────────────
    #  Etapa 1: Detecção de Cantos + Perspectiva
    # ──────────────────────────────────────────────────────────────────

    def _detect_plate_corners(
        self, gray: np.ndarray, original: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Detecta os 4 cantos da placa usando contornos OpenCV.

        Aplica Canny edge detection → findContours → approxPolyDP
        para encontrar o contorno quadrilateral com aspect ratio de placa.
        
        Para imagens pequenas (<40px), faz upscale 2× antes da detecção
        para melhorar a resolução de bordas e depois escala os cantos de volta.

        Args:
            gray: Imagem grayscale
            original: Imagem original (BGR ou gray)

        Returns:
            Array 4×2 com coordenadas dos cantos (tl, tr, br, bl)
            ou None se não encontrar
        """
        h, w = gray.shape[:2]

        # Upscale prévio para imagens pequenas: bordas ficam mais detectáveis
        scale_factor = 1.0
        working_gray = gray
        if h < 40 and h > 0:
            scale_factor = 2.0
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            working_gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            logger.debug(f"GeometricNormalizer: Upscale 2× para detecção de cantos ({w}×{h} → {new_w}×{new_h})")

        wh, ww = working_gray.shape[:2]
        img_area = wh * ww

        # Pré-processamento para detecção de bordas
        blurred = cv2.GaussianBlur(working_gray, (5, 5), 0)

        # Tentar múltiplos thresholds de Canny para robustez
        for low_thresh, high_thresh in [(30, 100), (50, 150), (80, 200)]:
            edges = cv2.Canny(blurred, low_thresh, high_thresh)

            # Dilatar para conectar bordas quebradas
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

            corners = self._find_quadrilateral(edges, img_area, wh, ww)
            if corners is not None:
                # Escalar cantos de volta se houve upscale
                if scale_factor != 1.0:
                    corners = (corners / scale_factor).astype(np.float32)
                return corners

        # Fallback: tentar threshold adaptativo
        try:
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            corners = self._find_quadrilateral(thresh, img_area, wh, ww)
            if corners is not None:
                if scale_factor != 1.0:
                    corners = (corners / scale_factor).astype(np.float32)
                return corners
        except cv2.error as e:
            logger.debug(f"Falha no fallback de threshold adaptativo: {e}")
        except Exception as e:
            logger.debug(f"Erro inesperado ao detectar quadrilátero (fallback): {e}")

        return None

    def _find_quadrilateral(
        self, edges: np.ndarray, img_area: int, img_h: int, img_w: int
    ) -> Optional[np.ndarray]:
        """
        Encontra contorno quadrilateral com aspect ratio de placa.

        Args:
            edges: Imagem de bordas (binária)
            img_area: Área total da imagem
            img_h: Altura da imagem
            img_w: Largura da imagem

        Returns:
            Array 4×2 com cantos ordenados ou None
        """
        contours, _ = cv2.findContours(
            edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Ordenar por área decrescente, pegar os maiores
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        min_area = img_area * self.min_contour_area_ratio

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Aproximar polígono
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Queremos exatamente 4 vértices
            if len(approx) == 4:
                rect = cv2.boundingRect(approx)
                rx, ry, rw, rh = rect

                if rh == 0:
                    continue

                aspect_ratio = float(rw) / rh

                # Verificar aspect ratio de placa
                if self.MIN_PLATE_ASPECT_RATIO <= aspect_ratio <= self.MAX_PLATE_ASPECT_RATIO:
                    # Ordenar cantos: top-left, top-right, bottom-right, bottom-left
                    ordered = self._order_corners(approx.reshape(4, 2))
                    return ordered

        return None

    @staticmethod
    def _order_corners(pts: np.ndarray) -> np.ndarray:
        """
        Ordena 4 pontos em ordem: top-left, top-right, bottom-right, bottom-left.

        Usa a soma e diferença das coordenadas para determinar a posição.

        Args:
            pts: Array 4×2 com coordenadas dos pontos

        Returns:
            Array 4×2 com pontos ordenados
        """
        rect = np.zeros((4, 2), dtype=np.float32)

        # soma x+y: menor = top-left, maior = bottom-right
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right

        # diferença y-x: menor = top-right, maior = bottom-left
        diff = np.diff(pts, axis=1).flatten()
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left

        return rect

    def _apply_perspective_transform(
        self, image: np.ndarray, corners: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Aplica transformação de perspectiva (homografia) para retificar a placa.

        Args:
            image: Imagem original (BGR ou gray)
            corners: Array 4×2 com cantos ordenados (tl, tr, br, bl)

        Returns:
            Imagem retificada ou None se falhar
        """
        tl, tr, br, bl = corners

        # Calcular dimensões da placa retificada
        width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        max_width = max(int(width_top), int(width_bottom))

        height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        max_height = max(int(height_left), int(height_right))

        if max_width <= 0 or max_height <= 0:
            return None

        # Verificar aspect ratio pós-transformação
        aspect = max_width / max_height
        if aspect < self.MIN_PLATE_ASPECT_RATIO * 0.5 or aspect > self.MAX_PLATE_ASPECT_RATIO * 1.5:
            logger.debug(
                f"GeometricNormalizer: Aspect ratio pós-homografia inválido: {aspect:.2f}"
            )
            return None

        # Pontos de destino (retângulo perfeito)
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        # Calcular e aplicar homografia
        M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
        warped = cv2.warpPerspective(
            image, M, (max_width, max_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return warped

    # ──────────────────────────────────────────────────────────────────
    #  Etapa 2: Correção de Rotação
    # ──────────────────────────────────────────────────────────────────

    def _correct_rotation(
        self, gray: np.ndarray, image: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Corrige rotação da imagem usando detecção de linhas de Hough.

        Detecta linhas horizontais dominantes e calcula o ângulo mediano
        para aplicar rotação corretiva via warpAffine.

        Args:
            gray: Imagem grayscale
            image: Imagem original (BGR ou gray) para aplicar a rotação

        Returns:
            Imagem rotacionada ou None se não necessário/possível
        """
        # Detectar bordas
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detectar linhas com Hough Transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=30,
            minLineLength=gray.shape[1] // 4,
            maxLineGap=10
        )

        if lines is None or len(lines) < 2:
            return None

        # Calcular ângulo médio das linhas
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Filtrar apenas ângulos pequenos (< 30°) — inclinação real, não linhas verticais
            if abs(angle) < 30:
                angles.append(angle)

        if not angles:
            return None

        median_angle = float(np.median(angles))

        # Só corrigir se inclinação for significativa mas não excessiva
        if abs(median_angle) < self.min_rotation_angle or abs(median_angle) > self.max_rotation_angle:
            return None

        # Rotacionar imagem
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        logger.debug(f"GeometricNormalizer: Rotação corrigida = {median_angle:.1f}°")
        return rotated

    # ──────────────────────────────────────────────────────────────────
    #  Etapa 3: Equalização de Contraste
    # ──────────────────────────────────────────────────────────────────

    def _equalize_contrast(
        self, gray: np.ndarray, image: np.ndarray, is_color: bool
    ) -> Optional[np.ndarray]:
        """
        Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization)
        com parâmetros adaptativos baseados no histograma da imagem.

        Args:
            gray: Imagem grayscale (para análise de histograma)
            image: Imagem original para aplicar equalização
            is_color: Se a imagem é colorida

        Returns:
            Imagem com contraste equalizado
        """
        # Calcular clipLimit adaptativo
        clip_limit = self.clahe_clip_limit
        if clip_limit is None:
            clip_limit = self._compute_adaptive_clip_limit(gray)

        # tileGridSize adaptativo: tiles grandes em imagens pequenas causam artefatos
        h_img = gray.shape[0]
        if h_img < 50:
            effective_tile_grid = (4, 4)
        elif h_img < 100:
            effective_tile_grid = (6, 6)
        else:
            effective_tile_grid = self.clahe_tile_grid

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=effective_tile_grid
        )

        if is_color:
            # Converter para LAB, aplicar CLAHE no canal L
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            l_equalized = clahe.apply(l_channel)
            lab_equalized = cv2.merge([l_equalized, a_channel, b_channel])
            result = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)
        else:
            result = clahe.apply(image)

        logger.debug(f"GeometricNormalizer: CLAHE aplicado (clipLimit={clip_limit:.1f})")
        return result

    @staticmethod
    def _compute_adaptive_clip_limit(gray: np.ndarray) -> float:
        """
        Calcula clipLimit adaptativo para CLAHE baseado no histograma.

        Imagens escuras → clipLimit mais alto (mais contraste)
        Imagens claras/já contrastadas → clipLimit mais baixo

        Args:
            gray: Imagem grayscale

        Returns:
            Valor de clipLimit (entre 1.0 e 4.0)
        """
        mean_val = np.mean(gray)
        std_val = np.std(gray)

        # Imagem escura: precisa de mais contraste
        if mean_val < 80:
            clip_limit = 3.5
        elif mean_val < 120:
            clip_limit = 2.5
        else:
            clip_limit = 1.5

        # Desvio padrão alto = bom contraste → reduzir
        if std_val > 60:
            clip_limit = max(1.0, clip_limit - 0.5)
        elif std_val < 30:
            # Contraste muito baixo: aumentar
            clip_limit = min(4.0, clip_limit + 0.5)

        return clip_limit

    # ──────────────────────────────────────────────────────────────────
    #  Etapa 4: Redimensionamento Padronizado
    # ──────────────────────────────────────────────────────────────────

    def _standard_resize(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Redimensiona a imagem para dimensões padronizadas.

        Mantém o aspect ratio original da placa e adiciona padding
        se necessário para atingir as dimensões alvo.

        Args:
            image: Imagem a redimensionar

        Returns:
            Imagem redimensionada
        """
        h, w = image.shape[:2]

        if h == 0 or w == 0:
            return None

        # Calcular fator de escala para encaixar na dimensão alvo
        scale_w = self.target_width / w
        scale_h = self.target_height / h
        scale = min(scale_w, scale_h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        if new_w <= 0 or new_h <= 0:
            return None

        # Escolher interpolação baseada na direção do escalonamento
        if scale > 1.0:
            interpolation = cv2.INTER_CUBIC  # upscale
        else:
            interpolation = cv2.INTER_AREA  # downscale

        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

        # Adicionar padding para atingir dimensões alvo exatas
        pad_top = (self.target_height - new_h) // 2
        pad_bottom = self.target_height - new_h - pad_top
        pad_left = (self.target_width - new_w) // 2
        pad_right = self.target_width - new_w - pad_left

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            # Usar borda replicada para padding
            resized = cv2.copyMakeBorder(
                resized, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_REPLICATE
            )

        return resized
