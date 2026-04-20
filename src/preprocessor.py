# src/preprocessor.py
"""
Módulo responsável pelo pré-processamento de imagens para melhorar
a detecção e reconhecimento de placas, incluindo funcionalidades de blur.
"""

import logging
from typing import Any, List, Optional

import cv2
import numpy as np

# import os # Removido, não usado diretamente aqui
# import imutils # Removido - não usado no código

# Configurar logging
# logging.basicConfig( # Configuração de logging movida para o main.py ou orquestrador
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Classe para pré-processamento de imagens de placas veiculares."""

    def __init__(
        self,
        enhance_contrast: bool = True,
        remove_noise: bool = True,
        sharpen: bool = True,
        adaptive_threshold: bool = True,
        optimize_for_brazilian_plates: bool = True,
        morphological_cleanup: bool = True,
        deskew: bool = True,
        multi_binarization: bool = True,
        adaptive_clahe: bool = True,
        use_nlmeans_denoising: bool = True,
    ):
        """
        Inicializa o preprocessador de imagens.

        Args:
            enhance_contrast: Se deve aplicar técnicas de melhoria de contraste
            remove_noise: Se deve aplicar técnicas de remoção de ruído
            sharpen: Se deve aplicar técnicas de nitidez
            adaptive_threshold: Se deve aplicar threshold adaptativo
            optimize_for_brazilian_plates: Se deve otimizar para placas brasileiras (Mercosul/antigas)
            morphological_cleanup: Se deve limpar com operações morfológicas após binarização
            deskew: Se deve corrigir inclinação da placa via Hough Lines
            multi_binarization: Se deve gerar múltiplas binarizações (Otsu + Adaptativo)
            adaptive_clahe: Se deve adaptar parâmetros CLAHE ao histograma
            use_nlmeans_denoising: Se deve usar fastNlMeansDenoising ao invés de bilateral

        Nota: A correção de perspectiva foi migrada para ``GeometricNormalizer``
        (``src/geometric_normalizer.py``), que é chamado antes do preprocessor no
        pipeline.
        """
        self.enhance_contrast = enhance_contrast
        self.remove_noise = remove_noise
        self.sharpen = sharpen
        self.adaptive_threshold = adaptive_threshold
        self.optimize_for_brazilian_plates = optimize_for_brazilian_plates
        self.morphological_cleanup = morphological_cleanup
        self.deskew = deskew
        self.multi_binarization = multi_binarization
        self.adaptive_clahe = adaptive_clahe
        self.use_nlmeans_denoising = use_nlmeans_denoising

        logger.info(f"ImagePreprocessor inicializado com config: enhance_contrast={enhance_contrast}, "
                    f"remove_noise={remove_noise}, sharpen={sharpen}, adaptive_threshold={adaptive_threshold}, "
                    f"morphological_cleanup={morphological_cleanup}, deskew={deskew}, "
                    f"multi_binarization={multi_binarization}, adaptive_clahe={adaptive_clahe}")

    def process(self, image: np.ndarray, quality_result: Optional[Any] = None) -> List[np.ndarray]:
        """
        Aplica pré-processamento à imagem.
        Retorna uma lista de imagens processadas, sendo a última geralmente a mais otimizada para OCR.
        A primeira imagem na lista é sempre a original.

        Args:
            image: Imagem original (numpy array BGR)
            quality_result: Resultado opcional de ImageQualityAssessor. Quando fornecido,
                ajusta dinamicamente o número e intensidade das variantes geradas:
                  - Excelente (>= 0.75): conjunto mínimo (original + CLAHE + Otsu) — rápido
                  - Suficiente (>= 0.50): fluxo padrão (todas as binarizações)
                  - Crítica (>= 0.25): fluxo padrão + augmentation agressivo
                  - Insuficiente (< 0.25): fluxo agressivo máximo (sharpen forte + todas as variantes)

        Returns:
            Lista com versões processadas da imagem (a primeira é a original)
        """
        if image is None or image.size == 0:
            logger.warning("Imagem vazia recebida para pré-processamento no ImagePreprocessor.")
            return []

        # ------------------------------------------------------------------
        # Modo adaptativo: ajustar flags com base na qualidade da imagem
        # ------------------------------------------------------------------
        adaptive_mode = 'default'
        enable_multi_bin = self.multi_binarization
        enable_augmentation = True
        aggressive_sharpen = False
        denoise_boost = False
        motion_blur_high = False
        if quality_result is not None:
            score = getattr(quality_result, 'quality_score', None)
            snr = getattr(quality_result, 'snr', None)
            motion_blur = getattr(quality_result, 'motion_blur', None)
            if score is None:
                score = 0.5
            if score >= 0.75:
                # Imagem excelente — poucas variantes são suficientes
                adaptive_mode = 'excelente'
                enable_multi_bin = False
                enable_augmentation = False
            elif score >= 0.50:
                adaptive_mode = 'suficiente'
            elif score >= 0.25:
                adaptive_mode = 'critica'
                enable_multi_bin = True
                enable_augmentation = True
            else:
                adaptive_mode = 'insuficiente'
                enable_multi_bin = True
                enable_augmentation = True
                aggressive_sharpen = True
            if snr is not None and snr < 9.0:
                adaptive_mode = f'{adaptive_mode}_low_snr'
                enable_multi_bin = True
                enable_augmentation = True
                denoise_boost = True
            if motion_blur is not None and motion_blur > 0.35:
                adaptive_mode = f'{adaptive_mode}_motion'
                enable_multi_bin = True
                enable_augmentation = True
                aggressive_sharpen = True
                motion_blur_high = True
            logger.debug(
                f"Preprocessor adaptativo: modo={adaptive_mode} (score={score:.2f}), "
                f"multi_bin={enable_multi_bin}, augmentation={enable_augmentation}"
            )
        self._adaptive_mode = adaptive_mode
        self._enable_multi_bin_runtime = enable_multi_bin
        self._enable_augmentation_runtime = enable_augmentation
        self._aggressive_sharpen_runtime = aggressive_sharpen
        self._denoise_boost_runtime = denoise_boost
        self._motion_blur_high_runtime = motion_blur_high

        original = image.copy()
        processed_images = [original]

        # Converter para escala de cinza se for colorida
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2: # Já é cinza
            gray = image.copy()
        else:
            logger.warning(f"Formato de imagem inesperado no pré-processador: {image.shape}")
            return [original] # Retorna original se não for BGR ou GRAY

        current_processed = gray.copy() # Imagem que será modificada sequencialmente

        try:
            # NOTA: Correção de inclinação (deskew) e perspectiva foram migradas
            # para o GeometricNormalizer (src/geometric_normalizer.py).
            # O deskew agora é aplicado ANTES do pré-processamento, entre detecção e OCR.

            # 1. Redimensionar se necessário (para OCR, um tamanho mínimo pode ajudar)
            h, w = current_processed.shape[:2]
            min_height_for_ocr = 64  # Altura mínima para OCR confiável (era 30)
            self._input_height = h   # Guardar altura original para parâmetros adaptativos
            if h < min_height_for_ocr and h > 0: # Evitar divisão por zero
                scale_factor = min_height_for_ocr / h
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                if new_w > 0 and new_h > 0:
                    # LANCZOS4 preserva detalhes melhor que INTER_CUBIC para upscale
                    current_processed = cv2.resize(current_processed, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                    logger.debug(f"Redimensionado para OCR: {current_processed.shape} (LANCZOS4)")

            # 2. Melhorar contraste (CLAHE adaptativo)
            if self.enhance_contrast:
                # Tiles adaptativos: imagens pequenas precisam de tiles menores
                # para evitar artefatos de bloco (8×8 em imagem de 40px = 5px/tile → ruim)
                input_h = getattr(self, '_input_height', h)
                if input_h < 50:
                    tile_grid = (4, 4)
                elif input_h < 100:
                    tile_grid = (6, 6)
                else:
                    tile_grid = (8, 8)

                if self.adaptive_clahe:
                    # Adaptar clipLimit ao histograma da imagem
                    clip_limit = self._compute_adaptive_clip_limit(current_processed)
                    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
                    logger.debug(f"CLAHE adaptativo: clipLimit={clip_limit:.1f}, tiles={tile_grid}")
                else:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=tile_grid)
                current_processed = clahe.apply(current_processed)
                processed_images.append(current_processed.copy())
                logger.debug("Contraste melhorado com CLAHE.")

            # 3. Remover ruído (parâmetros adaptativos ao tamanho)
            if self.remove_noise:
                input_h = getattr(self, '_input_height', h)
                if self.use_nlmeans_denoising:
                    # Non-Local Means preserva bordas melhor que bilateral
                    # h adaptativo: valores altos destroem traços finos em placas pequenas
                    # Valores reduzidos para preservar caracteres como 1, I, J, 7
                    if input_h < 50:
                        nlm_h, tmpl_w, srch_w = 3, 5, 13
                    elif input_h < 80:
                        nlm_h, tmpl_w, srch_w = 4, 5, 15
                    else:
                        nlm_h, tmpl_w, srch_w = 7, 7, 21
                    if getattr(self, '_denoise_boost_runtime', False):
                        nlm_h += 2
                    current_processed = cv2.fastNlMeansDenoising(
                        current_processed, None, h=nlm_h, templateWindowSize=tmpl_w, searchWindowSize=srch_w
                    )
                    logger.debug(f"Ruído removido com fastNlMeansDenoising (h={nlm_h}).")
                else:
                    # Bilateral filter — parâmetros também adaptativos
                    if input_h < 60:
                        current_processed = cv2.bilateralFilter(current_processed, d=5, sigmaColor=50, sigmaSpace=50)
                    else:
                        current_processed = cv2.bilateralFilter(current_processed, d=9, sigmaColor=75, sigmaSpace=75)
                    logger.debug("Ruído removido com Bilateral Filter.")
                processed_images.append(current_processed.copy())

            # 4. Melhorar nitidez (unsharp mask em vez de kernel agressivo)
            if self.sharpen:
                # Unsharp mask: mais suave que o kernel [-1,-1,-1; -1,9,-1; -1,-1,-1]
                gaussian = cv2.GaussianBlur(current_processed, (0, 0), 3)
                # Em modo insuficiente, usamos sharpen mais forte
                if getattr(self, '_aggressive_sharpen_runtime', False):
                    current_processed = cv2.addWeighted(current_processed, 2.0, gaussian, -1.0, 0)
                    logger.debug("Nitidez AGRESSIVA aplicada (unsharp mask forte — modo insuficiente).")
                else:
                    current_processed = cv2.addWeighted(current_processed, 1.5, gaussian, -0.5, 0)
                    logger.debug("Nitidez aplicada (unsharp mask).")
                processed_images.append(current_processed.copy())

                if getattr(self, '_motion_blur_high_runtime', False):
                    directional_kernel = np.array([[0, -1, 0], [-1, 6, -1], [0, -1, 0]])
                    current_processed = cv2.filter2D(current_processed, -1, directional_kernel)
                    processed_images.append(current_processed.copy())
                    logger.debug("Kernel extra aplicado para compensar motion blur.")

            # 5. Aplicar thresholding (geração de múltiplas binarizações)
            if self.adaptive_threshold:
                # blockSize adaptativo: blocos grandes (11) em imagens pequenas
                # fazem caracteres fundirem ou fragmentarem
                input_h = getattr(self, '_input_height', h)
                if input_h < 50:
                    block_gauss, block_mean, c_val = 7, 9, 3
                elif input_h < 80:
                    block_gauss, block_mean, c_val = 9, 11, 2
                else:
                    block_gauss, block_mean, c_val = 11, 15, 2

                # Binarização principal: Gaussian adaptativo
                thresh_gauss = cv2.adaptiveThreshold(
                    current_processed, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, block_gauss, c_val
                )

                # Limpeza morfológica pós-binarização
                if self.morphological_cleanup:
                    thresh_gauss = self._morphological_cleanup(thresh_gauss)

                processed_images.append(thresh_gauss.copy())
                current_processed = thresh_gauss
                logger.debug("Threshold adaptativo Gaussiano (invertido) aplicado.")

                # Binarizações alternativas para OCR multi-variante
                if enable_multi_bin:
                    # Otsu threshold (invertido)
                    try:
                        pre_thresh = processed_images[-2] if len(processed_images) >= 2 else gray
                        _, thresh_otsu = cv2.threshold(
                            pre_thresh, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                        )
                        if self.morphological_cleanup:
                            thresh_otsu = self._morphological_cleanup(thresh_otsu)
                        processed_images.append(thresh_otsu.copy())
                        logger.debug("Binarização Otsu (invertida) gerada.")
                    except Exception as e:
                        logger.debug(f"Erro na binarização Otsu: {e}")

                    # Adaptive Mean threshold (invertido)
                    try:
                        pre_thresh = processed_images[-3] if len(processed_images) >= 3 else gray
                        thresh_mean = cv2.adaptiveThreshold(
                            pre_thresh, 255,
                            cv2.ADAPTIVE_THRESH_MEAN_C,
                            cv2.THRESH_BINARY_INV, block_mean, 4
                        )
                        if self.morphological_cleanup:
                            thresh_mean = self._morphological_cleanup(thresh_mean)
                        processed_images.append(thresh_mean.copy())
                        logger.debug("Binarização Mean (invertida) gerada.")
                    except Exception as e:
                        logger.debug(f"Erro na binarização Mean: {e}")

                    # ====== VARIANTES NÃO-INVERTIDAS (texto escuro em fundo claro) ======
                    # Engines OCR (GLM / OLMoOCR2 e a maioria dos OCRs em geral)
                    # esperam texto escuro sobre fundo claro. Binarização BINARY
                    # (não invertida) garante compatibilidade universal.
                    try:
                        pre_thresh = processed_images[-4] if len(processed_images) >= 4 else gray
                        # Gaussian não-invertido
                        thresh_gauss_normal = cv2.adaptiveThreshold(
                            pre_thresh, 255,
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, block_gauss, c_val
                        )
                        if self.morphological_cleanup:
                            thresh_gauss_normal = self._morphological_cleanup(thresh_gauss_normal)
                        processed_images.append(thresh_gauss_normal.copy())
                        logger.debug("Binarização Gaussian (não-invertida) gerada.")
                    except Exception as e:
                        logger.debug(f"Erro na binarização Gaussian não-invertida: {e}")

                    try:
                        pre_thresh = processed_images[-5] if len(processed_images) >= 5 else gray
                        # Otsu não-invertido
                        _, thresh_otsu_normal = cv2.threshold(
                            pre_thresh, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU
                        )
                        if self.morphological_cleanup:
                            thresh_otsu_normal = self._morphological_cleanup(thresh_otsu_normal)
                        processed_images.append(thresh_otsu_normal.copy())
                        logger.debug("Binarização Otsu (não-invertida) gerada.")
                    except Exception as e:
                        logger.debug(f"Erro na binarização Otsu não-invertida: {e}")

            # As operações abaixo (perspectiva, cor, etc.) podem ser menos genéricas
            # e são mantidas opcionais e desabilitadas por padrão.

            # NOTA: Correção de perspectiva migrada para GeometricNormalizer

            # Otimizações específicas para placas brasileiras (se habilitado)
            # Estas podem gerar imagens alternativas
            if self.optimize_for_brazilian_plates and len(original.shape) == 3:
                try:
                    mercosul_optimized = self._optimize_for_mercosul(original)
                    if mercosul_optimized is not None:
                        processed_images.append(mercosul_optimized)
                        logger.debug("Otimização para Mercosul aplicada.")

                    antigas_optimized = self._optimize_for_antigas(original)
                    if antigas_optimized is not None:
                        processed_images.append(antigas_optimized)
                        logger.debug("Otimização para placas antigas aplicada.")
                except Exception as e:
                    logger.debug(f"Erro na otimização para placas brasileiras: {e}")

            # Data augmentation: rotação leve e correção de gamma
            # Ajuda OCR em placas com ângulo ou iluminação irregular
            if enable_augmentation:
                try:
                    aug_base = gray if len(original.shape) == 2 else cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

                    # Rotação leve (±2°) — corrige inclinação residual
                    h_aug, w_aug = aug_base.shape[:2]
                    for angle in (-2.0, 2.0):
                        center = (w_aug // 2, h_aug // 2)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        rotated = cv2.warpAffine(aug_base, M, (w_aug, h_aug),
                                                flags=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_REPLICATE)
                        processed_images.append(rotated)

                    # Gamma correction (0.7 e 1.4) — compensa sub/sobre-exposição
                    for gamma in (0.7, 1.4):
                        gamma_corrected = self._adjust_gamma(aug_base, gamma)
                        processed_images.append(gamma_corrected)

                    logger.debug("Data augmentation: 2 rotações + 2 gamma geradas.")
                except Exception as e:
                    logger.debug(f"Erro no data augmentation: {e}")
            else:
                logger.debug("Data augmentation desabilitado (modo adaptativo=excelente).")

            logger.info(f"ImagePreprocessor gerou {len(processed_images)} versões da imagem.")
            return processed_images

        except Exception as e:
            logger.error(f"Erro no pré-processamento da imagem (ImagePreprocessor): {e}", exc_info=True)
            return [original] # Retorna pelo menos a imagem original em caso de erro

    def _deskew_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Corrige inclinação da imagem usando detecção de linhas de Hough.
        
        Args:
            image: Imagem grayscale
            
        Returns:
            Imagem corrigida ou None se correção não for possível/necessária
        """
        try:
            # Detectar bordas
            edges = cv2.Canny(image, 50, 150, apertureSize=3)

            # Detectar linhas com Hough Transform
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=30,
                minLineLength=image.shape[1] // 4,
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

            median_angle = np.median(angles)

            # Só corrigir se inclinação for significativa (> 0.5°) mas não excessiva (< 15°)
            if abs(median_angle) < 0.5 or abs(median_angle) > 15:
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

            logger.debug(f"Deskew: ângulo corrigido = {median_angle:.1f}°")
            return rotated

        except Exception as e:
            logger.debug(f"Erro no deskew: {e}")
            return None

    @staticmethod
    def _compute_adaptive_clip_limit(image: np.ndarray) -> float:
        """
        Calcula clipLimit adaptativo para CLAHE baseado no histograma da imagem.
        
        Imagens escuras → clipLimit mais alto (mais contraste)
        Imagens claras/já contrastadas → clipLimit mais baixo (menos artefatos)
        
        Args:
            image: Imagem grayscale
            
        Returns:
            Valor de clipLimit recomendado (entre 1.0 e 4.0)
        """
        mean_val = np.mean(image)
        std_val = np.std(image)

        # Imagem escura (média baixa): precisa de mais contraste
        if mean_val < 80:
            clip_limit = 3.5
        elif mean_val < 120:
            clip_limit = 2.5
        else:
            clip_limit = 1.5

        # Se desvio padrão já é alto (bom contraste), reduzir clipLimit
        if std_val > 60:
            clip_limit = max(1.0, clip_limit - 0.5)
        elif std_val < 30:
            # Contraste muito baixo: aumentar
            clip_limit = min(4.0, clip_limit + 0.5)

        return clip_limit

    def _morphological_cleanup(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Aplica operações morfológicas para limpar imagem binarizada.
        
        Parâmetros adaptativos ao tamanho da imagem original:
        - Imagens pequenas (<60px): pula MORPH_OPEN (destrói chars finos 1,I,L,7)
          e usa kernel (1,1) para CLOSE
        - Imagens grandes: MORPH_OPEN + MORPH_CLOSE com kernel (2,2)
        
        Args:
            binary_image: Imagem binarizada (preto e branco)
            
        Returns:
            Imagem limpa
        """
        input_h = getattr(self, '_input_height', binary_image.shape[0])

        if input_h < 60:
            # Imagem pequena: caracteres finos (1, I, L, 7) têm apenas 1-2px de largura.
            # MORPH_OPEN com kernel 2×2 os destrói. Apenas fechar gaps suavemente.
            kernel_close = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_close, iterations=1)
            logger.debug("Morphology adaptativa: skip OPEN, CLOSE (1,1) — imagem pequena")
        elif input_h < 100:
            # Imagem média: MORPH_OPEN (2,2) pode fundir traços finos (1→3).
            # Usar kernel retangular vertical (1,2) para OPEN: preserva traços verticais
            # como '1', '7', 'I', 'J', enquanto remove ruído horizontal.
            kernel_open = np.ones((1, 2), np.uint8)  # Não afetar colunas verticais
            kernel_close = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_open, iterations=1)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)
            logger.debug("Morphology adaptativa: OPEN (1,2) + CLOSE (2,2) — imagem média")
        else:
            # Imagem normal: pipeline padrão
            kernel_open = np.ones((2, 2), np.uint8)
            kernel_close = np.ones((2, 2), np.uint8)
            # Remover ruído (pontos isolados)
            cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_open, iterations=1)
            # Fechar gaps em caracteres
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        return cleaned

    def _adjust_gamma(self, image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def _optimize_for_mercosul(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Otimização específica para placas Mercosul: isola faixa azul + binariza."""
        try:
            if len(image.shape) != 3:
                return None
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([140, 255, 255])
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(max_contour)
                if w > 30 and h > 15:
                    roi = image[y:y + h, x:x + w]
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(
                        gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                    )
                    return thresh
        except Exception as e:
            logger.debug(f"Erro otimização Mercosul: {e}")
        return None

    def _optimize_for_antigas(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Otimização específica para placas antigas: equalização + threshold adaptativo."""
        try:
            if len(image.shape) != 3:
                return None
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            thresh = cv2.adaptiveThreshold(
                equalized, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            return opening
        except Exception as e:
            logger.debug(f"Erro otimização placas antigas: {e}")
        return None

