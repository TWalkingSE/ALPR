# src/detector.py
"""
Módulo responsável pela detecção de placas em imagens e vídeos
usando modelos YOLOv11.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

# Configurar logging
logger = logging.getLogger(__name__)

class PlateDetector:
    """Classe para detecção de placas veiculares usando YOLOv11."""

    # DEFAULT_MODEL_PATH não é mais usado diretamente aqui, vem do config
    # DEFAULT_CONFIDENCE não é mais usado diretamente aqui, vem do config

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = 0.25, # Reduzido de 0.5 para detectar placas distantes/difíceis
        device: Optional[str] = None, # 'auto', 'cpu', 'cuda', 'mps', etc.
        enable_sahi: bool = True,  # SAHI (Sliced Inference) para placas pequenas
        sahi_slice_size: int = 640,
        sahi_overlap_ratio: float = 0.25,
        sahi_retry_confidence_threshold: float = 0.55,
        sahi_retry_area_ratio_threshold: float = 0.01,
        sahi_retry_large_image_threshold: int = 1600,
        sahi_merge_iou_threshold: float = 0.45,
    ):
        """
        Inicializa o detector de placas.
        
        Args:
            model_path: Caminho para o modelo YOLOv11 (.pt). Essencial.
            confidence: Limiar de confiança para detecções (0-1).
            device: Dispositivo para inferência ('auto', 'cpu', 'cuda', 'mps').
                    Se 'auto' ou None, tentará detectar o melhor disponível.
            enable_sahi: Ativar SAHI (Sliced Aided Hyper Inference) para detectar
                         placas pequenas em imagens grandes (câmeras de rodovia).
            sahi_slice_size: Tamanho de cada slice em pixels.
            sahi_overlap_ratio: Sobreposição entre slices (0-1).
        """
        if model_path is None:
            logger.error("O caminho do modelo (model_path) deve ser fornecido para PlateDetector.")
            raise ValueError("model_path não pode ser None")

        self.model_path = model_path
        self.confidence = confidence
        self.enable_sahi = enable_sahi
        self.sahi_slice_size = sahi_slice_size
        self.sahi_overlap_ratio = sahi_overlap_ratio
        self.sahi_retry_confidence_threshold = sahi_retry_confidence_threshold
        self.sahi_retry_area_ratio_threshold = sahi_retry_area_ratio_threshold
        self.sahi_retry_large_image_threshold = sahi_retry_large_image_threshold
        self.sahi_merge_iou_threshold = sahi_merge_iou_threshold
        self._is_loaded = False  # Atributo privado para controlar estado de carregamento

        if device and device.lower() == 'auto':
            self.device_str = self._get_auto_device()
        elif device:
            self.device_str = device
        else: # device is None or 'auto'
            self.device_str = self._get_auto_device()

        logger.info(f"PlateDetector: Usando dispositivo '{self.device_str}' para inferência.")

        self.model = None
        self._load_model()

    def _get_auto_device(self) -> str:
        """Detecta automaticamente o melhor dispositivo disponível ou usa CPU como fallback."""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"CUDA disponível. Usando GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
            except Exception:
                logger.info(f"CUDA disponível. Usando GPU: {gpu_name}")
            return "cuda"

        # Diagnóstico detalhado quando CUDA não está disponível
        logger.warning(
            "CUDA não disponível. Possíveis causas: "
            "(1) PyTorch instalado sem CUDA (versão +cpu) — reinstale com: "
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 | "
            "(2) Driver NVIDIA desatualizado | "
            "(3) Nenhuma GPU NVIDIA presente. "
            "Usando CPU."
        )
        return "cpu"

    def _load_model(self):
        """
        Carrega o modelo YOLO de um arquivo .pt
        """
        try:
            import torch
            from ultralytics import YOLO

            # Verificar se CUDA está realmente disponível
            cuda_available = torch.cuda.is_available()

            # Se device configurado para CUDA mas CUDA não disponível, usar CPU
            if self.device_str == 'cuda' and not cuda_available:
                logger.warning(
                    "Device 'cuda' solicitado mas CUDA não disponível. "
                    "Verifique se o PyTorch com CUDA está instalado: "
                    "pip install torch --index-url https://download.pytorch.org/whl/cu128"
                )
                self.device_str = 'cpu'

            model_file = Path(self.model_path)

            if not model_file.exists():
                logger.warning(f"Modelo {self.model_path} não encontrado. Tentando carregar diretamente pelo nome (pode tentar baixar)...")
                # Ultralytics pode tentar baixar modelos padrão se o nome for fornecido (ex: "yolo11n.pt")
                # No entanto, para caminhos customizados, isso falhará se o arquivo não existir.
                # A lógica de download explícito foi removida daqui para ser gerenciada externamente se necessário.
                # Se o model_path for um nome como "yolo11n", ele tentará baixar.
                # Se for um caminho como "models/yolo/meumodelo.pt", ele espera que exista.
                try:
                    self.model = YOLO(self.model_path) # Tenta carregar, pode baixar se for nome de modelo padrão
                except Exception as e_load:
                    logger.error(f"Falha ao carregar/baixar o modelo '{self.model_path}': {e_load}")
                    raise FileNotFoundError(f"Modelo {self.model_path} não encontrado e falha ao carregar/baixar.") from e_load
            else:
                logger.info(f"Carregando modelo de {self.model_path}")
                self.model = YOLO(self.model_path)

            # Mover modelo para o device correto com tratamento de erro
            try:
                self.model.to(self.device_str)
                self._is_loaded = True
                logger.info(f"✅ Modelo YOLOv11 carregado com sucesso no device: {self.device_str.upper()}")
            except (AssertionError, RuntimeError) as e:
                # Se falhou ao mover para CUDA, tentar CPU
                error_msg = str(e)
                if 'CUDA' in error_msg or 'cuda' in error_msg:
                    logger.warning(f"Falha ao usar CUDA: {e}. Tentando CPU...")
                    self.device_str = 'cpu'
                    self.model.to('cpu')
                    self._is_loaded = True
                    logger.info("✅ Modelo YOLOv11 carregado com CPU (fallback)")
                else:
                    raise

            logger.info(f"Modelo YOLO carregado com sucesso em '{self.model_path}' no dispositivo '{self.device_str}'")
            logger.info(f"SAHI (Sliced Inference): {'habilitado' if self.enable_sahi else 'desabilitado'} | slice={self.sahi_slice_size}px overlap={self.sahi_overlap_ratio:.0%}")

        except ImportError:
            logger.error("Falha ao importar YOLO. Verifique se 'ultralytics' está instalado.")
            raise
        except Exception as e:
            logger.error(f"Erro ao carregar modelo YOLO: {e}", exc_info=True)
            raise

    @property
    def is_loaded(self) -> bool:
        """Verifica se o modelo está carregado."""
        return self._is_loaded and self.model is not None

    def _ensure_bgr(self, image: np.ndarray) -> np.ndarray:
        """
        Garante que a imagem tem 3 canais BGR.
        
        Câmeras de vigilância/rodovia frequentemente produzem imagens
        grayscale (1 canal) ou com canal alpha (4 canais). O YOLO
        espera imagens BGR com 3 canais.
        """
        if len(image.shape) == 2:
            logger.info("Imagem grayscale (1 canal) detectada — convertendo para BGR")
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3:
            if image.shape[2] == 1:
                logger.info("Imagem 1 canal (HxWx1) detectada — convertendo para BGR")
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                logger.info("Imagem BGRA (4 canais) detectada — convertendo para BGR")
                return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    def _parse_yolo_results(self, results) -> List[Dict[str, Any]]:
        """
        Converte resultados brutos do YOLO em lista de dicts.
        
        Args:
            results: Resultados retornados pelo modelo YOLO
            
        Returns:
            Lista de detecções [{bbox, confidence, class_id, class_name}]
        """
        detections = []
        if not results:
            return detections

        for result_item in results:
            if hasattr(result_item, 'boxes') and result_item.boxes is not None:
                for box in result_item.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf_val = float(box.conf[0])

                    if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
                        logger.warning(f"Detecção com coordenadas inválidas ignorada: ({x1}, {y1}, {x2}, {y2})")
                        continue

                    class_id = 0
                    class_name = "plate"
                    if box.cls is not None and len(box.cls) > 0:
                        class_id = int(box.cls[0])
                        if hasattr(result_item, 'names') and result_item.names and class_id in result_item.names:
                            class_name = result_item.names[class_id]

                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "confidence": conf_val,
                        "class_id": class_id,
                        "class_name": class_name
                    })
        return detections

    def _detect_standard(self, image: np.ndarray, conf: float) -> List[Dict[str, Any]]:
        """Detecção YOLO padrão (imagem inteira)."""
        results = self.model(image, conf=conf, verbose=False)
        detections = self._parse_yolo_results(results)
        for det in detections:
            det.setdefault('source', 'full_frame')
        return detections

    def _detect_with_sahi(
        self, image: np.ndarray, conf: float
    ) -> List[Dict[str, Any]]:
        """
        Detecção com SAHI — Sliced Aided Hyper Inference.
        
        Divide a imagem em slices sobrepostos e executa o YOLO em cada um.
        Essencial para detectar placas pequenas em imagens grandes
        (câmeras de rodovia, vigilância).
        
        Após processar todos os slices, aplica NMS para remover duplicatas
        das regiões de sobreposição.
        """
        h, w = image.shape[:2]
        slice_h = self.sahi_slice_size
        slice_w = self.sahi_slice_size
        overlap = self.sahi_overlap_ratio

        step_h = int(slice_h * (1 - overlap))
        step_w = int(slice_w * (1 - overlap))

        # Se a imagem é menor que o slice, SAHI não faz sentido
        if h <= slice_h and w <= slice_w:
            logger.debug("Imagem menor que slice_size — SAHI desnecessário")
            return []

        all_detections = []
        slice_count = 0

        for y in range(0, h, step_h):
            for x in range(0, w, step_w):
                x1 = x
                y1 = y
                x2 = min(x + slice_w, w)
                y2 = min(y + slice_h, h)

                # Ignorar slices muito pequenos nas bordas
                if (x2 - x1) < slice_w * 0.4 or (y2 - y1) < slice_h * 0.4:
                    continue

                slice_img = image[y1:y2, x1:x2]
                slice_count += 1

                results = self.model(slice_img, conf=conf, verbose=False)
                slice_dets = self._parse_yolo_results(results)

                # Mapear coordenadas do slice para a imagem original
                for det in slice_dets:
                    bx1, by1, bx2, by2 = det['bbox']
                    det['bbox'] = (bx1 + x1, by1 + y1, bx2 + x1, by2 + y1)
                    det['source'] = 'sahi'
                    all_detections.append(det)

        logger.info(f"SAHI: {slice_count} slices processados, {len(all_detections)} detecções brutas")

        # NMS para remover duplicatas entre slices sobrepostos
        if len(all_detections) > 1:
            all_detections = self._nms_detections(all_detections)
            logger.info(f"SAHI após NMS: {len(all_detections)} detecções")

        return all_detections

    def _nms_detections(
        self, detections: List[Dict[str, Any]], iou_threshold: float = 0.45
    ) -> List[Dict[str, Any]]:
        """Aplica Non-Maximum Suppression para eliminar duplicatas."""
        if not detections:
            return []

        boxes = np.array([d['bbox'] for d in detections], dtype=np.float32)
        scores = np.array([d['confidence'] for d in detections], dtype=np.float32)

        # cv2.dnn.NMSBoxes espera (x, y, w, h)
        nms_boxes = [(int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])) for b in boxes]

        indices = cv2.dnn.NMSBoxes(
            bboxes=nms_boxes,
            scores=scores.tolist(),
            score_threshold=0.1,
            nms_threshold=iou_threshold
        )

        if len(indices) == 0:
            return []

        if isinstance(indices, np.ndarray):
            indices = indices.flatten()

        return [detections[i] for i in indices]

    def _should_retry_with_sahi(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> tuple[bool, str]:
        if not self.enable_sahi:
            return False, ''

        if not detections:
            return True, 'no_detections'

        image_h, image_w = image.shape[:2]
        if max(image_h, image_w) < self.sahi_retry_large_image_threshold:
            return False, ''

        best_confidence = max(float(det.get('confidence', 0.0)) for det in detections)
        if best_confidence < self.sahi_retry_confidence_threshold:
            return True, 'low_confidence'

        image_area = max(1, image_h * image_w)
        largest_area_ratio = max(
            (
                max(0, int(det['bbox'][2]) - int(det['bbox'][0]))
                * max(0, int(det['bbox'][3]) - int(det['bbox'][1]))
            )
            / image_area
            for det in detections
        )
        if largest_area_ratio <= self.sahi_retry_area_ratio_threshold:
            return True, 'small_plate_candidate'

        return False, ''

    def detect(
        self,
        image: np.ndarray,
        confidence: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Detecta placas em uma única imagem.
        
        Primeiro tenta detecção padrão (imagem inteira). Se não encontrar
        nenhuma placa e SAHI estiver habilitado, divide a imagem em slices
        sobrepostos para detectar placas pequenas/distantes.
        
        Args:
            image: Array NumPy contendo a imagem (BGR ou grayscale)
            confidence: Opcional - sobrescreve o limiar de confiança padrão
            
        Returns:
            Lista de detecções, cada uma com coordenadas e confiança
        """
        if not self.is_loaded:
            logger.error("Modelo YOLO não carregado. Não é possível detectar.")
            raise RuntimeError("Modelo não carregado")

        if image is None or image.size == 0:
            logger.warning("Imagem vazia recebida para detecção")
            return []

        current_conf = confidence if confidence is not None else self.confidence

        # === Garantir BGR (3 canais) — essencial para imagens de câmeras IR/grayscale ===
        processed = self._ensure_bgr(image)

        try:
            # === Tentativa 1: Detecção padrão (imagem inteira) ===
            standard_detections = self._detect_standard(processed, current_conf)

            retry_sahi, retry_reason = self._should_retry_with_sahi(processed, standard_detections)
            detections = list(standard_detections)

            # === Tentativa 2: SAHI (sliced inference) quando necessário ===
            if retry_sahi:
                logger.info(
                    "Executando SAHI (Sliced Aided Hyper Inference) "
                    f"por motivo '{retry_reason}' "
                    f"com slices de {self.sahi_slice_size}px e {self.sahi_overlap_ratio:.0%} overlap"
                )
                sahi_conf = max(0.05, current_conf - 0.05)
                sahi_detections = self._detect_with_sahi(processed, sahi_conf)
                for det in detections:
                    det.setdefault('retry_reason', retry_reason)
                for det in sahi_detections:
                    det['retry_reason'] = retry_reason
                if detections and sahi_detections:
                    detections = self._nms_detections(
                        detections + sahi_detections,
                        iou_threshold=self.sahi_merge_iou_threshold,
                    )
                elif sahi_detections:
                    detections = sahi_detections

            if detections:
                logger.info(f"{len(detections)} placa(s) detectada(s)")
            else:
                logger.info("Nenhuma placa detectada na imagem (padrão + SAHI)")

            return detections

        except Exception as e:
            logger.error(f"Erro durante a detecção YOLO: {e}", exc_info=True)
            return []

    # Altura mínima do crop para OCR confiável (pixels)
    MIN_PLATE_HEIGHT_FOR_OCR = 80
    # Limiar abaixo do qual a margem adaptativa é aplicada
    SMALL_PLATE_HEIGHT_THRESHOLD = 60

    def extract_plate_regions(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        add_margin: float = 0.10 # Default aumentado (era 0.05)
    ) -> List[Dict[str, Any]]:
        """
        Extrai as regiões de placa da imagem original com base nas detecções.
        
        Aplica margem adaptativa (maior para placas pequenas/distantes) e
        upscale automático de crops muito pequenos para garantir qualidade OCR.
        
        Args:
            image: Imagem original
            detections: Lista de detecções retornadas pelo método detect()
            add_margin: Margem base para recorte (0.10 = 10% de margem)
            
        Returns:
            Lista de regiões de placa com imagens recortadas
        """
        if image is None or image.size == 0:
            logger.warning("Imagem vazia recebida para extração de regiões")
            return []

        img_h, img_w = image.shape[:2]
        plate_regions = []

        for detection in detections:
            try:
                bbox = detection["bbox"]
                x1, y1, x2, y2 = bbox

                plate_w, plate_h = x2 - x1, y2 - y1

                # ── Margem adaptativa: placas pequenas recebem margem maior ──
                # Câmeras de vigilância produzem crops pequenos onde caracteres
                # das bordas são facilmente cortados com margem fixa de 5%.
                if plate_h < self.SMALL_PLATE_HEIGHT_THRESHOLD and plate_h > 0:
                    # Escala inversamente proporcional ao tamanho (quanto menor, maior a margem)
                    adaptive_factor = min(0.25, 0.15 * (self.SMALL_PLATE_HEIGHT_THRESHOLD / plate_h))
                    effective_margin = max(add_margin, adaptive_factor)
                    logger.debug(
                        f"Margem adaptativa: {effective_margin:.0%} "
                        f"(placa pequena: {plate_w}×{plate_h}px)"
                    )
                else:
                    effective_margin = add_margin

                margin_x = int(plate_w * effective_margin)
                margin_y = int(plate_h * effective_margin)

                x1_margin = max(0, x1 - margin_x)
                y1_margin = max(0, y1 - margin_y)
                x2_margin = min(img_w, x2 + margin_x)
                y2_margin = min(img_h, y2 + margin_y)

                if x1_margin >= x2_margin or y1_margin >= y2_margin:
                    logger.warning(f"Recorte inválido (após margem) ignorado: ({x1_margin}, {y1_margin}, {x2_margin}, {y2_margin})")
                    continue

                plate_img_cropped = image[y1_margin:y2_margin, x1_margin:x2_margin]

                if plate_img_cropped.size == 0:
                    logger.warning("Recorte de placa resultou em imagem vazia")
                    continue

                # ── Upscale de crops pequenos para resolução mínima OCR ──
                # Placas distantes/pequenas (~20-40px) são irrecuperáveis pelo OCR.
                # Upscale com LANCZOS4 preserva detalhes melhor que INTER_CUBIC.
                crop_h, crop_w = plate_img_cropped.shape[:2]
                if crop_h < self.MIN_PLATE_HEIGHT_FOR_OCR and crop_h > 0:
                    scale = self.MIN_PLATE_HEIGHT_FOR_OCR / crop_h
                    new_w = int(crop_w * scale)
                    new_h = int(crop_h * scale)
                    if new_w > 0 and new_h > 0:
                        plate_img_cropped = cv2.resize(
                            plate_img_cropped, (new_w, new_h),
                            interpolation=cv2.INTER_LANCZOS4
                        )
                        logger.info(
                            f"Upscale do crop: {crop_w}×{crop_h} → "
                            f"{new_w}×{new_h}px (×{scale:.1f}) via LANCZOS4"
                        )

                plate_region_data = {
                    "image": plate_img_cropped,
                    "bbox": (x1_margin, y1_margin, x2_margin, y2_margin), # Bbox com margem
                    "original_bbox": bbox, # Bbox original da detecção YOLO
                    "confidence": detection["confidence"], # Confiança da detecção YOLO
                    "detector_metadata": {
                        "source": detection.get("source", "full_frame"),
                        "retry_reason": detection.get("retry_reason", ""),
                    },
                }
                plate_regions.append(plate_region_data)
            except Exception as e:
                logger.error(f"Erro ao extrair região da placa: {e}", exc_info=True)
                continue

        return plate_regions

    # O método process_frame foi removido daqui, pois essa lógica de combinar
    # detecção e anotação agora reside no AgentOrchestrator ou no script principal/UI.
    # O PlateDetector foca apenas em carregar o modelo e realizar detecções.

    @staticmethod
    def list_available_models(models_dir: str = "models/yolo", recursive: bool = True) -> List[str]:
        # (Implementação original mantida)
        abs_path = os.path.abspath(models_dir)
        logger.debug(f"Procurando modelos em: {abs_path} {'(recursivamente)' if recursive else ''}")

        if not os.path.exists(models_dir):
            # os.makedirs(models_dir, exist_ok=True) # Não criar aqui, apenas listar
            logger.warning(f"Diretório de modelos {models_dir} não encontrado.")
            return []

        models = []
        if recursive:
            for root, _, files in os.walk(models_dir):
                for file in files:
                    if file.endswith('.pt'):
                        models.append(os.path.join(root, file))
        else:
            try:
                files = os.listdir(models_dir)
                models = [os.path.join(models_dir, f) for f in files if f.endswith(".pt")]
            except FileNotFoundError:
                logger.warning(f"Diretório {models_dir} não encontrado ao listar modelos não recursivamente.")
                return []

        logger.debug(f"Modelos .pt encontrados em {models_dir}: {len(models)}")
        return models

