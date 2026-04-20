# src/video_processor.py
"""
Módulo de processamento de vídeo para ALPR.
Suporte a MP4, AVI, MOV, MKV, WMV, WEBM e DAV.

Extrai frames do vídeo, processa cada frame pelo pipeline LPR,
agrega resultados e gera vídeo anotado com placas detectadas.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.constants import CHAR_CONFIDENCE_THRESHOLD, MAX_UNIQUE_PLATES

try:
    from src.temporal_voting import TemporalVotingEngine
except ImportError:
    TemporalVotingEngine = None

logger = logging.getLogger(__name__)


# ==================== Modo de análise do veículo ====================

class VehicleMode(str, Enum):
    """Modo de análise do veículo no vídeo."""
    STATIONARY = 'stationary'  # Carro parado
    MOVING = 'moving'          # Carro em movimento


# Parâmetros otimizados por modo
# skip_frames: pular N frames entre processamentos
# confidence_early_stop: se atingir esta confiança, reduz processamento restante
# min_stable_detections: detecções consecutivas iguais para early-stop (parado)
# sharpness_filter: usar filtro de nitidez para selecionar frames (parado)
VEHICLE_MODE_PARAMS = {
    VehicleMode.STATIONARY: {
        'skip_frames': 5,
        'confidence_early_stop': 0.85,
        'min_stable_detections': 3,
        'sharpness_filter': True,
        'sharpness_threshold': 50.0,
        'description': 'Veículo parado: amostra menos frames, prioriza nitidez e aplica early-stop ao atingir alta confiança.',
    },
    VehicleMode.MOVING: {
        'skip_frames': 2,
        'confidence_early_stop': 0.0,  # Sem early-stop
        'min_stable_detections': 0,
        'sharpness_filter': False,
        'sharpness_threshold': 0.0,
        'description': 'Veículo em movimento: amostra mais frames para capturar a placa em diferentes posições.',
    },
}

# Extensões de vídeo suportadas
SUPPORTED_VIDEO_EXTENSIONS = {
    '.mp4': 'MP4 (MPEG-4)',
    '.avi': 'AVI (Audio Video Interleave)',
    '.mov': 'MOV (QuickTime)',
    '.mkv': 'MKV (Matroska)',
    '.wmv': 'WMV (Windows Media Video)',
    '.webm': 'WEBM (Web Media)',
    '.dav': 'DAV (Dahua DVR/NVR)',
}

# Codecs para saída de vídeo por extensão
OUTPUT_CODECS = {
    '.mp4': 'mp4v',
    '.avi': 'XVID',
    '.mov': 'mp4v',
    '.mkv': 'XVID',
    '.wmv': 'WMV2',
    '.webm': 'VP80',
    '.dav': 'mp4v',
}


@dataclass
class FrameResult:
    """Resultado de processamento de um frame individual."""
    frame_number: int
    timestamp_ms: float
    plates_found: int = 0
    plate_texts: List[str] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    bboxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    lpr_results: List[Any] = field(default_factory=list)
    processing_time_ms: float = 0.0
    annotated_frame: Optional[np.ndarray] = None


@dataclass
class VideoResult:
    """Resultado consolidado do processamento de um vídeo."""
    video_path: str = ''
    total_frames: int = 0
    processed_frames: int = 0
    skipped_frames: int = 0
    fps: float = 0.0
    duration_seconds: float = 0.0
    resolution: Tuple[int, int] = (0, 0)
    codec: str = ''

    # Resultados agregados
    unique_plates: Dict[str, Dict] = field(default_factory=dict)
    frame_results: List[FrameResult] = field(default_factory=list)
    total_detections: int = 0
    total_processing_time_ms: float = 0.0

    # Arquivo de saída
    output_video_path: Optional[str] = None

    @property
    def avg_processing_time_per_frame(self) -> float:
        if self.processed_frames == 0:
            return 0.0
        return self.total_processing_time_ms / self.processed_frames

    @property
    def processing_fps(self) -> float:
        if self.total_processing_time_ms == 0:
            return 0.0
        return self.processed_frames / (self.total_processing_time_ms / 1000)


class VideoProcessor:
    """
    Processador de vídeo para ALPR.
    
    Extrai frames, processa pelo pipeline LPR e gera:
    - Lista de placas únicas com melhor confiança
    - Vídeo anotado com bounding boxes e textos das placas
    - Métricas de processamento por frame
    """

    def __init__(
        self,
        skip_frames: int = 2,
        max_frames: int = 0,
        generate_output_video: bool = True,
        output_dir: str = 'data/results',
        confidence_threshold: float = 0.3,
        plate_tracking_iou: float = 0.5,
        annotation_color: Tuple[int, int, int] = (0, 255, 0),
        annotation_thickness: int = 2,
        font_scale: float = 0.8,
        vehicle_mode: VehicleMode = VehicleMode.MOVING,
        enable_temporal_voting: bool = True,
        temporal_voting_strategy: str = 'hybrid',
        temporal_min_observations: int = 2,
    ):
        """
        Inicializa o processador de vídeo.
        
        Args:
            skip_frames: Processar 1 a cada N frames (1 = todos, 2 = metade, etc.)
                         Se vehicle_mode estiver definido, este valor é sobrescrito
                         pelos parâmetros do modo, a menos que explicitamente fornecido.
            max_frames: Máximo de frames a processar (0 = sem limite)
            generate_output_video: Se deve gerar vídeo de saída anotado
            output_dir: Diretório para salvar vídeo de saída
            confidence_threshold: Confiança mínima para considerar uma detecção
            plate_tracking_iou: IoU mínimo para considerar mesma placa entre frames
            annotation_color: Cor BGR das anotações
            annotation_thickness: Espessura das linhas de anotação
            font_scale: Escala da fonte para textos
            vehicle_mode: Modo de análise (STATIONARY ou MOVING)
        """
        self.vehicle_mode = vehicle_mode
        mode_params = VEHICLE_MODE_PARAMS[vehicle_mode]

        # Usar skip_frames do modo se o caller não forneceu um valor explícito
        self.skip_frames = max(1, skip_frames)
        self.max_frames = max_frames
        self.generate_output_video = generate_output_video
        self.output_dir = Path(output_dir)
        self.confidence_threshold = confidence_threshold
        self.plate_tracking_iou = plate_tracking_iou
        self.annotation_color = annotation_color
        self.annotation_thickness = annotation_thickness
        self.font_scale = font_scale

        # Parâmetros específicos do modo
        self.confidence_early_stop = mode_params['confidence_early_stop']
        self.min_stable_detections = mode_params['min_stable_detections']
        self.sharpness_filter = mode_params['sharpness_filter']
        self.sharpness_threshold = mode_params['sharpness_threshold']

        # Criar diretório de saída
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Inicializar votação temporal
        self.enable_temporal_voting = enable_temporal_voting
        self._temporal_engine = None
        if enable_temporal_voting and TemporalVotingEngine is not None:
            self._temporal_engine = TemporalVotingEngine(
                enabled=True,
                iou_threshold=plate_tracking_iou,
                min_observations=temporal_min_observations,
                strategy=temporal_voting_strategy,
            )
            logger.info(f"Votação temporal habilitada (strategy={temporal_voting_strategy})")

        mode_label = 'parado' if vehicle_mode == VehicleMode.STATIONARY else 'em movimento'
        logger.info(
            f"VideoProcessor inicializado: modo={mode_label}, skip={self.skip_frames}, "
            f"max_frames={max_frames}, output_video={generate_output_video}"
        )

    @staticmethod
    def get_supported_extensions() -> List[str]:
        """Retorna lista de extensões suportadas."""
        return list(SUPPORTED_VIDEO_EXTENSIONS.keys())

    @staticmethod
    def get_extensions_for_uploader() -> List[str]:
        """Retorna extensões sem ponto para uso no file_uploader."""
        return [ext.lstrip('.') for ext in SUPPORTED_VIDEO_EXTENSIONS]

    @staticmethod
    def is_supported(filepath: str) -> bool:
        """Verifica se o arquivo é um vídeo suportado."""
        ext = Path(filepath).suffix.lower()
        return ext in SUPPORTED_VIDEO_EXTENSIONS

    def get_video_info(self, video_path: str) -> Dict:
        """
        Obtém informações do vídeo sem processá-lo.
        
        Args:
            video_path: Caminho para o arquivo de vídeo
            
        Returns:
            Dicionário com metadados do vídeo
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((codec_int >> 8 * i) & 0xFF) for i in range(4)])
            duration = total_frames / fps if fps > 0 else 0

            # Frames que serão processados
            frames_to_process = total_frames // self.skip_frames
            if self.max_frames > 0:
                frames_to_process = min(frames_to_process, self.max_frames)

            return {
                'total_frames': total_frames,
                'fps': fps,
                'width': width,
                'height': height,
                'resolution': f"{width}x{height}",
                'codec': codec.strip(),
                'duration_seconds': duration,
                'duration_formatted': self._format_duration(duration),
                'file_size_mb': Path(video_path).stat().st_size / (1024 * 1024),
                'frames_to_process': frames_to_process,
                'extension': Path(video_path).suffix.lower(),
                'format_name': SUPPORTED_VIDEO_EXTENSIONS.get(
                    Path(video_path).suffix.lower(), 'Desconhecido'
                ),
            }
        finally:
            cap.release()

    def process_video(
        self,
        video_path: str,
        pipeline,
        detector_confidence: float = 0.5,
        progress_callback: Optional[Callable[[int, int, Optional[FrameResult]], None]] = None,
        stop_event: Optional[Callable[[], bool]] = None,
    ) -> VideoResult:
        """
        Processa um vídeo completo pelo pipeline LPR.
        
        Args:
            video_path: Caminho para o arquivo de vídeo
            pipeline: Instância do LPRPipeline
            detector_confidence: Threshold de confiança para detecção
            progress_callback: Callback(frame_atual, total_frames, frame_result) para progresso
            stop_event: Callable que retorna True se o processamento deve ser interrompido
            
        Returns:
            VideoResult com todos os resultados consolidados
        """
        result = VideoResult(video_path=video_path)

        # Reset temporal voting engine para novo vídeo
        if self._temporal_engine:
            self._temporal_engine.reset()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")

        try:
            # Metadados
            result.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            result.fps = cap.get(cv2.CAP_PROP_FPS)
            result.resolution = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
            codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            result.codec = "".join([chr((codec_int >> 8 * i) & 0xFF) for i in range(4)]).strip()
            result.duration_seconds = result.total_frames / result.fps if result.fps > 0 else 0

            # Preparar writer de saída
            video_writer = None
            if self.generate_output_video:
                output_path = self._get_output_path(video_path)
                ext = Path(output_path).suffix.lower()
                fourcc_str = OUTPUT_CODECS.get(ext, 'mp4v')
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                video_writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    result.fps,
                    result.resolution
                )
                if video_writer.isOpened():
                    result.output_video_path = str(output_path)
                    logger.info(f"Vídeo de saída: {output_path}")
                else:
                    logger.warning("Falha ao criar vídeo de saída, continuando sem anotações")
                    video_writer = None

            # Processar frames
            frame_count = 0
            processed_count = 0
            global_start = time.time()

            # Early-stop tracking (modo parado)
            stable_plate_text = None
            stable_count = 0
            early_stopped = False

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

                # Verificar se deve parar
                if stop_event and stop_event():
                    logger.info(f"Processamento interrompido pelo usuário no frame {frame_count}")
                    break

                # Verificar limite de frames
                if self.max_frames > 0 and processed_count >= self.max_frames:
                    logger.info(f"Limite de {self.max_frames} frames atingido")
                    break

                # Early-stop: modo parado — se já temos detecções estáveis com alta confiança
                if early_stopped:
                    # Apenas escrever frames restantes no vídeo com as últimas anotações
                    if video_writer:
                        annotated = self._annotate_frame_from_last(frame, result.frame_results)
                        video_writer.write(annotated)
                    if progress_callback:
                        progress_callback(frame_count, result.total_frames, None)
                    continue

                # Verificar skip
                should_process = (frame_count - 1) % self.skip_frames == 0

                if should_process:
                    # Filtro de nitidez (modo parado): pular frames borrados
                    if self.sharpness_filter:
                        sharpness = self._compute_sharpness(frame)
                        if sharpness < self.sharpness_threshold:
                            result.skipped_frames += 1
                            if video_writer:
                                annotated = self._annotate_frame_from_last(
                                    frame, result.frame_results
                                )
                                video_writer.write(annotated)
                            if progress_callback:
                                progress_callback(frame_count, result.total_frames, None)
                            continue

                    # Processar frame
                    frame_result = self._process_frame(
                        frame, frame_count, timestamp_ms,
                        pipeline,
                        detector_confidence,
                        temporal_prior=self._build_temporal_prior(result.unique_plates),
                    )

                    processed_count += 1
                    result.processed_frames = processed_count
                    result.total_detections += frame_result.plates_found
                    result.total_processing_time_ms += frame_result.processing_time_ms

                    # Agregar placas únicas
                    self._aggregate_plates(result, frame_result)

                    # Guardar resultado do frame (sem a imagem anotada para economizar memória)
                    compact_result = FrameResult(
                        frame_number=frame_result.frame_number,
                        timestamp_ms=frame_result.timestamp_ms,
                        plates_found=frame_result.plates_found,
                        plate_texts=frame_result.plate_texts,
                        confidences=frame_result.confidences,
                        bboxes=frame_result.bboxes,
                        processing_time_ms=frame_result.processing_time_ms,
                    )
                    result.frame_results.append(compact_result)

                    # Escrever frame anotado no vídeo
                    if video_writer and frame_result.annotated_frame is not None:
                        video_writer.write(frame_result.annotated_frame)
                    elif video_writer:
                        video_writer.write(frame)

                    # Callback de progresso
                    if progress_callback:
                        progress_callback(frame_count, result.total_frames, frame_result)

                    # Early-stop check (modo parado)
                    if (self.min_stable_detections > 0
                            and self.confidence_early_stop > 0
                            and frame_result.plates_found > 0):
                        best_text = frame_result.plate_texts[0] if frame_result.plate_texts else None
                        best_conf = max(frame_result.confidences) if frame_result.confidences else 0.0

                        if best_text and best_conf >= self.confidence_early_stop:
                            if best_text == stable_plate_text:
                                stable_count += 1
                            else:
                                stable_plate_text = best_text
                                stable_count = 1

                            if stable_count >= self.min_stable_detections:
                                logger.info(
                                    f"Early-stop (parado): placa '{stable_plate_text}' detectada "
                                    f"{stable_count}x consecutivas com conf >= "
                                    f"{self.confidence_early_stop:.0%} no frame {frame_count}"
                                )
                                early_stopped = True
                        else:
                            stable_count = 0
                            stable_plate_text = None
                else:
                    result.skipped_frames += 1

                    # Escrever frame original no vídeo (manter FPS)
                    if video_writer:
                        # Re-anotar com últimas detecções conhecidas
                        annotated = self._annotate_frame_from_last(
                            frame, result.frame_results
                        )
                        video_writer.write(annotated)

                    # Callback de progresso (sem resultado)
                    if progress_callback:
                        progress_callback(frame_count, result.total_frames, None)

            total_time = (time.time() - global_start) * 1000
            result.total_processing_time_ms = total_time

            # Aplicar votação temporal para consolidar leituras
            if self._temporal_engine and self._temporal_engine.enabled:
                temporal_results = self._temporal_engine.get_consolidated_results()
                for tr in temporal_results:
                    if tr.get('voting_applied') and tr['text']:
                        voted_text = tr['text']
                        voted_norm = self._normalize_plate(voted_text)

                        # Atualizar ou adicionar placa votada
                        if voted_norm in result.unique_plates:
                            entry = result.unique_plates[voted_norm]
                            # Se votação gerou resultado melhor, atualizar
                            if tr['confidence'] > entry['best_confidence']:
                                entry['plate_text'] = voted_text
                                entry['best_confidence'] = tr['confidence']
                                entry['temporal_voted'] = True
                                entry['temporal_observations'] = tr['observations']
                                entry['temporal_span_frames'] = max(
                                    entry.get('temporal_span_frames', 1),
                                    tr.get('last_frame', 0) - tr.get('first_frame', 0) + 1,
                                )
                                logger.info(
                                    f"Temporal voting atualizou '{voted_norm}': "
                                    f"conf {entry['best_confidence']:.2f} → {tr['confidence']:.2f} "
                                    f"({tr['observations']} observações)"
                                )
                        elif voted_norm:
                            # Placa nova gerada por votação posicional
                            result.unique_plates[voted_norm] = {
                                'plate_text': voted_text,
                                'best_confidence': tr['confidence'],
                                'total_detections': tr['observations'],
                                'first_seen_frame': tr.get('first_frame', 0),
                                'last_seen_frame': tr.get('last_frame', 0),
                                'first_seen_time': 0,
                                'last_seen_time': 0,
                                'all_confidences': [tr['confidence']],
                                'best_char_confidences': [],
                                'quality_scores': [tr['confidence']],
                                'char_confirmation_scores': [tr['confidence']],
                                'char_confirmation_ratio': tr['confidence'],
                                'best_quality_score': tr['confidence'],
                                'best_frame_number': tr.get('first_frame', 0),
                                'best_timestamp_ms': 0,
                                'best_bbox': (0, 0, 0, 0),
                                'stable_bbox': (0, 0, 0, 0),
                                'temporal_voted': True,
                                'temporal_observations': tr['observations'],
                                'temporal_span_frames': tr.get('last_frame', 0) - tr.get('first_frame', 0) + 1,
                                'scenario_counts': {},
                                'artifact_samples': [],
                            }
                            logger.info(
                                f"Temporal voting gerou nova placa: '{voted_text}' "
                                f"(conf={tr['confidence']:.2f}, obs={tr['observations']})"
                            )

            mode_label = 'parado' if self.vehicle_mode == VehicleMode.STATIONARY else 'em movimento'
            early_msg = " (early-stop)" if early_stopped else ""
            logger.info(
                f"Vídeo processado [{mode_label}]{early_msg}: "
                f"{processed_count}/{result.total_frames} frames, "
                f"{len(result.unique_plates)} placas únicas, "
                f"{total_time:.0f}ms total"
            )

        finally:
            cap.release()
            if video_writer:
                video_writer.release()

        return result

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp_ms: float,
        pipeline,
        detector_confidence: float,
        temporal_prior: Optional[Dict[str, float]] = None,
    ) -> FrameResult:
        """Processa um frame individual pelo pipeline LPR."""
        frame_result = FrameResult(
            frame_number=frame_number,
            timestamp_ms=timestamp_ms,
        )

        start_time = time.time()

        try:
            try:
                lpr_results = pipeline.process_image(
                    frame,
                    detector_confidence=detector_confidence,
                    temporal_prior=temporal_prior,
                )
            except TypeError:
                lpr_results = pipeline.process_image(
                    frame,
                    detector_confidence=detector_confidence,
                )

            annotated_frame = frame.copy()

            for lpr in lpr_results:
                if lpr.confidence < self.confidence_threshold:
                    continue

                frame_result.plates_found += 1
                frame_result.plate_texts.append(lpr.plate_text)
                frame_result.confidences.append(lpr.confidence)
                frame_result.lpr_results.append(lpr)

                # Extrair bbox se disponível
                bbox = getattr(lpr, 'bbox', None)
                if bbox is not None:
                    frame_result.bboxes.append(tuple(bbox))
                    # Anotar frame
                    annotated_frame = self._annotate_frame(
                        annotated_frame, bbox, lpr.plate_text, lpr.confidence
                    )

            frame_result.annotated_frame = annotated_frame

        except Exception as e:
            logger.error(f"Erro no frame {frame_number}: {e}")
            frame_result.annotated_frame = frame.copy()

        frame_result.processing_time_ms = (time.time() - start_time) * 1000
        return frame_result

    def _annotate_frame(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        plate_text: str,
        confidence: float,
    ) -> np.ndarray:
        """Anota um frame com bounding box e texto da placa."""
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Cor baseada na confiança
        if confidence >= 0.8:
            color = (0, 255, 0)  # Verde - alta confiança
        elif confidence >= 0.6:
            color = (0, 255, 255)  # Amarelo - média
        else:
            color = (0, 165, 255)  # Laranja - baixa

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.annotation_thickness)

        # Fundo para o texto
        label = f"{plate_text} ({confidence:.0%})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, self.font_scale, self.annotation_thickness
        )

        # Background do label
        label_y = max(y1 - 10, text_h + 10)
        cv2.rectangle(
            frame,
            (x1, label_y - text_h - 8),
            (x1 + text_w + 8, label_y + 4),
            color,
            cv2.FILLED
        )

        # Texto
        cv2.putText(
            frame, label,
            (x1 + 4, label_y - 2),
            font, self.font_scale,
            (0, 0, 0),  # Preto sobre fundo colorido
            self.annotation_thickness,
            cv2.LINE_AA
        )

        return frame

    def _annotate_frame_from_last(
        self,
        frame: np.ndarray,
        previous_results: List[FrameResult],
    ) -> np.ndarray:
        """Anota frame usando últimas detecções conhecidas (para frames pulados)."""
        if not previous_results:
            return frame

        last = previous_results[-1]
        annotated = frame.copy()

        for i, bbox in enumerate(last.bboxes):
            text = last.plate_texts[i] if i < len(last.plate_texts) else "?"
            conf = last.confidences[i] if i < len(last.confidences) else 0.0
            annotated = self._annotate_frame(annotated, bbox, text, conf)

        return annotated

    @staticmethod
    def _compute_sharpness(frame: np.ndarray) -> float:
        """
        Calcula a nitidez de um frame usando variância do Laplaciano.
        Valores mais altos = imagem mais nítida.
        
        Args:
            frame: Imagem BGR
            
        Returns:
            Score de nitidez (variância do Laplaciano)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    @staticmethod
    def _compute_bbox_iou(
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int],
    ) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
        area2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _smooth_bbox(
        previous_bbox: Tuple[int, int, int, int],
        current_bbox: Tuple[int, int, int, int],
        alpha: float = 0.65,
    ) -> Tuple[int, int, int, int]:
        return tuple(
            int(round(alpha * previous_bbox[index] + (1.0 - alpha) * current_bbox[index]))
            for index in range(4)
        )

    @staticmethod
    def _average_char_confidence(char_confidences: List[Tuple[str, float]]) -> float:
        if not char_confidences:
            return 0.0
        return float(np.mean([float(confidence) for _char, confidence in char_confidences]))

    def _build_temporal_prior(self, unique_plates: Dict[str, Dict]) -> Dict[str, float]:
        if not unique_plates:
            return {}

        max_detections = max(
            (int(info.get('total_detections', 0)) for info in unique_plates.values()),
            default=1,
        )
        max_span = max(
            (int(info.get('temporal_span_frames', 0)) for info in unique_plates.values()),
            default=1,
        )

        temporal_prior: Dict[str, float] = {}
        for normalized, info in unique_plates.items():
            detection_score = info.get('total_detections', 0) / max_detections if max_detections > 0 else 0.0
            span_score = info.get('temporal_span_frames', 0) / max_span if max_span > 0 else 0.0
            char_score = float(
                np.mean(info.get('char_confirmation_scores', []) or [info.get('char_confirmation_ratio', 0.0)])
            )
            voted_bonus = 1.0 if info.get('temporal_voted') else min(
                1.0,
                info.get('temporal_observations', 0) / 5,
            )
            temporal_prior[normalized] = min(
                1.0,
                0.35 * detection_score
                + 0.20 * span_score
                + 0.20 * float(info.get('best_confidence', 0.0))
                + 0.15 * float(info.get('best_quality_score', 0.0))
                + 0.10 * max(char_score, voted_bonus),
            )

        return temporal_prior

    def _aggregate_plates(self, video_result: VideoResult, frame_result: FrameResult):
        """
        Agrega placas detectadas mantendo a melhor confiança de cada placa única.
        Normaliza o texto da placa antes de comparar.
        Rastreia confiança por caractere para leitura confirmada.
        """
        for i, plate_text in enumerate(frame_result.plate_texts):
            # Normalizar texto para comparação
            normalized = self._normalize_plate(plate_text)

            if not normalized:
                continue

            conf = frame_result.confidences[i] if i < len(frame_result.confidences) else 0.0
            bbox = frame_result.bboxes[i] if i < len(frame_result.bboxes) else None

            # Obter char_confidences do LPRResult se disponível
            char_confs = []
            lpr = None
            if i < len(frame_result.lpr_results):
                lpr = frame_result.lpr_results[i]
                char_confs = getattr(lpr, 'char_confidences', []) or []

            # Se não temos confiança por caractere, usar confiança global para cada char
            if not char_confs and normalized:
                char_confs = [(c, conf) for c in normalized]

            avg_char_conf = self._average_char_confidence(char_confs)
            quality_score = max(
                conf,
                float(getattr(lpr, 'quality_score', 0.0) or 0.0),
                0.50 * conf + 0.50 * avg_char_conf,
            )
            scenario_tags = list(getattr(lpr, 'scenario_tags', []) or []) if lpr is not None else []
            artifact_dir = str(getattr(lpr, 'artifact_dir', '') or '') if lpr is not None else ''

            # Adicionar observação ao motor de votação temporal
            if self._temporal_engine:
                self._temporal_engine.add_observation(
                    frame_number=frame_result.frame_number,
                    plate_text=normalized,
                    confidence=conf,
                    bbox=bbox or (0, 0, 0, 0),
                    char_confidences=char_confs,
                )

            if normalized not in video_result.unique_plates:
                # Inicializar best_char_confidences com as confiancas deste frame
                best_char_confs = [(c, cf) for c, cf in char_confs] if char_confs else []
                scenario_counts = {tag: 1 for tag in scenario_tags}

                video_result.unique_plates[normalized] = {
                    'plate_text': plate_text,
                    'best_confidence': conf,
                    'total_detections': 1,
                    'first_seen_frame': frame_result.frame_number,
                    'last_seen_frame': frame_result.frame_number,
                    'first_seen_time': frame_result.timestamp_ms,
                    'last_seen_time': frame_result.timestamp_ms,
                    'all_confidences': [conf],
                    'best_char_confidences': best_char_confs,
                    'quality_scores': [quality_score],
                    'char_confirmation_scores': [avg_char_conf],
                    'char_confirmation_ratio': avg_char_conf,
                    'best_quality_score': quality_score,
                    'best_frame_number': frame_result.frame_number,
                    'best_timestamp_ms': frame_result.timestamp_ms,
                    'best_bbox': bbox or (0, 0, 0, 0),
                    'stable_bbox': bbox or (0, 0, 0, 0),
                    'temporal_voted': False,
                    'temporal_observations': 1 if self._temporal_engine else 0,
                    'temporal_span_frames': 1,
                    'scenario_counts': scenario_counts,
                    'artifact_samples': [artifact_dir] if artifact_dir else [],
                }
            else:
                entry = video_result.unique_plates[normalized]
                entry['total_detections'] += 1
                entry['last_seen_frame'] = frame_result.frame_number
                entry['last_seen_time'] = frame_result.timestamp_ms
                entry['all_confidences'].append(conf)
                entry.setdefault('quality_scores', []).append(quality_score)
                entry.setdefault('char_confirmation_scores', []).append(avg_char_conf)
                entry['char_confirmation_ratio'] = float(np.mean(entry['char_confirmation_scores']))
                entry['temporal_observations'] = max(
                    entry.get('temporal_observations', 0),
                    entry['total_detections'],
                )
                entry['temporal_span_frames'] = max(
                    1,
                    frame_result.frame_number - entry.get('first_seen_frame', frame_result.frame_number) + 1,
                )

                if bbox is not None:
                    previous_bbox = entry.get('stable_bbox') or bbox
                    if previous_bbox and self._compute_bbox_iou(previous_bbox, bbox) >= self.plate_tracking_iou * 0.5:
                        smoothed_bbox = self._smooth_bbox(previous_bbox, bbox)
                    else:
                        smoothed_bbox = tuple(bbox)
                    entry['stable_bbox'] = smoothed_bbox
                    if i < len(frame_result.bboxes):
                        frame_result.bboxes[i] = smoothed_bbox

                for tag in scenario_tags:
                    entry.setdefault('scenario_counts', {})[tag] = entry.setdefault('scenario_counts', {}).get(tag, 0) + 1

                if artifact_dir:
                    samples = entry.setdefault('artifact_samples', [])
                    if artifact_dir not in samples and len(samples) < 3:
                        samples.append(artifact_dir)

                # Atualizar confiança por caractere (manter a melhor para cada posição)
                if char_confs:
                    existing = entry.get('best_char_confidences', [])
                    merged = []
                    for pos in range(max(len(existing), len(char_confs))):
                        old_c, old_cf = existing[pos] if pos < len(existing) else ('', 0.0)
                        new_c, new_cf = char_confs[pos] if pos < len(char_confs) else ('', 0.0)
                        if new_cf > old_cf:
                            merged.append((new_c, new_cf))
                        else:
                            merged.append((old_c, old_cf))
                    entry['best_char_confidences'] = merged

                # Atualizar se melhor confiança
                if conf > entry['best_confidence']:
                    entry['best_confidence'] = conf
                    entry['plate_text'] = plate_text  # Usar texto da melhor detecção

                if quality_score >= entry.get('best_quality_score', 0.0):
                    entry['best_quality_score'] = quality_score
                    entry['best_frame_number'] = frame_result.frame_number
                    entry['best_timestamp_ms'] = frame_result.timestamp_ms
                    if bbox is not None:
                        entry['best_bbox'] = entry.get('stable_bbox', bbox)

            if normalized in video_result.unique_plates and bbox is not None and i < len(frame_result.bboxes):
                stable_bbox = video_result.unique_plates[normalized].get('stable_bbox')
                if stable_bbox:
                    frame_result.bboxes[i] = stable_bbox

    @staticmethod
    def _normalize_plate(text: str) -> str:
        """Normaliza texto de placa para comparação."""
        return ''.join(c for c in text.upper() if c.isalnum())

    def _get_output_path(self, input_path: str) -> Path:
        """Gera caminho para o vídeo de saída."""
        input_name = Path(input_path).stem
        ext = Path(input_path).suffix.lower()

        # Usar mp4 como fallback se extensão não suportada para output
        if ext not in OUTPUT_CODECS:
            ext = '.mp4'

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_name = f"{input_name}_alpr_{timestamp}{ext}"
        return self.output_dir / output_name

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Formata duração em formato legível."""
        if seconds < 0:
            return "00:00"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    @staticmethod
    def build_confirmed_reading(
        plate_info: Dict,
        threshold: float = CHAR_CONFIDENCE_THRESHOLD,
    ) -> str:
        """
        Gera leitura confirmada da placa: caracteres com confiança >= threshold
        aparecem normalmente, os demais são substituídos por *.
        
        Exemplo: 'MBC*D2*' (posições 4 e 7 abaixo do limiar)
        
        Args:
            plate_info: Dicionário de informações da placa (de unique_plates)
            threshold: Limiar mínimo de confiança por caractere
            
        Returns:
            String com leitura confirmada (ex: 'MBC*D2*')
        """
        char_confs = plate_info.get('best_char_confidences', [])
        plate_text = plate_info.get('plate_text', '')

        if not char_confs:
            # Sem dados por caractere — usar texto completo se confiança geral >= threshold
            if plate_info.get('best_confidence', 0) >= threshold:
                return plate_text
            return '*' * len(plate_text)

        confirmed = []
        for char, conf in char_confs:
            if conf >= threshold:
                confirmed.append(char)
            else:
                confirmed.append('*')

        return ''.join(confirmed)

    @staticmethod
    def rank_unique_plates(
        unique_plates: Dict[str, Dict],
        max_plates: int = MAX_UNIQUE_PLATES,
    ) -> Dict[str, Dict]:
        """
        Classifica e limita as placas únicas, retornando as mais prováveis.
        
        O score composto considera:
        - Número de detecções (mais detecções = mais confiável)
        - Melhor confiança individual
        - Confiança média
        
        Pesos: detecções (40%), melhor confiança (35%), confiança média (25%)
        
        Args:
            unique_plates: Dicionário de placas únicas do VideoResult
            max_plates: Máximo de placas a retornar (padrão: MAX_UNIQUE_PLATES)
            
        Returns:
            Dicionário ordenado com as top N placas mais prováveis
        """
        if not unique_plates:
            return {}

        if len(unique_plates) <= max_plates:
            # Mesmo se não precisar cortar, ordena por score
            pass

        # Calcular score composto para cada placa
        scored = []
        max_detections = max(
            (info.get('total_detections', 0) for info in unique_plates.values()),
            default=1
        )
        max_span = max(
            (info.get('temporal_span_frames', 0) for info in unique_plates.values()),
            default=1
        )

        for normalized, info in unique_plates.items():
            avg_conf = np.mean(info.get('all_confidences', [])) if info.get('all_confidences') else 0.0
            avg_quality = np.mean(info.get('quality_scores', [])) if info.get('quality_scores') else info.get('best_confidence', 0.0)
            char_confirmation = np.mean(info.get('char_confirmation_scores', [])) if info.get('char_confirmation_scores') else info.get('char_confirmation_ratio', 0.0)
            span_score = info.get('temporal_span_frames', 0) / max_span if max_span > 0 else 0.0
            temporal_bonus = 1.0 if info.get('temporal_voted') else min(
                1.0,
                info.get('temporal_observations', 0) / 5,
            )

            # Normalizar detecções para 0-1
            detection_score = info.get('total_detections', 0) / max_detections if max_detections > 0 else 0.0

            # Score composto
            composite_score = (
                0.22 * detection_score +
                0.20 * info.get('best_confidence', 0.0) +
                0.13 * avg_conf +
                0.15 * avg_quality +
                0.15 * char_confirmation +
                0.10 * span_score +
                0.05 * temporal_bonus
            )

            info['composite_score'] = composite_score
            info['avg_confidence'] = float(avg_conf)
            info['avg_quality_score'] = float(avg_quality)
            info['char_confirmation_ratio'] = float(char_confirmation)
            scored.append((normalized, info, composite_score))

        # Ordenar por score decrescente
        scored.sort(key=lambda x: x[2], reverse=True)

        # Limitar a max_plates
        top_plates = {}
        for normalized, info, _ in scored[:max_plates]:
            top_plates[normalized] = info

        logger.info(
            f"Ranking de placas: {len(unique_plates)} detectadas -> "
            f"top {len(top_plates)} retornadas"
        )

        return top_plates

    def extract_best_frames(
        self,
        video_result: VideoResult,
        video_path: str,
        top_n: int = 5,
    ) -> List[Tuple[str, np.ndarray, FrameResult]]:
        """
        Extrai os melhores frames (maior confiança) do vídeo.
        
        Args:
            video_result: Resultado do processamento
            video_path: Caminho do vídeo original
            top_n: Número de frames a extrair
            
        Returns:
            Lista de (plate_text, frame_image, frame_result)
        """
        # Encontrar frames com melhores detecções
        ranked = self.rank_unique_plates(video_result.unique_plates, max_plates=top_n)
        requested_frames = [
            (info.get('best_frame_number'), info.get('plate_text', '?'))
            for info in ranked.values()
            if info.get('best_frame_number')
        ]

        if requested_frames:
            best_frames = []
            frame_map = {fr.frame_number: fr for fr in video_result.frame_results}
            for frame_number, plate_text in requested_frames:
                frame_result = frame_map.get(frame_number)
                if frame_result is not None:
                    best_frames.append((frame_result, plate_text))
        else:
            best_frames = [
                (fr, fr.plate_texts[0] if fr.plate_texts else '?')
                for fr in sorted(
                    [fr for fr in video_result.frame_results if fr.plates_found > 0],
                    key=lambda x: max(x.confidences) if x.confidences else 0,
                    reverse=True
                )[:top_n]
            ]

        if not best_frames:
            return []

        results = []
        cap = cv2.VideoCapture(video_path)

        try:
            for fr, plate_label in best_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fr.frame_number - 1)
                ret, frame = cap.read()
                if ret:
                    best_plate = plate_label or (fr.plate_texts[0] if fr.plate_texts else "?")
                    results.append((best_plate, frame, fr))
        finally:
            cap.release()

        return results

    def generate_timeline(self, video_result: VideoResult) -> List[Dict]:
        """
        Gera timeline de detecções para visualização.
        
        Returns:
            Lista de eventos com timestamp, frame, placas detectadas
        """
        timeline = []

        for fr in video_result.frame_results:
            if fr.plates_found > 0:
                timeline.append({
                    'frame': fr.frame_number,
                    'time_s': fr.timestamp_ms / 1000,
                    'time_formatted': self._format_duration(fr.timestamp_ms / 1000),
                    'plates': fr.plate_texts,
                    'max_confidence': max(fr.confidences) if fr.confidences else 0,
                    'processing_ms': fr.processing_time_ms,
                    'tracking_boxes': len(fr.bboxes),
                })

        return timeline
