# src/temporal_voting.py
"""
Votação temporal entre frames de vídeo para ALPR.

Quando processando vídeo, a mesma placa aparece em múltiplos frames.
Cada leitura pode ter erros ligeiramente diferentes. Este módulo agrega
leituras de múltiplos frames para produzir a leitura mais provável.

Estratégias:
1. Votação por caractere posicional: para cada posição, escolhe o caractere
   mais frequente entre todos os frames
2. Votação por placa completa: conta a placa mais frequente entre frames
3. Votação híbrida: combina placa completa + votação posicional

Funciona com tracking IoU para associar a mesma placa entre frames.
"""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.constants import SIMILAR_CHARS

logger = logging.getLogger(__name__)


@dataclass
class TemporalPlateTrack:
    """Rastreamento de uma placa ao longo de múltiplos frames."""
    track_id: int

    # Observações por frame
    observations: List[Dict[str, Any]] = field(default_factory=list)
    # Formato: {text, confidence, char_confidences, frame_number, bbox}

    # Resultado consolidado
    best_text: str = ''
    best_confidence: float = 0.0
    voted_text: str = ''
    voted_confidence: float = 0.0

    # Metadados
    first_frame: int = 0
    last_frame: int = 0
    total_observations: int = 0

    @property
    def has_enough_observations(self) -> bool:
        """Precisa de pelo menos 2 observações para votação."""
        return len(self.observations) >= 2

    @property
    def duration_frames(self) -> int:
        return self.last_frame - self.first_frame + 1 if self.observations else 0


class TemporalVotingEngine:
    """
    Motor de votação temporal para consolidar leituras de placa entre frames.
    
    Mantém tracks de placas (associadas por IoU) e aplica votação
    por caractere posicional para produzir a melhor leitura.
    
    Uso:
        engine = TemporalVotingEngine()
        
        # A cada frame processado:
        engine.add_observation(frame_num, plate_text, confidence, bbox, char_confs)
        
        # Ao final ou periodicamente:
        results = engine.get_consolidated_results()
    """

    def __init__(
        self,
        enabled: bool = True,
        iou_threshold: float = 0.4,
        min_observations: int = 2,
        max_tracks: int = 20,
        decay_factor: float = 0.95,
        strategy: str = 'hybrid',
    ):
        """
        Args:
            enabled: Se votação temporal está ativa
            iou_threshold: IoU mínimo para associar detecção à mesma track
            min_observations: Mínimo de observações para gerar resultado votado
            max_tracks: Máximo de tracks ativos
            decay_factor: Decaimento de confiança por distância temporal
            strategy: 'positional', 'majority', 'hybrid'
        """
        self.enabled = enabled
        self.iou_threshold = iou_threshold
        self.min_observations = min_observations
        self.max_tracks = max_tracks
        self.decay_factor = decay_factor
        self.strategy = strategy

        self._tracks: List[TemporalPlateTrack] = []
        self._next_track_id = 0
        self._last_frame = 0

        logger.info(
            f"TemporalVotingEngine: enabled={enabled}, iou_thresh={iou_threshold}, "
            f"min_obs={min_observations}, strategy={strategy}"
        )

    def reset(self):
        """Limpa todas as tracks (início de novo vídeo)."""
        self._tracks.clear()
        self._next_track_id = 0
        self._last_frame = 0

    def add_observation(
        self,
        frame_number: int,
        plate_text: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],
        char_confidences: Optional[List[Tuple[str, float]]] = None,
    ):
        """
        Adiciona uma observação de placa de um frame.
        
        Associa automaticamente à track existente por IoU, ou cria nova track.
        
        Args:
            frame_number: Número do frame
            plate_text: Texto da placa lido neste frame
            confidence: Confiança geral da leitura
            bbox: Bounding box (x1, y1, x2, y2)
            char_confidences: Confiança por caractere [(char, conf), ...]
        """
        if not self.enabled:
            return

        clean_text = plate_text.replace('-', '').upper()
        if len(clean_text) < 5:
            return

        self._last_frame = frame_number

        observation = {
            'text': clean_text,
            'confidence': confidence,
            'char_confidences': char_confidences or [(c, confidence) for c in clean_text],
            'frame_number': frame_number,
            'bbox': bbox,
        }

        # Tentar associar a uma track existente
        best_track = self._find_matching_track(bbox, clean_text)

        if best_track is not None:
            best_track.observations.append(observation)
            best_track.last_frame = frame_number
            best_track.total_observations += 1

            # Atualizar best individual
            if confidence > best_track.best_confidence:
                best_track.best_text = clean_text
                best_track.best_confidence = confidence
        else:
            # Criar nova track
            if len(self._tracks) >= self.max_tracks:
                # Remover track mais antiga com menos observações
                self._tracks.sort(key=lambda t: (t.total_observations, t.last_frame))
                self._tracks.pop(0)

            track = TemporalPlateTrack(
                track_id=self._next_track_id,
                observations=[observation],
                best_text=clean_text,
                best_confidence=confidence,
                first_frame=frame_number,
                last_frame=frame_number,
                total_observations=1,
            )
            self._tracks.append(track)
            self._next_track_id += 1

    def _find_matching_track(
        self,
        bbox: Tuple[int, int, int, int],
        text: str,
    ) -> Optional[TemporalPlateTrack]:
        """
        Encontra a track que melhor corresponde a esta detecção.
        
        Critérios:
        1. IoU do bbox com a última observação da track
        2. Similaridade textual (distância de edição)
        """
        best_track = None
        best_score = 0.0

        for track in self._tracks:
            if not track.observations:
                continue

            last_obs = track.observations[-1]
            last_bbox = last_obs['bbox']

            # IoU
            iou = self._compute_iou(bbox, last_bbox)

            # Similaridade textual
            text_sim = self._text_similarity(text, last_obs['text'])

            # Score combinado
            score = 0.6 * iou + 0.4 * text_sim

            if score > best_score and (iou >= self.iou_threshold or text_sim >= 0.7):
                best_score = score
                best_track = track

        return best_track

    @staticmethod
    def _compute_iou(
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int],
    ) -> float:
        """Computa Intersection over Union entre dois bboxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _text_similarity(text1: str, text2: str) -> float:
        """Similaridade simples baseada em caracteres iguais por posição."""
        if not text1 or not text2:
            return 0.0
        min_len = min(len(text1), len(text2))
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 1.0
        matches = sum(1 for i in range(min_len) if text1[i] == text2[i])
        return matches / max_len

    def get_consolidated_results(self) -> List[Dict[str, Any]]:
        """
        Retorna resultados consolidados de todas as tracks.
        
        Para cada track com observações suficientes, aplica a estratégia
        de votação para produzir o melhor texto.
        
        Returns:
            Lista de dicts com:
              - text: texto votado
              - confidence: confiança votada
              - observations: número de observações
              - best_individual: melhor leitura individual
              - track_id: id da track
        """
        results = []

        for track in self._tracks:
            if track.total_observations < self.min_observations:
                # Poucos frames, usar melhor individual
                results.append({
                    'text': track.best_text,
                    'confidence': track.best_confidence,
                    'observations': track.total_observations,
                    'best_individual': track.best_text,
                    'track_id': track.track_id,
                    'voting_applied': False,
                })
                continue

            # Aplicar votação
            if self.strategy == 'positional':
                voted_text, voted_conf = self._vote_positional(track)
            elif self.strategy == 'majority':
                voted_text, voted_conf = self._vote_majority(track)
            else:  # hybrid
                voted_text, voted_conf = self._vote_hybrid(track)

            results.append({
                'text': voted_text,
                'confidence': voted_conf,
                'observations': track.total_observations,
                'best_individual': track.best_text,
                'best_individual_confidence': track.best_confidence,
                'track_id': track.track_id,
                'voting_applied': True,
                'first_frame': track.first_frame,
                'last_frame': track.last_frame,
            })

            # Atualizar track
            track.voted_text = voted_text
            track.voted_confidence = voted_conf

        # Ordenar por confiança
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results

    def _vote_positional(self, track: TemporalPlateTrack) -> Tuple[str, float]:
        """
        Votação posicional: para cada posição, escolhe o caractere mais frequente
        ponderado por confiança.
        
        Returns:
            (texto_votado, confiança_média)
        """
        # Normalizar todos os textos para 7 caracteres
        texts_7 = [obs['text'] for obs in track.observations if len(obs['text']) == 7]

        if not texts_7:
            return (track.best_text, track.best_confidence)

        voted_chars = []
        total_confidence = 0.0

        for pos in range(7):
            char_weights = defaultdict(float)

            for obs in track.observations:
                text = obs['text']
                if len(text) != 7:
                    continue

                char = text[pos]
                conf = obs['confidence']

                # Usar confiança por caractere se disponível
                char_confs = obs.get('char_confidences', [])
                if char_confs and pos < len(char_confs):
                    _, char_conf = char_confs[pos]
                    conf = char_conf

                # Agrupar similares para votação
                for sim_char in self._get_similarity_group(char):
                    char_weights[sim_char] += conf * 0.3  # Vizinhos com desconto
                char_weights[char] += conf  # O próprio com peso total

            if char_weights:
                best_char = max(char_weights, key=char_weights.get)
                best_weight = char_weights[best_char]
                voted_chars.append(best_char)
                total_confidence += min(1.0, best_weight / max(1, len(texts_7)))
            else:
                voted_chars.append('?')
                total_confidence += 0.0

        voted_text = ''.join(voted_chars)
        avg_confidence = total_confidence / 7

        return (voted_text, avg_confidence)

    def _vote_majority(self, track: TemporalPlateTrack) -> Tuple[str, float]:
        """
        Votação por maioria: a placa completa mais frequente vence.
        """
        text_counts = Counter()
        text_best_conf = {}

        for obs in track.observations:
            text = obs['text']
            if len(text) == 7:
                text_counts[text] += 1
                if text not in text_best_conf or obs['confidence'] > text_best_conf[text]:
                    text_best_conf[text] = obs['confidence']

        if not text_counts:
            return (track.best_text, track.best_confidence)

        most_common = text_counts.most_common(1)[0]
        text = most_common[0]
        count = most_common[1]

        # Confiança = (frequência relativa + melhor confiança) / 2
        freq_conf = count / sum(text_counts.values())
        ocr_conf = text_best_conf.get(text, 0.5)
        confidence = (freq_conf + ocr_conf) / 2

        return (text, confidence)

    def _vote_hybrid(self, track: TemporalPlateTrack) -> Tuple[str, float]:
        """
        Votação híbrida: combina votação por maioria e posicional.
        
        1. Se há uma placa com maioria clara (>50% dos frames), usa ela
        2. Senão, usa votação posicional
        3. Valida resultado: se inválido, usa melhor individual
        """
        import re

        # 1. Tentar maioria
        text_counts = Counter()
        for obs in track.observations:
            if len(obs['text']) == 7:
                text_counts[obs['text']] += 1

        total_7 = sum(text_counts.values())

        if text_counts and total_7 > 0:
            _most_common_text, most_common_count = text_counts.most_common(1)[0]

            # Se maioria clara (>50%), usar
            if most_common_count / total_7 > 0.5:
                majority_text, majority_conf = self._vote_majority(track)
                # Validar formato
                if (re.match(r'^[A-Z]{3}[0-9]{4}$', majority_text) or
                    re.match(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$', majority_text)):
                    return (majority_text, majority_conf)

        # 2. Votação posicional
        positional_text, positional_conf = self._vote_positional(track)

        # Validar formato
        if (re.match(r'^[A-Z]{3}[0-9]{4}$', positional_text) or
            re.match(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$', positional_text)):
            return (positional_text, positional_conf)

        # 3. Fallback: melhor individual
        return (track.best_text, track.best_confidence)

    @staticmethod
    def _get_similarity_group(char: str) -> List[str]:
        """Retorna grupo de caracteres similares."""
        return SIMILAR_CHARS.get(char, [])

    def get_track_for_plate(self, plate_text: str) -> Optional[TemporalPlateTrack]:
        """Busca track para uma placa específica."""
        clean = plate_text.replace('-', '').upper()
        for track in self._tracks:
            if track.best_text == clean or track.voted_text == clean:
                return track
        return None

    def get_status(self) -> Dict[str, Any]:
        """Retorna status para UI."""
        return {
            'enabled': self.enabled,
            'active_tracks': len(self._tracks),
            'strategy': self.strategy,
            'min_observations': self.min_observations,
            'total_observations': sum(t.total_observations for t in self._tracks),
        }
