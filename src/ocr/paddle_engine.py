# src/ocr/paddle_engine.py
"""
PaddleOCR Engine — OCR determinístico clássico como alternativa aos VLMs Ollama.

Diferencial em relação a GLM/OLMoOCR2:
- Totalmente offline e determinístico (mesma saída para a mesma entrada).
- Expõe **confiança por caractere** nativa (char_confidences reais, não reconstruídas).
- Muito mais leve que VLMs 7B; roda em CPU razoavelmente bem.
- Não precisa de Ollama nem de modelos pullados; só do pacote `paddleocr`.

Limitações:
- Qualidade em placas BR é inferior à de VLMs bem prompted — mas é um bom 3º
  engine e excelente para rodar offline sem GPU.

Instalação (opcional — não está em requirements.txt por padrão):
    pip install paddleocr paddlepaddle       # CPU (recomendado no Windows)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, List, Optional

import cv2
import numpy as np

from src.ocr.base import OCREngine
from src.ocr.confidence import format_aderence_confidence
from src.ocr.types import OCRResult, create_ocr_result

logger = logging.getLogger(__name__)


_RE_OLD = re.compile(r'^[A-Z]{3}[0-9]{4}$')
_RE_MERCOSUL = re.compile(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$')


def _paddle_available() -> bool:
    """Verifica se paddleocr + paddlepaddle estão instalados (sem importar)."""
    try:
        import importlib.util
        if importlib.util.find_spec('paddleocr') is None:
            return False
        return not (
            importlib.util.find_spec('paddle') is None
            and importlib.util.find_spec('paddlepaddle') is None
        )
    except Exception:
        return False


def _paddle_gpu_runtime_available() -> bool:
    """Retorna True se a runtime Paddle atual consegue usar GPU."""
    try:
        import paddle  # type: ignore
        if not paddle.device.is_compiled_with_cuda():
            return False
        return int(paddle.device.cuda.device_count()) > 0
    except Exception:
        return False


class PaddleOCREngine(OCREngine):
    """OCR via PaddleOCR (determinístico clássico)."""

    engine_name = 'paddle_ocr'

    def __init__(
        self,
        lang: str = 'pt',
        use_gpu: bool = True,
        use_angle_cls: bool = True,
        det_model: Optional[str] = None,
        rec_model: Optional[str] = None,
        det_limit_side_len: int = 960,
        rec_batch_num: int = 6,
        min_score: float = 0.3,
    ):
        """
        Args:
            lang: Idioma do PaddleOCR. 'pt' para placas BR (aceitável).
            use_gpu: Se True, usa GPU quando disponível.
            use_angle_cls: Habilita classificação de ângulo (rotações 0/90/180/270).
            det_model: Caminho customizado do modelo de detecção (None = padrão).
            rec_model: Caminho customizado do modelo de reconhecimento.
            det_limit_side_len: Lado máximo após redimensionamento interno.
            rec_batch_num: Tamanho do batch no reconhecimento.
            min_score: Confiança mínima nativa do PaddleOCR para aceitar um resultado.
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.use_angle_cls = use_angle_cls
        self.det_model = det_model
        self.rec_model = rec_model
        self.det_limit_side_len = det_limit_side_len
        self.rec_batch_num = rec_batch_num
        self.min_score = min_score
        self._ocr: Optional[Any] = None
        self.active_device: Optional[str] = None
        self.init_error: Optional[str] = None
        self.is_available = self._init_engine()

    def _init_engine(self) -> bool:
        """Lazy import de PaddleOCR; retorna False sem quebrar se não instalado."""
        if not _paddle_available():
            self.init_error = (
                'Pacotes paddleocr/paddle não encontrados no ambiente Python atual.'
            )
            logger.info(
                "PaddleOCR não está instalado. Para usar, rode: "
                "pip install paddleocr paddlepaddle"
            )
            return False

        # PaddleOCR 3.x no Windows pode importar `torch` indiretamente via
        # PaddleX/ModelScope; carregar torch antes evita falhas de DLL em alguns
        # ambientes mistos com PyTorch + Paddle.
        try:
            import torch  # type: ignore  # noqa: F401
        except Exception:
            pass

        os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')

        try:
            from paddleocr import PaddleOCR  # type: ignore

            devices = ['gpu:0', 'cpu'] if self.use_gpu and _paddle_gpu_runtime_available() else ['cpu']
            last_error: Optional[Exception] = None

            for device in devices:
                try:
                    kwargs: dict[str, Any] = {
                        'lang': self.lang,
                        'device': device,
                        'use_doc_orientation_classify': False,
                        'use_doc_unwarping': False,
                        'use_textline_orientation': self.use_angle_cls,
                        'text_det_limit_side_len': self.det_limit_side_len,
                        'text_recognition_batch_size': self.rec_batch_num,
                        'text_rec_score_thresh': self.min_score,
                        'enable_hpi': False,
                        'enable_mkldnn': False,
                    }
                    if self.det_model:
                        kwargs['text_detection_model_dir'] = self.det_model
                    if self.rec_model:
                        kwargs['text_recognition_model_dir'] = self.rec_model

                    self._ocr = PaddleOCR(**kwargs)
                    self.active_device = device
                    self.init_error = None
                    logger.info(
                        "✅ PaddleOCR disponível "
                        f"(lang={self.lang}, requested_gpu={self.use_gpu}, device={device})"
                    )
                    return True
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"Falha ao inicializar PaddleOCR em {device}: {e}"
                    )

            self._ocr = None
            if last_error is not None:
                self.init_error = f'{type(last_error).__name__}: {last_error}'
            return False
        except Exception as e:
            self.init_error = f'{type(e).__name__}: {e}'
            logger.warning(f"Falha ao inicializar PaddleOCR: {e}")
            self._ocr = None
            return False

    def recognize(self, image: np.ndarray) -> List[OCRResult]:
        """Reconhece texto via PaddleOCR."""
        if not self.is_available or self._ocr is None:
            return []
        if image is None or image.size == 0:
            return []

        try:
            prepared = self._prepare_input_image(image)
            raw = self._ocr.predict(prepared)
        except Exception as e:
            logger.error(f"PaddleOCR falhou: {e}", exc_info=True)
            return []

        return self._parse_paddlex_output(raw)

    @staticmethod
    def _prepare_input_image(image: np.ndarray) -> np.ndarray:
        """Normaliza a imagem para o formato HxWx3 uint8 esperado pelo PaddleOCR."""
        prepared = np.asarray(image)

        if prepared.dtype != np.uint8:
            if (
                np.issubdtype(prepared.dtype, np.floating)
                and prepared.size
                and float(prepared.max()) <= 1.0
            ):
                prepared = (prepared * 255.0).clip(0, 255).astype(np.uint8)
            else:
                prepared = prepared.clip(0, 255).astype(np.uint8)

        if prepared.ndim == 2:
            prepared = cv2.cvtColor(prepared, cv2.COLOR_GRAY2BGR)
        elif prepared.ndim == 3 and prepared.shape[2] == 1:
            prepared = cv2.cvtColor(prepared, cv2.COLOR_GRAY2BGR)
        elif prepared.ndim == 3 and prepared.shape[2] == 4:
            prepared = cv2.cvtColor(prepared, cv2.COLOR_BGRA2BGR)
        elif prepared.ndim != 3 or prepared.shape[2] != 3:
            raise ValueError(f'Formato de imagem OCR nao suportado: {prepared.shape}')

        return np.ascontiguousarray(prepared)

    @classmethod
    def _parse_paddlex_output(cls, raw: Any) -> List[OCRResult]:
        """Converte a saída do PaddleOCR 3.x / PaddleX pipeline para OCRResult."""
        if not raw:
            return []

        first = raw[0]
        if first is None:
            return []

        texts = list(first.get('rec_texts', []) or [])
        scores = list(first.get('rec_scores', []) or [])
        polys = list(first.get('rec_polys', []) or first.get('dt_polys', []) or [])
        if not texts:
            return []

        fragments: List[tuple[str, float, Any]] = []
        for index, text_raw in enumerate(texts):
            cleaned = cls._clean_plate_text(str(text_raw))
            if not cleaned:
                continue
            score = float(scores[index]) if index < len(scores) else 0.0
            bbox = polys[index] if index < len(polys) else None
            fragments.append((cleaned, score, bbox))

        if not fragments:
            return []

        return cls._rank_fragments_to_results(fragments)

    @classmethod
    def _rank_fragments_to_results(
        cls,
        fragments: List[tuple[str, float, Any]],
    ) -> List[OCRResult]:
        """Aplica a mesma estratégia de ranking sobre fragmentos reconhecidos."""
        concat = ''.join(fragment[0] for fragment in fragments)
        concat_clean = cls._clean_plate_text(concat)
        if _RE_MERCOSUL.match(concat_clean) or _RE_OLD.match(concat_clean):
            avg_score = float(np.mean([fragment[1] for fragment in fragments]))
            char_confs = [(ch, avg_score) for ch in concat_clean]
            conf = cls._combine_confidence(avg_score, concat_clean)
            return [create_ocr_result(
                text=concat_clean,
                confidence=conf,
                engine=cls.engine_name,
                char_confidences=char_confs,
                bbox=fragments[0][2] if fragments else None,
            )]

        fragments.sort(
            key=lambda fragment: (cls._format_rank(fragment[0]), fragment[1]),
            reverse=True,
        )
        best_text, best_score, best_bbox = fragments[0]
        char_confs = [(ch, best_score) for ch in best_text]
        conf = cls._combine_confidence(best_score, best_text)
        return [create_ocr_result(
            text=best_text,
            confidence=conf,
            engine=cls.engine_name,
            char_confidences=char_confs,
            bbox=best_bbox,
        )]

    @classmethod
    def _parse_paddle_output(cls, raw: Any) -> List[OCRResult]:
        """
        Converte a saída nativa do PaddleOCR em OCRResult.

        Estrutura esperada (PaddleOCR >= 2.6):
            [[ [bbox, (text, score)], [bbox, (text, score)], ... ]]
        ou None / [] quando nada é detectado.
        """
        if not raw:
            return []

        # PaddleOCR retorna lista-de-listas (1 elemento por imagem)
        lines = raw[0] if isinstance(raw[0], list) or raw[0] is None else raw
        if not lines:
            return []

        # Coletar todos os fragmentos detectados
        fragments: List[tuple[str, float, List[Any]]] = []
        for item in lines:
            if not item or len(item) < 2:
                continue
            bbox = item[0]
            text_score = item[1]
            if not text_score or len(text_score) < 2:
                continue
            text_raw, score = text_score[0], float(text_score[1])
            cleaned = cls._clean_plate_text(str(text_raw))
            if cleaned:
                fragments.append((cleaned, score, bbox))

        if not fragments:
            return []

        return cls._rank_fragments_to_results(fragments)

    @staticmethod
    def _clean_plate_text(text: str) -> str:
        """Remove caracteres não alfanuméricos e normaliza para maiúsculas."""
        if not text:
            return ''
        text = text.strip().upper()
        alnum = re.sub(r'[^A-Z0-9]', '', text)
        # Se há mais de 7 caracteres, tentar extrair uma sequência que case
        # com formato BR (old ou Mercosul)
        if len(alnum) > 7:
            m = re.search(r'[A-Z]{3}[0-9][A-Z0-9][0-9]{2}', alnum)
            if m:
                return m.group(0)
            return alnum[:7]
        return alnum

    @staticmethod
    def _format_rank(text: str) -> int:
        """Score de rank: 2=formato BR exato, 1=7 chars, 0=outro."""
        if _RE_MERCOSUL.match(text) or _RE_OLD.match(text):
            return 2
        if len(text) == 7:
            return 1
        return 0

    @staticmethod
    def _combine_confidence(native_score: float, text: str) -> float:
        """
        Combina a confiança nativa do PaddleOCR com aderência ao formato BR.

        - Native score é per-fragmento, já por-caractere agregado; mantemos como base.
        - Se o texto casa com formato BR, aplicamos um bônus pequeno (cap 1.0).
        - Se não casa, aplicamos um desconto leve.
        """
        native_score = max(0.0, min(1.0, float(native_score)))
        fmt_bonus = format_aderence_confidence(text)  # 0.88 Mercosul, 0.85 Old, etc.
        # Média ponderada: 70% nativo + 30% aderência ao formato
        combined = 0.7 * native_score + 0.3 * fmt_bonus
        return max(0.0, min(1.0, combined))

    def __repr__(self) -> str:
        return (
            f"PaddleOCREngine(lang={self.lang!r}, use_gpu={self.use_gpu}, "
            f"device={self.active_device!r}, available={self.is_available})"
        )
