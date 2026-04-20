# src/ocr/manager.py
"""
OCRManager — orquestrador simplificado para a nova arquitetura.

Rationale:
- Apenas UM engine OCR ativo por vez (selecionado pelo usuário via UI/config).
- GLM OCR é o primário padrão; OLMoOCR2 é alternativa selecionável.
- Não há paralelismo entre engines. Não há voting/merge entre engines diferentes.
- Fallback automático apenas em caso de falha catastrófica (timeout/vazio) do
  engine selecionado, se `auto_fallback_on_failure=True`.

Mantém as features do pipeline:
- Multi-variant: roda o engine em várias versões preprocessadas da mesma placa
  e escolhe a de melhor confiança (_run_ocr_on_variants + _apply_voting).
- visual_format_hint: propaga a dica visual (Mercosul vs old) via argumento.
- char_confidences: engines Ollama não fornecem, mas o pipeline reconstrói no
  LPRPipeline._build_char_confidences.
"""

import logging
from typing import Callable, List, Optional

import numpy as np

from src.ocr.base import OCREngine
from src.ocr.types import OCRResult, clone_ocr_result

logger = logging.getLogger(__name__)


class OCRManager:
    """
    Orquestra um único engine OCR ativo, com fallback on-failure opcional.

    Args:
        engine: Instância do engine primário (GLMOCREngine ou OLMoOCREngine).
        fallback_factory: Callable que instancia o engine de fallback sob
            demanda (evita carregar dois modelos em VRAM à toa).
        auto_fallback_on_failure: Se True e o engine primário retornar vazio
            ou falhar, tenta o fallback uma vez.
        try_multiple_variants: Se True, roda OCR em todas as variantes
            preprocessadas e escolhe a melhor.
        max_variants: Máximo de variantes a processar.
    """

    def __init__(
        self,
        engine: OCREngine,
        fallback_factory: Optional[Callable[[], OCREngine]] = None,
        auto_fallback_on_failure: bool = True,
        try_multiple_variants: bool = True,
        max_variants: int = 5,
    ):
        self.engine = engine
        self.fallback_factory = fallback_factory
        self.auto_fallback_on_failure = auto_fallback_on_failure
        self.try_multiple_variants = try_multiple_variants
        self.max_variants = max_variants
        self._visual_format_hint: Optional[str] = None
        # Mantido como propriedade para compatibilidade com código existente
        # (ex: pipeline.get_pipeline_info)
        self.engines: List[OCREngine] = [engine]

    def recognize(
        self,
        image: np.ndarray,
        original_image: Optional[np.ndarray] = None,
        preprocessed_variants: Optional[List[np.ndarray]] = None,
        visual_format_hint: Optional[str] = None,
    ) -> List[OCRResult]:
        """
        Reconhece placa usando o engine ativo.

        Args:
            image: Imagem preprocessada principal (BGR ou gray).
            original_image: Imagem original sem preprocess — usado se o primário
                falhar com a preprocessada.
            preprocessed_variants: Variantes alternativas (rotações, gamma, etc).
            visual_format_hint: 'mercosul' | 'old' | None — passado adiante.

        Returns:
            Lista com o melhor resultado OCR (tipicamente 1 item).
        """
        if not self.engine:
            logger.error("Nenhum engine OCR configurado")
            return []

        self._visual_format_hint = visual_format_hint

        # Coletar todas as variantes para tentar
        variants: List[np.ndarray] = [image]
        if self.try_multiple_variants and preprocessed_variants:
            variants.extend(preprocessed_variants[: self.max_variants - 1])

        # Rodar engine ativo em todas as variantes
        results = self._run_on_variants(self.engine, variants)

        if results:
            best = self._best_result(results)
            if best:
                return [best]

        # Fallback com imagem original, se fornecida
        if not results and original_image is not None:
            logger.info("OCR primário vazio. Tentando com imagem original...")
            orig_results = self._safe_recognize(self.engine, original_image)
            if orig_results:
                for r in orig_results:
                    r['used_original'] = True
                best = self._best_result(orig_results)
                if best:
                    return [best]

        # Fallback catastrófico para o outro engine (apenas se configurado)
        if self.auto_fallback_on_failure and self.fallback_factory:
            logger.warning(
                f"Engine '{self.engine.engine_name}' não produziu resultado. "
                f"Acionando fallback..."
            )
            try:
                fallback_engine = self.fallback_factory()
                fb_results = self._run_on_variants(fallback_engine, variants)
                if fb_results:
                    best = self._best_result(fb_results)
                    if best:
                        best['used_fallback'] = True
                        logger.info(
                            f"✅ Fallback produziu: '{best['text']}' "
                            f"(conf: {best['confidence']:.2f})"
                        )
                        return [best]
            except Exception as e:
                logger.error(f"Erro ao acionar fallback: {e}")

        return []

    def _run_on_variants(
        self, engine: OCREngine, variants: List[np.ndarray]
    ) -> List[OCRResult]:
        """Roda o engine em cada variante e coleta todos os resultados."""
        all_results: List[OCRResult] = []
        for idx, variant in enumerate(variants):
            if variant is None or variant.size == 0:
                continue
            try:
                results = self._safe_recognize(engine, variant)
                for r in results:
                    all_results.append(clone_ocr_result(r, variant_idx=idx))
            except Exception as e:
                logger.debug(f"Erro na variante {idx}: {e}")
        return all_results

    @staticmethod
    def _safe_recognize(engine: OCREngine, image: np.ndarray) -> List[OCRResult]:
        """Chama engine.recognize com supressão de exceções genéricas."""
        try:
            return engine.recognize(image) or []
        except Exception as e:
            logger.error(f"Erro em {engine.__class__.__name__}.recognize: {e}")
            return []

    @staticmethod
    def _best_result(results: List[OCRResult]) -> Optional[OCRResult]:
        """Retorna o resultado com maior confiança."""
        if not results:
            return None
        return max(results, key=lambda r: r.get('confidence', 0.0))

    def get_status(self) -> dict:
        """Retorna status para exibição na UI."""
        return {
            'active_engine': self.engine.__class__.__name__ if self.engine else None,
            'active_engine_name': getattr(self.engine, 'engine_name', 'unknown'),
            'auto_fallback_on_failure': self.auto_fallback_on_failure,
            'fallback_configured': self.fallback_factory is not None,
            'total_engines': 1,
            # Aliases para retrocompatibilidade com display_pipeline_info
            'primary_engines': [self.engine.__class__.__name__] if self.engine else [],
            'fallback_engines': (
                ['<lazy>'] if self.fallback_factory else []
            ),
            'fallback_enabled': self.auto_fallback_on_failure and bool(self.fallback_factory),
            'fallback_threshold': 0.0,  # Não aplicável
        }
