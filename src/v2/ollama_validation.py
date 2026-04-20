"""Optional Ollama-backed smart validation used only after deterministic top-k."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Sequence

import httpx

logger = logging.getLogger(__name__)

GPU_MODEL_PROFILES = [
    {
        'min_vram_gb': 32.0,
        'recommended_models': [
            'qwen3.5:35b-a3b-q4_K_M',
            'gemma4:31b-it-q4_K_M',
        ],
        'label': '32GB',
    },
    {
        'min_vram_gb': 24.0,
        'recommended_models': [
            'qwen3.5:27b-q4_K_M',
            'gemma4:26b',
        ],
        'label': '24GB',
    },
    {
        'min_vram_gb': 0.0,
        'recommended_models': [
            'qwen3.5:9b-q8_0',
            'gemma4:e4b-it-q8_0',
        ],
        'label': '16GB',
    },
]


class OllamaSmartValidator:
    """Optional LLM tiebreaker for OCR candidate validation."""

    def __init__(
        self,
        base_url: str = 'http://127.0.0.1:11434',
        model: str = '',
        timeout: float = 20.0,
        allow_override: bool = True,
        ambiguity_gap_threshold: float = 0.12,
        min_decision_confidence: float = 0.70,
    ):
        self.base_url = base_url.rstrip('/')
        self.model = model.strip()
        self.timeout = float(timeout)
        self.allow_override = bool(allow_override)
        self.ambiguity_gap_threshold = float(ambiguity_gap_threshold)
        self.min_decision_confidence = float(min_decision_confidence)
        self._installed_models: List[str] | None = None
        self.last_resolved_model: str = ''
        self.last_error: str = ''

    @classmethod
    def from_settings(cls, config) -> 'OllamaSmartValidator':
        return cls(
            base_url=config.base_url,
            model=config.model,
            timeout=config.timeout,
            allow_override=config.allow_override,
            ambiguity_gap_threshold=config.ambiguity_gap_threshold,
            min_decision_confidence=config.min_decision_confidence,
        )

    @staticmethod
    def detect_gpu_vram_gb() -> float:
        try:
            import torch

            if not torch.cuda.is_available():
                return 0.0
            device_count = torch.cuda.device_count()
            if device_count <= 0:
                return 0.0
            return max(
                float(torch.cuda.get_device_properties(index).total_memory / (1024**3))
                for index in range(device_count)
            )
        except Exception:
            return 0.0

    @classmethod
    def recommended_models_for_vram(cls, vram_gb: float) -> List[str]:
        for profile in GPU_MODEL_PROFILES:
            if vram_gb >= profile['min_vram_gb']:
                return list(profile['recommended_models'])
        return list(GPU_MODEL_PROFILES[-1]['recommended_models'])

    @classmethod
    def profile_label_for_vram(cls, vram_gb: float) -> str:
        if vram_gb <= 0:
            return 'CPU / ate 16GB'
        for profile in GPU_MODEL_PROFILES:
            if vram_gb >= profile['min_vram_gb']:
                return str(profile['label'])
        return '16GB'

    def list_installed_models(self, refresh: bool = False) -> List[str]:
        if self._installed_models is not None and not refresh:
            return list(self._installed_models)
        try:
            with httpx.Client(timeout=min(self.timeout, 4.0)) as client:
                response = client.get(f'{self.base_url}/api/tags')
                response.raise_for_status()
            payload = response.json()
            models = sorted(
                {
                    str(item.get('name', '')).strip()
                    for item in payload.get('models', [])
                    if str(item.get('name', '')).strip()
                }
            )
            self._installed_models = models
            self.last_error = ''
            return list(models)
        except Exception as exc:
            self.last_error = str(exc)
            logger.debug('Falha ao listar modelos Ollama: %s', exc)
            self._installed_models = []
            return []

    def suggest_default_model(self, installed_models: Sequence[str] | None = None) -> str:
        installed = list(installed_models) if installed_models is not None else self.list_installed_models()
        recommended = self.recommended_models_for_vram(self.detect_gpu_vram_gb())
        for candidate in recommended:
            if candidate in installed:
                return candidate
        if self.model:
            return self.model
        if installed:
            return installed[0]
        return recommended[0] if recommended else ''

    def resolve_model(self) -> str:
        if self.model:
            self.last_resolved_model = self.model
            return self.model
        chosen = self.suggest_default_model()
        self.last_resolved_model = chosen
        return chosen

    def validate_candidates(
        self,
        raw_text: str,
        current_plate: str,
        candidates: Sequence[Dict[str, Any]],
        validation_details: Dict[str, Any],
        quality_metrics: Dict[str, float],
        char_confidences: Sequence[tuple[str, float]],
        scenario_tags: Sequence[str],
    ) -> Dict[str, Any]:
        resolved_model = self.resolve_model()
        if not resolved_model:
            return {
                'enabled': True,
                'performed': False,
                'reason': 'no_model_available',
                'available': False,
                'model': '',
            }

        prompt_payload = {
            'raw_text': raw_text,
            'current_plate': current_plate,
            'candidates': list(candidates),
            'validation_details': validation_details,
            'quality_metrics': quality_metrics,
            'char_confidences': [[char, confidence] for char, confidence in char_confidences],
            'scenario_tags': list(scenario_tags),
            'rules': {
                'old_format': 'AAA1234',
                'mercosul_format': 'AAA1A23',
                'mercosul_position_5_cannot_be_vowel': True,
                'must_choose_from_candidates_only': True,
                'abstain_when_uncertain': True,
            },
        }
        prompt = (
            'Voce esta validando candidatos OCR de placa brasileira. '
            'Nao invente placa nova. Escolha apenas uma das placas candidatas fornecidas ou ABSTAIN. '
            'Use o OCR bruto, os scores determinísticos, o formato brasileiro e a confiança por caractere. '
            'Retorne JSON estrito com as chaves selected_plate, should_override, decision_confidence e reason.\n\n'
            f'{json.dumps(prompt_payload, ensure_ascii=True, indent=2)}'
        )

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f'{self.base_url}/api/generate',
                    json={
                        'model': resolved_model,
                        'prompt': prompt,
                        'stream': False,
                        'format': 'json',
                        'options': {'temperature': 0, 'top_p': 0.1, 'num_predict': 200},
                    },
                )
                response.raise_for_status()
            payload = response.json()
            parsed = self._parse_llm_response(str(payload.get('response', '')).strip())
        except Exception as exc:
            self.last_error = str(exc)
            logger.info('LLM validation indisponível: %s', exc)
            return {
                'enabled': True,
                'performed': False,
                'available': False,
                'model': resolved_model,
                'reason': 'request_failed',
                'error': str(exc),
            }

        selected_plate = str(parsed.get('selected_plate', 'ABSTAIN')).strip().upper()
        decision_confidence = float(parsed.get('decision_confidence', 0.0) or 0.0)
        should_override = bool(parsed.get('should_override', False))
        candidate_texts = {str(item.get('text', '')).strip().upper() for item in candidates}
        if selected_plate not in candidate_texts:
            selected_plate = 'ABSTAIN'
            should_override = False

        return {
            'enabled': True,
            'performed': True,
            'available': True,
            'model': resolved_model,
            'selected_plate': selected_plate,
            'decision_confidence': decision_confidence,
            'should_override': should_override,
            'reason': str(parsed.get('reason', '')).strip(),
            'allow_override': self.allow_override,
        }

    @staticmethod
    def _parse_llm_response(response_text: str) -> Dict[str, Any]:
        if not response_text:
            return {}
        text = response_text.strip()
        if text.startswith('```'):
            text = text.strip('`')
            if text.lower().startswith('json'):
                text = text[4:].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find('{')
            end = text.rfind('}')
            if start >= 0 and end > start:
                return json.loads(text[start:end + 1])
            raise


def build_runtime_profile(base_url: str, timeout: float) -> Dict[str, Any]:
    validator = OllamaSmartValidator(base_url=base_url, timeout=timeout)
    vram_gb = validator.detect_gpu_vram_gb()
    installed_models = validator.list_installed_models()
    default_model = validator.suggest_default_model(installed_models)
    return {
        'vram_gb': vram_gb,
        'profile_label': validator.profile_label_for_vram(vram_gb),
        'recommended_models': validator.recommended_models_for_vram(vram_gb),
        'installed_models': installed_models,
        'default_model': default_model,
        'available': bool(installed_models),
        'last_error': validator.last_error,
    }