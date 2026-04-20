from __future__ import annotations

from src.v2.ollama_validation import OllamaSmartValidator, build_runtime_profile


def test_recommended_models_follow_gpu_tiers():
    assert OllamaSmartValidator.recommended_models_for_vram(15.5) == [
        'qwen3.5:9b-q8_0',
        'gemma4:e4b-it-q8_0',
    ]
    assert OllamaSmartValidator.recommended_models_for_vram(24.0) == [
        'qwen3.5:27b-q4_K_M',
        'gemma4:26b',
    ]
    assert OllamaSmartValidator.recommended_models_for_vram(32.0) == [
        'qwen3.5:35b-a3b-q4_K_M',
        'gemma4:31b-it-q4_K_M',
    ]


def test_suggest_default_model_prefers_installed_recommendation(monkeypatch):
    validator = OllamaSmartValidator(model='')
    monkeypatch.setattr(
        OllamaSmartValidator,
        'detect_gpu_vram_gb',
        staticmethod(lambda: 24.0),
    )

    selected = validator.suggest_default_model(
        installed_models=['tinyllama:latest', 'qwen3.5:27b-q4_K_M'],
    )

    assert selected == 'qwen3.5:27b-q4_K_M'


def test_validate_candidates_parses_ollama_response(monkeypatch):
    seen = {}

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                'response': '{"selected_plate": "ABC1D23", "should_override": true, "decision_confidence": 0.88, "reason": "mercosul wins"}'
            }

    class _FakeClient:
        def __init__(self, timeout):
            seen['timeout'] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json):
            seen['url'] = url
            seen['json'] = json
            return _FakeResponse()

    monkeypatch.setattr('src.v2.ollama_validation.httpx.Client', _FakeClient)

    validator = OllamaSmartValidator(
        base_url='http://127.0.0.1:11434',
        model='qwen3.5:9b-q8_0',
        timeout=9.0,
    )
    result = validator.validate_candidates(
        raw_text='ABC1023',
        current_plate='ABC1023',
        candidates=[
            {'text': 'ABC1023', 'probability': 0.51},
            {'text': 'ABC1D23', 'probability': 0.49},
        ],
        validation_details={'is_valid': True},
        quality_metrics={'snr': 7.0},
        char_confidences=[('A', 0.9)] * 7,
        scenario_tags=['low_light'],
    )

    assert seen['timeout'] == 9.0
    assert seen['url'] == 'http://127.0.0.1:11434/api/generate'
    assert seen['json']['model'] == 'qwen3.5:9b-q8_0'
    assert result['performed'] is True
    assert result['selected_plate'] == 'ABC1D23'
    assert result['should_override'] is True
    assert result['decision_confidence'] == 0.88


def test_build_runtime_profile_aggregates_gpu_and_model_state(monkeypatch):
    monkeypatch.setattr(
        OllamaSmartValidator,
        'detect_gpu_vram_gb',
        staticmethod(lambda: 32.0),
    )
    monkeypatch.setattr(
        OllamaSmartValidator,
        'list_installed_models',
        lambda self: ['gemma4:31b-it-q4_K_M'],
    )

    profile = build_runtime_profile('http://127.0.0.1:11434', 4.0)

    assert profile['profile_label'] == '32GB'
    assert profile['installed_models'] == ['gemma4:31b-it-q4_K_M']
    assert profile['default_model'] == 'gemma4:31b-it-q4_K_M'