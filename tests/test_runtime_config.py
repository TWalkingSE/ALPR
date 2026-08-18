# tests/test_runtime_config.py
"""
Separação entre configuração de CONSTRUÇÃO e de RUNTIME.

Antes, `AppConfig.signature()` cobria ~90 campos e qualquer mudança — inclusive
o limiar que decide se um PNG de diagnóstico vai para o disco — invalidava o
bundle e recarregava o modelo YOLO e o PaddleOCR. Estes testes travam o
critério: só a identidade dos modelos entra na assinatura; o resto é aplicado
no pipeline vivo.
"""

from types import SimpleNamespace

import pytest

from src.v2.application import ensure_service_bundle
from src.v2.config import AppConfig, build_v2_config
from src.v2.contracts import ServiceBundle


@pytest.fixture
def config():
    return AppConfig()


class TestSignatureScope:
    """Campos que NÃO podem forçar rebuild."""

    @pytest.mark.parametrize(
        ('path', 'value'),
        [
            (('artifacts', 'confidence_threshold'), 0.42),
            (('artifacts', 'enabled'), False),
            (('artifacts', 'save_invalid'), False),
            (('reports', 'output_dir'), 'outro/caminho'),
            (('reports', 'enabled'), False),
            (('video', 'skip_frames'), 9),
            (('video', 'max_frames'), 123),
            (('video', 'generate_output_video'), False),
            (('ocr', 'confidence_threshold'), 0.11),
            (('ocr', 'fallback_threshold'), 0.99),
            (('ocr', 'max_variants'), 2),
            (('ocr', 'top_k_candidates'), 2),
            (('ocr', 'try_multiple_variants'), False),
            (('ocr', 'add_quiet_zone'), True),
            (('detector', 'confidence'), 0.77),
            (('detector', 'enable_sahi'), False),
            (('detector', 'sahi_slice_size'), 1024),
            (('detector', 'crop_margin'), 0.3),
            (('quality', 'enabled'), False),
            (('forensic', 'enabled'), False),
            (('llm_validation', 'enabled'), True),
            (('premium', 'min_confidence'), 0.95),
            (('vehicle_attributes', 'enabled'), True),
        ],
    )
    def test_runtime_field_does_not_change_signature(self, config, path, value):
        before = config.signature()
        section, field = path
        setattr(getattr(config, section), field, value)
        assert config.signature() == before, (
            f'{section}.{field} não deveria invalidar os modelos carregados'
        )

    @pytest.mark.parametrize(
        ('path', 'value'),
        [
            (('detector', 'model_name'), 'outro-modelo.pt'),
            (('detector', 'models_dir'), 'outro/dir'),
            (('detector', 'device'), 'cpu'),
            (('detector', 'use_gpu'), False),
            (('ocr', 'lang'), 'en'),
            (('ocr', 'use_gpu'), False),
            (('ocr', 'use_angle_cls'), False),
            (('ocr', 'det_limit_side_len'), 640),
            (('ocr', 'rec_batch_num'), 1),
            (('ocr', 'min_score'), 0.9),
            (('premium', 'enabled'), True),
            (('premium', 'provider'), 'outro'),
            (('premium', 'api_key'), 'x' * 20),
        ],
    )
    def test_build_field_changes_signature(self, config, path, value):
        before = config.signature()
        section, field = path
        setattr(getattr(config, section), field, value)
        assert config.signature() != before, (
            f'{section}.{field} exige reconstruir os objetos pesados'
        )


class TestEnsureServiceBundleAppliesRuntimeConfig:
    def test_reused_bundle_receives_runtime_config(self):
        pipeline = SimpleNamespace(applied=[])
        pipeline.apply_runtime_config = pipeline.applied.append
        premium = SimpleNamespace(applied=[])
        premium.apply_runtime_config = premium.applied.append

        config = AppConfig()
        session = {
            'v2_pipeline': pipeline,
            'v2_premium': premium,
            'v2_signature': ('model.pt', config.signature()),
        }
        builder = lambda *args, **kwargs: pytest.fail('não deveria reconstruir')  # noqa: E731

        ensure_service_bundle(session, config, 'proj', 'model.pt', builder=builder)

        assert pipeline.applied == [config]
        assert premium.applied == [config.premium]

    def test_rebuild_does_not_double_apply(self):
        bundle = ServiceBundle(pipeline=SimpleNamespace(), premium=SimpleNamespace())
        config = AppConfig()

        result = ensure_service_bundle(
            {}, config, 'proj', 'model.pt', builder=lambda *a, **k: bundle
        )

        assert result is bundle

    def test_tolerates_services_without_the_hook(self):
        """Dublês de teste não precisam implementar apply_runtime_config."""
        config = AppConfig()
        session = {
            'v2_pipeline': SimpleNamespace(),
            'v2_premium': SimpleNamespace(),
            'v2_signature': ('model.pt', config.signature()),
        }

        bundle = ensure_service_bundle(
            session, config, 'proj', 'model.pt',
            builder=lambda *a, **k: pytest.fail('não deveria reconstruir'),
        )

        assert bundle.pipeline is session['v2_pipeline']


class TestSidebarChangesDoNotRebuild:
    def test_artifact_slider_keeps_signature(self):
        """Cenário concreto: mover o slider de artefatos na sidebar."""
        raw = {'artifacts': {'confidence_threshold': 0.75}}
        before = build_v2_config(raw).signature()
        after = build_v2_config({'artifacts': {'confidence_threshold': 0.30}}).signature()
        assert before == after


class TestVideoProcessorCaching:
    """`_build_video_processor` roda a cada rerun do Streamlit.

    Reconstruir o objeto recria o diretório de saída e reinicializa a votação
    temporal sem necessidade — a renderização só usa os formatadores.
    """

    @staticmethod
    def _config(**overrides):
        base = {
            'vehicle_mode': 'moving',
            'skip_frames': 2,
            'max_frames': 0,
            'generate_output_video': False,
            'confidence_threshold': 0.6,
            'enable_temporal_voting': False,
            'temporal_strategy': 'hybrid',
            'temporal_min_observations': 2,
        }
        base.update(overrides)
        return SimpleNamespace(video=SimpleNamespace(**base))

    def test_same_config_reuses_the_instance(self, tmp_path):
        import app

        app.st.session_state.pop('v2_video_processor', None)
        app.st.session_state.pop('v2_video_processor_signature', None)

        config = self._config(output_dir=str(tmp_path))
        first = app._build_video_processor(config)
        second = app._build_video_processor(config)

        assert first is second

    def test_changed_config_rebuilds(self, tmp_path):
        import app

        app.st.session_state.pop('v2_video_processor', None)
        app.st.session_state.pop('v2_video_processor_signature', None)

        first = app._build_video_processor(self._config(output_dir=str(tmp_path), skip_frames=2))
        second = app._build_video_processor(self._config(output_dir=str(tmp_path), skip_frames=7))

        assert first is not second
        assert second.skip_frames == 7

    def test_signature_normalizes_enum_and_string_vehicle_mode(self, tmp_path):
        import app
        from src.video_processor import VehicleMode

        como_enum = app._video_processor_signature(
            self._config(output_dir=str(tmp_path), vehicle_mode=VehicleMode.MOVING)
        )
        como_texto = app._video_processor_signature(
            self._config(output_dir=str(tmp_path), vehicle_mode='moving')
        )

        assert como_enum == como_texto
