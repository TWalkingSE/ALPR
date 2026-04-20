"""Sidebar UI for the ALPR 2.0 Streamlit app."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.detector import PlateDetector
from src.video_processor import VehicleMode
from src.v2.config import AppConfig
from src.v2.ollama_validation import build_runtime_profile


@st.cache_data(ttl=15, show_spinner=False)
def _load_ollama_profile(base_url: str, timeout: float) -> dict:
    return build_runtime_profile(base_url, timeout)


def _ordered_model_options(installed_models: list[str], recommended_models: list[str], current_model: str) -> list[str]:
    ordered: list[str] = []
    for model_name in [*recommended_models, *installed_models, current_model]:
        if model_name and model_name not in ordered:
            ordered.append(model_name)
    return ordered


def render_sidebar(config: AppConfig, project_dir: Path) -> tuple[AppConfig, str, bool]:
    """Render the v2 sidebar and return the updated config plus init state."""
    with st.sidebar:
        st.header('ALPR 2.0')
        st.caption('Pipeline local offline com PaddleOCR, desempate LLM opcional via Ollama e Plate Recognizer sob demanda.')

        st.subheader('Deteccao')
        yolo_dir = project_dir / config.detector.models_dir
        yolo_dir.mkdir(parents=True, exist_ok=True)
        available_models = PlateDetector.list_available_models(str(yolo_dir))
        model_names = sorted(Path(item).name for item in available_models)

        if model_names:
            default_index = 0
            if config.detector.model_name in model_names:
                default_index = model_names.index(config.detector.model_name)
            selected_model = st.selectbox('Modelo YOLO', model_names, index=default_index)
            config.detector.model_name = selected_model
            model_path = str(yolo_dir / selected_model)
        else:
            st.warning(f'Nenhum modelo .pt encontrado em {yolo_dir}')
            model_path = str(yolo_dir / config.detector.model_name)

        config.detector.confidence = st.slider(
            'Confianca da deteccao',
            min_value=0.05,
            max_value=0.95,
            value=float(config.detector.confidence),
            step=0.05,
        )
        config.detector.enable_sahi = st.checkbox(
            'Usar SAHI para placas pequenas',
            value=config.detector.enable_sahi,
        )
        if config.detector.enable_sahi:
            config.detector.sahi_slice_size = st.selectbox(
                'Slice SAHI',
                [320, 480, 640, 800, 1024],
                index=[320, 480, 640, 800, 1024].index(config.detector.sahi_slice_size)
                if config.detector.sahi_slice_size in [320, 480, 640, 800, 1024]
                else 2,
            )
            config.detector.sahi_overlap_ratio = st.slider(
                'Overlap SAHI',
                min_value=0.10,
                max_value=0.50,
                value=float(config.detector.sahi_overlap_ratio),
                step=0.05,
            )
        config.detector.crop_margin = st.slider(
            'Margem do crop',
            min_value=0.00,
            max_value=0.30,
            value=float(config.detector.crop_margin),
            step=0.01,
        )
        config.detector.use_gpu = st.checkbox(
            'Permitir GPU no detector',
            value=config.detector.use_gpu,
        )

        st.subheader('OCR local')
        st.info('O fluxo local usa PaddleOCR como OCR padrao.')
        config.ocr.engine = 'paddle'
        config.ocr.try_multiple_variants = st.checkbox(
            'Testar multiplas variantes de preprocessamento',
            value=config.ocr.try_multiple_variants,
        )
        config.ocr.max_variants = st.slider(
            'Maximo de variantes OCR',
            min_value=1,
            max_value=6,
            value=int(config.ocr.max_variants),
            step=1,
        )
        config.ocr.use_gpu = st.checkbox(
            'Permitir GPU no PaddleOCR',
            value=config.ocr.use_gpu,
            help='No Windows a runtime pode cair automaticamente para CPU.',
        )
        config.ocr.use_angle_cls = st.checkbox(
            'Usar correcao de orientacao de texto',
            value=config.ocr.use_angle_cls,
        )

        st.subheader('Validacao inteligente')
        config.llm_validation.enabled = st.checkbox(
            'Usar validacao opcional via Ollama',
            value=config.llm_validation.enabled,
            help='Roda apenas depois do top-k para desempate; nao substitui o PaddleOCR.',
        )
        config.llm_validation.base_url = st.text_input(
            'Endpoint Ollama',
            value=config.llm_validation.base_url,
        )
        runtime_profile = _load_ollama_profile(
            config.llm_validation.base_url,
            min(float(config.llm_validation.timeout), 4.0),
        )
        vram_gb = float(runtime_profile.get('vram_gb', 0.0))
        st.caption(
            'Perfil GPU: '
            f"{runtime_profile.get('profile_label', '16GB')}"
            f" ({vram_gb:.1f} GB detectados)."
        )
        recommended_models = list(runtime_profile.get('recommended_models', []))
        if recommended_models:
            st.caption('Recomendados: ' + ', '.join(recommended_models))

        installed_models = list(runtime_profile.get('installed_models', []))
        model_options = _ordered_model_options(
            installed_models,
            recommended_models,
            config.llm_validation.model,
        )
        auto_label = 'Auto (recomendado)'
        if model_options:
            display_options = [auto_label, *model_options]
            selected_option = st.selectbox(
                'Modelo Ollama',
                options=display_options,
                index=0 if not config.llm_validation.model else display_options.index(config.llm_validation.model),
            )
            config.llm_validation.model = '' if selected_option == auto_label else selected_option
        else:
            config.llm_validation.model = st.text_input(
                'Modelo Ollama',
                value=config.llm_validation.model or str(runtime_profile.get('default_model', '')),
                help='Informe um modelo ja instalado no Ollama caso a listagem automatica falhe.',
            )

        if runtime_profile.get('last_error') and not installed_models:
            st.caption('Ollama nao respondeu ao listar modelos; a selecao manual continua disponivel.')

        if config.llm_validation.enabled:
            config.llm_validation.min_decision_confidence = st.slider(
                'Confianca minima do LLM',
                min_value=0.50,
                max_value=0.95,
                value=float(config.llm_validation.min_decision_confidence),
                step=0.05,
            )
            config.llm_validation.allow_override = st.checkbox(
                'Permitir override do top-1',
                value=config.llm_validation.allow_override,
            )
            config.llm_validation.ambiguity_gap_threshold = st.slider(
                'Gap maximo top-2 para desempate',
                min_value=0.02,
                max_value=0.30,
                value=float(config.llm_validation.ambiguity_gap_threshold),
                step=0.01,
            )
            config.llm_validation.timeout = st.slider(
                'Timeout do Ollama (s)',
                min_value=5.0,
                max_value=60.0,
                value=float(config.llm_validation.timeout),
                step=1.0,
            )

        st.subheader('Premium API')
        config.premium.enabled = st.checkbox(
            'Habilitar Plate Recognizer',
            value=config.premium.enabled,
        )
        if config.premium.api_key:
            st.caption('Chave Premium carregada do arquivo .env.')
        else:
            st.caption('Defina PLATE_RECOGNIZER_API_KEY no arquivo .env para habilitar este fluxo.')
        config.premium.min_confidence = st.slider(
            'Confianca minima Premium',
            min_value=0.10,
            max_value=1.00,
            value=float(config.premium.min_confidence),
            step=0.05,
        )

        st.subheader('Video')
        mode_index = 0 if config.video.vehicle_mode == VehicleMode.MOVING else 1
        mode_label = st.selectbox(
            'Modo de analise',
            options=['moving', 'stationary'],
            index=mode_index,
            format_func=lambda item: 'Veiculo em movimento' if item == 'moving' else 'Veiculo parado',
        )
        config.video.vehicle_mode = VehicleMode(mode_label)
        config.video.skip_frames = st.slider(
            'Processar a cada N frames',
            min_value=1,
            max_value=10,
            value=int(config.video.skip_frames),
            step=1,
        )
        config.video.max_frames = st.number_input(
            'Limite maximo de frames (0 = sem limite)',
            min_value=0,
            value=int(config.video.max_frames),
            step=10,
        )
        config.video.generate_output_video = st.checkbox(
            'Gerar video anotado',
            value=config.video.generate_output_video,
        )
        config.video.enable_temporal_voting = st.checkbox(
            'Usar votacao temporal',
            value=config.video.enable_temporal_voting,
        )

        st.subheader('Diagnostico')
        config.artifacts.enabled = st.checkbox(
            'Salvar artefatos de erro',
            value=config.artifacts.enabled,
        )
        if config.artifacts.enabled:
            config.artifacts.save_low_confidence = st.checkbox(
                'Salvar leituras com baixa confianca',
                value=config.artifacts.save_low_confidence,
            )
            config.artifacts.confidence_threshold = st.slider(
                'Limiar para salvar baixa confianca',
                min_value=0.10,
                max_value=0.95,
                value=float(config.artifacts.confidence_threshold),
                step=0.05,
            )
            st.caption(
                'Os cenarios de baixa iluminacao e placa pequena usam os thresholds '
                'definidos em config.yaml para flexibilizar OCR e fallback.'
            )

        st.markdown('---')
        init_button = st.button('Inicializar ALPR 2.0', type='primary', width='stretch')

    return config, model_path, init_button