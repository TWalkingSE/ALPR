"""Streamlit entrypoint for ALPR 2.0."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

import cv2
import streamlit as st
from dotenv import load_dotenv

st.set_page_config(
    page_title='ALPR 2.0',
    layout='wide',
    initial_sidebar_state='expanded',
    page_icon='🚗',
)

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from src.config_manager import load_config
from src.video_processor import VideoProcessor
from src.v2 import build_v2_config
from src.v2.application import (
    build_service_bundle,
    build_video_processor as build_video_processor_impl,
    decode_uploaded_image as decode_uploaded_image_impl,
    ensure_service_bundle,
    run_local_image_analysis,
    run_premium_image_analysis,
    run_video_analysis,
)
from src.v2.models import LocalPlateResult
from src.v2.state import (
    read_app_state,
    store_local_results,
    store_premium_result,
    store_video_result,
)
from src.v2.ui import (
    display_local_result,
    display_pipeline_info,
    display_premium_api_comparison,
    display_summary_table,
    display_video_results,
    render_sidebar,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def _decode_uploaded_image(uploaded_file):
    return decode_uploaded_image_impl(uploaded_file)


def _build_services(config, model_path: str):
    bundle = build_service_bundle(config, PROJECT_DIR, model_path)
    return bundle.pipeline, bundle.premium


def _ensure_services(config, model_path: str, force_rebuild: bool = False):
    state = read_app_state(st.session_state)
    signature = (model_path, config.signature())
    if force_rebuild or state.pipeline is None or state.signature != signature:
        with st.spinner('Inicializando ALPR 2.0...'):
            bundle = ensure_service_bundle(
                st.session_state,
                config,
                PROJECT_DIR,
                model_path,
                force_rebuild=force_rebuild,
            )
    else:
        bundle = ensure_service_bundle(st.session_state, config, PROJECT_DIR, model_path)
    return bundle.pipeline, bundle.premium


def _store_local_results(
    results: List[LocalPlateResult],
    image_name: str,
    session_state=None,
):
    session = session_state if session_state is not None else st.session_state
    store_local_results(session, results, image_name)


def _store_premium_result(premium_result, elapsed_ms: float, session_state=None):
    session = session_state if session_state is not None else st.session_state
    store_premium_result(session, premium_result, elapsed_ms)


def _store_video_result(video_result, session_state=None):
    session = session_state if session_state is not None else st.session_state
    store_video_result(session, video_result)


def _build_video_processor(config) -> VideoProcessor:
    return build_video_processor_impl(config)


def _render_image_outputs(session_state=None):
    session = session_state if session_state is not None else st.session_state
    state = read_app_state(session)
    local_results = state.image.local_results
    premium_result = state.image.premium_result
    premium_time_ms = state.image.premium_time_ms

    if local_results:
        _display_local_results(local_results)

    if premium_result is not None:
        display_premium_api_comparison(
            local_results,
            premium_result,
            premium_time_ms=premium_time_ms,
        )


def _render_video_outputs(config, session_state=None):
    session = session_state if session_state is not None else st.session_state
    video_result = read_app_state(session).video_result
    if video_result is None:
        return
    video_processor = _build_video_processor(config)
    display_video_results(video_result, video_processor)


def _display_local_results(results: List[LocalPlateResult]):
    if not results:
        st.warning('Nenhuma placa foi reconhecida localmente.')
        return

    display_summary_table(results)

    for index in range(0, len(results), 2):
        cols = st.columns(2)
        for col, result in zip(cols, results[index:index + 2]):
            display_local_result(result, col)


def main():
    st.title('ALPR 2.0')
    st.caption(
        'Pipeline local offline com YOLO + PaddleOCR, desempate inteligente opcional '
        'via Ollama apos o top-k e validacao cruzada via Plate Recognizer.'
    )

    raw_config = load_config()
    config = build_v2_config(raw_config)
    config, model_path, init_button = render_sidebar(config, PROJECT_DIR)

    try:
        pipeline, premium_service = _ensure_services(config, model_path, force_rebuild=init_button)
    except Exception as exc:
        st.error(f'Falha ao inicializar a aplicacao: {exc}')
        logger.error('Falha ao inicializar a aplicacao', exc_info=True)
        return

    display_pipeline_info(pipeline)

    image_tab, video_tab = st.tabs(['Imagem', 'Video'])

    with image_tab:
        uploaded_image = st.file_uploader(
            'Envie uma imagem de placa',
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            key='v2_image_upload',
        )

        if uploaded_image is not None:
            image, image_bytes = _decode_uploaded_image(uploaded_image)
            if image is None:
                st.error('Nao foi possivel ler a imagem enviada.')
            else:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=uploaded_image.name, width='stretch')
                col_local, col_premium = st.columns(2)

                with col_local:
                    if st.button('Processar localmente', key='v2_run_local', width='stretch'):
                        with st.spinner('Executando pipeline local...'):
                            local_results = run_local_image_analysis(
                                pipeline,
                                config,
                                image,
                                image_bytes,
                                uploaded_image.name,
                            )
                        _store_local_results(local_results, uploaded_image.name)

                with col_premium:
                    disabled = not premium_service.available
                    if st.button(
                        'Analisar com Plate Recognizer',
                        key='v2_run_premium',
                        width='stretch',
                        disabled=disabled,
                    ):
                        with st.spinner('Consultando API Premium...'):
                            premium_result, elapsed_ms = run_premium_image_analysis(
                                premium_service,
                                image,
                            )
                        _store_premium_result(premium_result, elapsed_ms)

                    if disabled:
                        st.caption('Habilite a API Premium e configure PLATE_RECOGNIZER_API_KEY no .env para usar este fluxo.')

        _render_image_outputs()

    with video_tab:
        uploaded_video = st.file_uploader(
            'Envie um video',
            type=VideoProcessor.get_extensions_for_uploader(),
            key='v2_video_upload',
        )

        if uploaded_video is not None:
            if st.button('Processar video', key='v2_run_video', width='stretch'):
                progress = st.progress(0)
                status = st.empty()

                def _on_progress(current: int, total: int, frame_result):
                    if total > 0:
                        progress.progress(min(current / total, 1.0))
                    if frame_result is not None and frame_result.plates_found > 0:
                        status.text(
                            f'Frame {current}/{total} | placas: {", ".join(frame_result.plate_texts)}'
                        )
                    else:
                        status.text(f'Frame {current}/{total}')

                with st.spinner('Processando video...'):
                    video_result = run_video_analysis(
                        uploaded_video.getvalue(),
                        uploaded_video.name,
                        pipeline,
                        config,
                        progress_callback=_on_progress,
                    )
                    _store_video_result(video_result)

                progress.empty()
                status.empty()

        _render_video_outputs(config)


if __name__ == '__main__':
    main()
