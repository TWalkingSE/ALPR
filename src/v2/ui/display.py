"""Display helpers for the ALPR v2 Streamlit app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import cv2
import pandas as pd
import streamlit as st

from src.v2.models import LocalPlateResult


def _engine_label(engine_name: str) -> str:
    key = (engine_name or '').lower()
    if key.startswith('paddle'):
        return 'PaddleOCR'
    if key.startswith('glm'):
        return 'GLM OCR'
    if key.startswith('olmocr'):
        return 'OLMoOCR2'
    return engine_name or '-'


def display_local_result(result: LocalPlateResult, col) -> None:
    """Render a single local recognition result."""
    with col:
        st.markdown(f"### {result.plate_text or '-'}")

        metrics = st.columns(3)
        metrics[0].metric('Confianca OCR', f'{result.confidence:.1%}')
        metrics[1].metric('Conf. deteccao', f'{result.detection_confidence:.1%}')
        metrics[2].metric('Formato', result.format_type.upper())

        tags = [
            'Valida' if result.is_valid else 'Revisar',
            f'OCR: {_engine_label(result.ocr_engine)}',
        ]
        if result.scenario_tags:
            tags.append('Cenario: ' + ', '.join(result.scenario_tags))
        if result.warnings:
            tags.append('Avisos: ' + ', '.join(result.warnings))
        st.caption(' | '.join(tags))

        if result.artifact_dir:
            st.caption(f'Artefatos salvos em: {result.artifact_dir}')
        if result.report_path:
            st.caption(f'Laudo salvo em: {result.report_path}')

        _display_char_confidences(result)

        st.markdown('#### Pipeline visual')
        tab_original, tab_normalized, tab_preprocessed = st.tabs(
            ['Original', 'Normalizada', 'Pre-processada']
        )

        with tab_original:
            original_rgb = cv2.cvtColor(result.original_crop, cv2.COLOR_BGR2RGB)
            st.image(original_rgb, caption='Crop original', width='stretch')

        with tab_normalized:
            if result.normalized_crop is not None:
                normalized_rgb = cv2.cvtColor(result.normalized_crop, cv2.COLOR_BGR2RGB)
                st.image(normalized_rgb, caption='Normalizacao geometrica', width='stretch')

        with tab_preprocessed:
            if result.preprocessed_image is not None:
                image = result.preprocessed_image
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(image, caption='Entrada final do OCR', width='stretch', clamp=True)

        if result.alternative_plates:
            with st.expander(f'Alternativas ({len(result.alternative_plates)})'):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                'Placa': item.get('text', ''),
                                'Probabilidade': f"{float(item.get('probability', 0.0)):.1%}",
                                'Mudancas': item.get('changes', ''),
                                'Apoios': item.get('support_count', 0),
                            }
                            for item in result.alternative_plates
                        ]
                    ),
                    width='stretch',
                    hide_index=True,
                )

        if result.validation_details:
            with st.expander('Validacao detalhada'):
                details = result.validation_details
                cols = st.columns(4)
                cols[0].metric('Prefixo', f"{float(details.get('prefix_score', 0.0)):.0%}")
                cols[1].metric('Fit old', f"{float(details.get('old_format_score', 0.0)):.0%}")
                cols[2].metric('Fit Mercosul', f"{float(details.get('mercosul_format_score', 0.0)):.0%}")
                cols[3].metric('Correcao', 'Sim' if details.get('correction_applied') else 'Nao')
                if details.get('issues'):
                    st.caption(' | '.join(details['issues']))

        if result.llm_validation:
            with st.expander('Validacao inteligente'):
                llm = result.llm_validation
                cols = st.columns(4)
                cols[0].metric('Executada', 'Sim' if llm.get('performed') else 'Nao')
                cols[1].metric('Override', 'Sim' if llm.get('applied_override') else 'Nao')
                cols[2].metric('Confianca', f"{float(llm.get('decision_confidence', 0.0)):.0%}")
                cols[3].metric('Gap top-2', f"{float(llm.get('ambiguity_gap', 0.0)):.0%}")
                if llm.get('model'):
                    st.caption(f"Modelo: {llm['model']}")
                if llm.get('reason'):
                    st.caption(str(llm['reason']))
                if llm.get('selected_plate'):
                    st.caption(f"Selecionada: {llm['selected_plate']}")

        if result.forensic_analysis:
            with st.expander('Revisao forense'):
                forensic = result.forensic_analysis
                cols = st.columns(3)
                cols[0].metric('Score', f"{float(forensic.get('tampering_score', 0.0)):.0%}")
                cols[1].metric('Severidade', str(forensic.get('severity', 'low')).upper())
                cols[2].metric(
                    'Revisao',
                    'Sim' if forensic.get('review_recommended') else 'Nao',
                )
                if forensic.get('signals'):
                    st.caption('Sinais: ' + ', '.join(forensic['signals']))
                metrics_map = forensic.get('metrics', {})
                if metrics_map:
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {'Indicador': key, 'Valor': f'{float(value):.0%}'}
                                for key, value in metrics_map.items()
                            ]
                        ),
                        width='stretch',
                        hide_index=True,
                    )

        if result.report_payload:
            st.download_button(
                'Baixar laudo JSON',
                data=json.dumps(result.report_payload, ensure_ascii=True, indent=2),
                file_name=Path(result.report_path).name if result.report_path else 'alpr_report.json',
                mime='application/json',
            )

        with st.expander('Performance'):
            st.metric('Tempo total', f'{result.processing_time_ms:.2f}ms')
            if result.quality_metrics:
                st.caption(
                    'Qualidade: '
                    f"score {result.quality_score:.1%} | "
                    f"brilho {result.quality_metrics.get('brightness', 0.0):.0f} | "
                    f"contraste {result.quality_metrics.get('contrast', 0.0):.0f} | "
                    f"nitidez {result.quality_metrics.get('sharpness', 0.0):.0f} | "
                    f"snr {result.quality_metrics.get('snr', 0.0):.1f}dB | "
                    f"motion {result.quality_metrics.get('motion_blur', 0.0):.0%}"
                )
            if result.pipeline_steps_time:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                'Etapa': step.replace('_', ' ').title(),
                                'Tempo (ms)': f'{elapsed:.2f}',
                            }
                            for step, elapsed in result.pipeline_steps_time.items()
                        ]
                    ),
                    width='stretch',
                    hide_index=True,
                )


def _display_char_confidences(result: LocalPlateResult) -> None:
    plate_text = ''.join(char for char in (result.plate_text or '') if char.isalnum())
    char_confidences = list(result.char_confidences or [])
    if not plate_text or not char_confidences:
        return

    if len(char_confidences) < len(plate_text):
        char_confidences.extend(
            (plate_text[index], result.confidence)
            for index in range(len(char_confidences), len(plate_text))
        )
    char_confidences = char_confidences[:len(plate_text)]

    def _conf_color(value: float) -> str:
        if value >= 0.85:
            return '#2f855a'
        if value >= 0.65:
            return '#d69e2e'
        if value >= 0.45:
            return '#dd6b20'
        return '#c53030'

    html_chars = []
    for index, (_, confidence) in enumerate(char_confidences):
        char = plate_text[index]
        color = _conf_color(float(confidence))
        html_chars.append(
            f'<div style="display:inline-block;text-align:center;margin:2px;min-width:38px;">'
            f'<div style="font-size:28px;font-weight:bold;font-family:monospace;'
            f'background:{color}22;border:2px solid {color};border-radius:6px;'
            f'padding:4px 6px;color:{color};">{char}</div>'
            f'<div style="font-size:11px;color:{color};font-weight:600;">{confidence:.0%}</div>'
            '</div>'
        )

    with st.expander('Confianca por caractere', expanded=True):
        st.markdown(
            '<div style="display:flex;align-items:flex-start;flex-wrap:wrap;">'
            + ''.join(html_chars)
            + '</div>',
            unsafe_allow_html=True,
        )


def display_pipeline_info(pipeline) -> None:
    """Render a v2-specific pipeline health block in the sidebar."""
    info = pipeline.get_pipeline_info()

    st.sidebar.markdown('---')
    st.sidebar.markdown('### Status da v2')

    status_items = [
        ('Detector', 'OK' if info.get('detector_loaded') else 'OFF'),
        ('Normalizacao', 'OK' if info.get('geometric_normalizer_enabled') else 'OFF'),
        ('Pre-processamento', 'OK' if info.get('preprocessor_enabled') else 'OFF'),
        ('OCR local', f"OK ({info.get('ocr_engines_count', 0)})"),
        ('Validador', 'OK' if info.get('validator_enabled') else 'OFF'),
        ('Ranking de candidatos', 'OK' if info.get('fallback_enabled') else 'OFF'),
        ('LLM opcional', 'ON' if info.get('llm_validation_enabled') else 'OFF'),
        ('Artefatos', 'ON' if info.get('artifact_capture') else 'OFF'),
        ('Qualidade', 'ON' if info.get('quality_assessment_enabled') else 'OFF'),
        ('Forense', 'ON' if info.get('forensic_review_enabled') else 'OFF'),
        ('Laudos', 'ON' if info.get('reporting_enabled') else 'OFF'),
    ]
    for label, value in status_items:
        st.sidebar.text(f'{label}: {value}')

    st.sidebar.caption(
        'Thresholds: '
        f"OCR {float(info.get('ocr_confidence_threshold', 0.0)):.0%} | "
        f"Fallback {float(info.get('fallback_threshold', 0.0)):.0%} | "
        f"Top-K {int(info.get('top_k_candidates', 0))}"
    )

    if info.get('llm_validation_enabled'):
        model_name = info.get('llm_validation_model') or 'auto'
        st.sidebar.caption(f'LLM: Ollama ({model_name})')

    premium = getattr(pipeline, 'premium_provider', None)
    if premium is not None:
        status = 'habilitada' if premium.available else 'indisponivel'
        st.sidebar.caption(f'Premium: {premium.provider} ({status})')


def display_premium_api_comparison(
    local_results: Sequence[LocalPlateResult],
    premium_result,
    premium_time_ms: float = 0.0,
) -> None:
    """Compare the best local result with the Premium API response."""
    st.subheader('Comparacao: local x Premium')

    if not premium_result.success:
        st.error(f"Falha na API Premium: {premium_result.error or 'erro desconhecido'}")
        return

    if not premium_result.plate_text:
        st.warning('A API Premium nao detectou nenhuma placa na imagem completa.')
        if premium_result.error:
            st.caption(premium_result.error)
        return

    local_best = local_results[0] if local_results else None
    local_col, premium_col = st.columns(2)

    with local_col:
        st.markdown('#### Pipeline local')
        if local_best is None:
            st.info('Nenhum resultado local disponivel.')
        else:
            st.metric('Placa', local_best.plate_text)
            st.metric('Confianca', f'{local_best.confidence:.1%}')
            st.caption(
                f"Formato: {local_best.format_type.upper()} | OCR: {_engine_label(local_best.ocr_engine)}"
            )

    with premium_col:
        st.markdown('#### Plate Recognizer')
        st.metric('Placa', premium_result.plate_text)
        st.metric('Confianca', f'{premium_result.confidence:.1%}')
        st.caption(
            f"Formato: {premium_result.format_type.upper()} | "
            f"Regiao: {premium_result.region or '-'} | "
            f"Veiculo: {premium_result.vehicle_type or '-'} | "
            f"Tempo: {premium_time_ms:.0f}ms"
        )

    if local_best is not None:
        local_clean = ''.join(char for char in local_best.plate_text if char.isalnum())
        premium_clean = ''.join(char for char in premium_result.plate_text if char.isalnum())
        if local_clean == premium_clean:
            st.success('As leituras local e Premium coincidem.')
        else:
            max_length = max(len(local_clean), len(premium_clean))
            st.warning('As leituras divergem entre o pipeline local e a API Premium.')
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            'Posicao': index + 1,
                            'Local': local_clean[index] if index < len(local_clean) else '-',
                            'Premium': premium_clean[index] if index < len(premium_clean) else '-',
                            'Match': 'OK'
                            if index < len(local_clean)
                            and index < len(premium_clean)
                            and local_clean[index] == premium_clean[index]
                            else 'Diff',
                        }
                        for index in range(max_length)
                    ]
                ),
                width='stretch',
                hide_index=True,
            )

    if premium_result.alternates:
        with st.expander('Alternativas da API'):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            'Placa': item.get('plate', ''),
                            'Score': f"{float(item.get('score', 0.0)):.1%}",
                        }
                        for item in premium_result.alternates
                    ]
                ),
                width='stretch',
                hide_index=True,
            )

    with st.expander('Resposta bruta da API', expanded=False):
        st.json(premium_result.raw_response)


def display_summary_table(results: Sequence[LocalPlateResult]) -> None:
    """Render a compact summary of local v2 results."""
    st.markdown('---')
    st.subheader('Resumo local')
    st.dataframe(
        pd.DataFrame(
            [
                {
                    '#': index,
                    'Placa': result.plate_text,
                    'Formato': result.format_type.upper(),
                    'Valida': 'OK' if result.is_valid else 'Revisar',
                    'Conf.OCR': f'{result.confidence:.1%}',
                    'Conf.Det': f'{result.detection_confidence:.1%}',
                    'OCR': _engine_label(result.ocr_engine),
                    'Cenario': ', '.join(result.scenario_tags) if result.scenario_tags else '-',
                    'Avisos': ', '.join(result.warnings) if result.warnings else '-',
                    'Tempo(ms)': f'{result.processing_time_ms:.2f}',
                }
                for index, result in enumerate(results, start=1)
            ]
        ),
        width='stretch',
        hide_index=True,
    )


def display_video_results(video_result, video_processor) -> None:
    """Render the consolidated video analysis output for v2."""
    st.subheader('Resultado do video')

    metrics = st.columns(4)
    metrics[0].metric('Frames processados', video_result.processed_frames)
    metrics[1].metric('Frames pulados', video_result.skipped_frames)
    metrics[2].metric('Placas unicas', len(video_result.unique_plates))
    metrics[3].metric('FPS efetivo', f'{video_result.processing_fps:.2f}')

    ranked = video_processor.rank_unique_plates(video_result.unique_plates)
    if ranked:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        'Placa': info['plate_text'],
                        'Leitura confirmada': video_processor.build_confirmed_reading(info),
                        'Score composto': f"{float(info.get('composite_score', 0.0)):.1%}",
                        'Melhor confianca': f"{info['best_confidence']:.1%}",
                        'Conf.char': f"{float(info.get('char_confirmation_ratio', 0.0)):.1%}",
                        'Deteccoes': info['total_detections'],
                        'Melhor frame': info.get('best_frame_number', '-'),
                        'Primeiro frame': info['first_seen_frame'],
                        'Ultimo frame': info['last_seen_frame'],
                    }
                    for info in ranked.values()
                ]
            ),
            width='stretch',
            hide_index=True,
        )
    else:
        st.info('Nenhuma placa consolidada no video.')

    timeline = video_processor.generate_timeline(video_result)
    if timeline:
        with st.expander('Timeline de deteccoes'):
            st.dataframe(pd.DataFrame(timeline), width='stretch', hide_index=True)

    output_path = getattr(video_result, 'output_video_path', '')
    if output_path and Path(output_path).exists():
        with open(output_path, 'rb') as handle:
            st.video(handle.read())