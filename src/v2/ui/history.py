"""Aba de histórico de leituras para o app Streamlit."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import pandas as pd
import streamlit as st

# Rótulo -> quantos dias para trás. None = sem limite inferior.
PERIOD_OPTIONS = {
    'Todo o periodo': None,
    'Ultimas 24h': 1,
    'Ultimos 7 dias': 7,
    'Ultimos 30 dias': 30,
    'Personalizado': 'custom',
}


def _resolve_period(label, custom_range=None):
    """Converte a escolha de período nos limites ISO-8601 usados pelo store.

    O `created_at` é gravado em UTC (`datetime.now(UTC).isoformat()` em
    `src/v2/storage.py`), então os limites também precisam ser UTC — comparar
    com a data local deslocaria a janela pelo fuso do usuário.
    """
    option = PERIOD_OPTIONS.get(label)

    if option is None:
        return None, None

    if option == 'custom':
        # O date_input devolve uma tupla só depois que as duas pontas foram
        # escolhidas; antes disso, não filtra.
        if not custom_range or len(custom_range) != 2:
            return None, None
        inicio, fim = custom_range
        since = datetime.combine(inicio, time.min, tzinfo=UTC).isoformat()
        until = datetime.combine(fim, time.max, tzinfo=UTC).isoformat()
        return since, until

    since = (datetime.now(UTC) - timedelta(days=int(option))).isoformat()
    return since, None


COLUMN_LABELS = {
    'created_at': 'Quando (UTC)',
    'plate_text': 'Placa',
    'format_type': 'Formato',
    'is_valid': 'Valida',
    'ocr_confidence': 'Conf.OCR',
    'detection_confidence': 'Conf.Det',
    'quality_score': 'Qualidade',
    'origin': 'Origem',
    'source_path': 'Arquivo',
    'report_path': 'Laudo',
}


def _rows_to_frame(rows) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                COLUMN_LABELS['created_at']: row.created_at.replace('T', ' ')[:19],
                COLUMN_LABELS['plate_text']: row.plate_text,
                COLUMN_LABELS['format_type']: row.format_type.upper(),
                COLUMN_LABELS['is_valid']: 'OK' if row.is_valid else 'Revisar',
                COLUMN_LABELS['ocr_confidence']: f'{row.ocr_confidence:.1%}',
                COLUMN_LABELS['detection_confidence']: f'{row.detection_confidence:.1%}',
                COLUMN_LABELS['quality_score']: f'{row.quality_score:.1%}',
                COLUMN_LABELS['origin']: row.origin,
                COLUMN_LABELS['source_path']: row.source_path,
                COLUMN_LABELS['report_path']: row.report_path,
            }
            for row in rows
        ]
    )


def render_history_tab(store: Any | None) -> None:
    """Renderiza a aba de histórico.

    `store` é um `ReadingStore` ou None quando `storage.enabled` está desligado.
    """
    st.subheader('Historico de leituras')

    if store is None or not getattr(store, 'enabled', False):
        st.info(
            'O historico esta desabilitado. Ative `storage.enabled: true` no '
            'config.yaml e reinicialize a aplicacao para registrar as leituras.'
        )
        return

    stats = store.stats()
    metrics = st.columns(4)
    metrics[0].metric('Leituras', stats['total'])
    metrics[1].metric('Validas', stats['validas'])
    metrics[2].metric('Placas distintas', stats['placas_distintas'])
    metrics[3].metric('Videos processados', stats['videos'])

    if stats['total'] == 0:
        st.caption('Nenhuma leitura registrada ainda.')
        return

    st.markdown('---')

    filtros = st.columns([3, 2, 2])
    consulta = filtros[0].text_input(
        'Buscar placa',
        key='v2_history_query',
        placeholder='Casamento parcial, ex: ABC ou 1D23',
    )
    validade = filtros[1].selectbox(
        'Validade',
        options=['Todas', 'Apenas validas', 'Apenas para revisar'],
        key='v2_history_validity',
    )
    limite = filtros[2].number_input(
        'Maximo de linhas',
        min_value=10,
        max_value=2000,
        value=200,
        step=10,
        key='v2_history_limit',
    )

    periodo = st.columns([2, 2, 4])
    periodo_label = periodo[0].selectbox(
        'Periodo',
        options=list(PERIOD_OPTIONS),
        key='v2_history_period',
    )
    intervalo = None
    if periodo_label == 'Personalizado':
        intervalo = periodo[1].date_input(
            'Intervalo',
            value=(date.today() - timedelta(days=7), date.today()),
            key='v2_history_range',
        )

    only_valid = None
    if validade == 'Apenas validas':
        only_valid = True
    elif validade == 'Apenas para revisar':
        only_valid = False

    since, until = _resolve_period(periodo_label, intervalo)

    rows = store.search(
        plate=consulta,
        only_valid=only_valid,
        since=since,
        until=until,
        limit=int(limite),
    )
    if not rows:
        st.warning('Nenhuma leitura encontrada com esses filtros.')
        return

    frame = _rows_to_frame(rows)
    st.caption(f'{len(rows)} leitura(s)')
    st.dataframe(frame, width='stretch', hide_index=True)

    st.download_button(
        'Baixar CSV',
        data=frame.to_csv(index=False).encode('utf-8'),
        file_name='alpr_historico.csv',
        mime='text/csv',
    )

    top = store.top_plates(limit=10)
    if len(top) > 1:
        with st.expander('Placas mais lidas'):
            st.dataframe(
                pd.DataFrame(
                    [{'Placa': placa, 'Leituras': total} for placa, total in top]
                ),
                width='stretch',
                hide_index=True,
            )
