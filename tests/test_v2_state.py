from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.v2.models import LocalPlateResult
from src.v2.state import (
    read_app_state,
    store_local_results,
    store_premium_result,
    store_service_bundle,
    store_video_result,
)


def _local_result() -> LocalPlateResult:
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    return LocalPlateResult(
        plate_text='ABC1D23',
        confidence=0.93,
        detection_confidence=0.88,
        format_type='mercosul',
        is_valid=True,
        original_crop=image,
        bbox=(1, 2, 10, 5),
    )


def test_store_and_read_app_state_roundtrip():
    session = {}
    pipeline = SimpleNamespace(name='pipeline')
    premium = SimpleNamespace(name='premium')
    premium_result = SimpleNamespace(success=True)
    video_result = SimpleNamespace(processed_frames=4)

    store_service_bundle(
        session,
        SimpleNamespace(pipeline=pipeline, premium=premium),
        signature=('model.pt', ('sig',)),
    )
    store_local_results(session, [_local_result()], 'plate.jpg')
    store_premium_result(session, premium_result, 45.6)
    store_video_result(session, video_result)

    state = read_app_state(session)

    assert state.pipeline is pipeline
    assert state.premium_service is premium
    assert state.signature == ('model.pt', ('sig',))
    assert len(state.image.local_results) == 1
    assert state.image.local_image_name == 'plate.jpg'
    assert state.image.premium_result is premium_result
    assert state.image.premium_time_ms == 45.6
    assert state.video_result is video_result