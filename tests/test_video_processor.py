# tests/test_video_processor.py
"""
Testes para o VideoProcessor (src/video_processor.py).

Cobre:
  - VehicleMode params (STATIONARY / MOVING)
  - Extensões suportadas + is_supported()
  - _normalize_plate / _format_duration / _compute_sharpness (estáticos)
  - _aggregate_plates — agregação de placas únicas
  - _annotate_frame — anotação visual
  - _get_output_path — composição do caminho
  - build_confirmed_reading — leitura confirmada por caractere
  - rank_unique_plates — ranking composto
  - generate_timeline — timeline de detecções
  - VideoResult / FrameResult — propriedades
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from src.video_processor import (
    VideoProcessor,
    VideoResult,
    FrameResult,
    VehicleMode,
    VEHICLE_MODE_PARAMS,
    SUPPORTED_VIDEO_EXTENSIONS,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_bgr(h: int = 100, w: int = 200, val: int = 128) -> np.ndarray:
    return np.full((h, w, 3), val, dtype=np.uint8)


@pytest.fixture
def processor(tmp_path):
    return VideoProcessor(
        skip_frames=2,
        max_frames=0,
        generate_output_video=False,
        output_dir=str(tmp_path),
        vehicle_mode=VehicleMode.MOVING,
        enable_temporal_voting=False,
    )


@pytest.fixture
def stationary_processor(tmp_path):
    return VideoProcessor(
        skip_frames=5,
        generate_output_video=False,
        output_dir=str(tmp_path),
        vehicle_mode=VehicleMode.STATIONARY,
        enable_temporal_voting=False,
    )


# ============================================================================
# Inicialização e VehicleMode
# ============================================================================

class TestInitialization:
    def test_default_init(self, tmp_path):
        p = VideoProcessor(output_dir=str(tmp_path), enable_temporal_voting=False)
        assert p.skip_frames == 2  # clamp min 1
        assert p.max_frames == 0
        assert p.vehicle_mode == VehicleMode.MOVING

    def test_skip_frames_clamped_to_min_1(self, tmp_path):
        p = VideoProcessor(
            skip_frames=0, output_dir=str(tmp_path),
            enable_temporal_voting=False,
        )
        assert p.skip_frames == 1

    def test_stationary_mode_params_applied(self, tmp_path):
        p = VideoProcessor(
            output_dir=str(tmp_path),
            vehicle_mode=VehicleMode.STATIONARY,
            enable_temporal_voting=False,
        )
        assert p.confidence_early_stop > 0
        assert p.min_stable_detections > 0
        assert p.sharpness_filter is True

    def test_moving_mode_no_early_stop(self, tmp_path):
        p = VideoProcessor(
            output_dir=str(tmp_path),
            vehicle_mode=VehicleMode.MOVING,
            enable_temporal_voting=False,
        )
        assert p.confidence_early_stop == 0.0
        assert p.min_stable_detections == 0
        assert p.sharpness_filter is False

    def test_output_dir_created(self, tmp_path):
        new_dir = tmp_path / "newdir"
        VideoProcessor(
            output_dir=str(new_dir),
            enable_temporal_voting=False,
        )
        assert new_dir.exists()


class TestVehicleModeParams:
    def test_all_modes_have_required_keys(self):
        required = {'skip_frames', 'confidence_early_stop', 'min_stable_detections',
                    'sharpness_filter', 'sharpness_threshold', 'description'}
        for mode, params in VEHICLE_MODE_PARAMS.items():
            assert required.issubset(params.keys()), f"Mode {mode} missing keys"

    def test_stationary_params_more_conservative(self):
        stat = VEHICLE_MODE_PARAMS[VehicleMode.STATIONARY]
        mov = VEHICLE_MODE_PARAMS[VehicleMode.MOVING]
        # Stationary: mais skip + filtro de nitidez
        assert stat['skip_frames'] >= mov['skip_frames']
        assert stat['sharpness_filter'] is True
        assert mov['sharpness_filter'] is False


# ============================================================================
# Extensões suportadas
# ============================================================================

class TestSupportedExtensions:
    def test_common_formats_supported(self):
        exts = VideoProcessor.get_supported_extensions()
        for e in ('.mp4', '.avi', '.mov', '.mkv'):
            assert e in exts

    def test_uploader_extensions_no_dot(self):
        exts = VideoProcessor.get_extensions_for_uploader()
        assert all(not e.startswith('.') for e in exts)
        assert 'mp4' in exts

    def test_is_supported_positive(self):
        assert VideoProcessor.is_supported('video.mp4')
        assert VideoProcessor.is_supported('VIDEO.MP4')  # case-insensitive
        assert VideoProcessor.is_supported('/path/to/x.mkv')

    def test_is_supported_negative(self):
        assert not VideoProcessor.is_supported('document.pdf')
        assert not VideoProcessor.is_supported('image.jpg')
        assert not VideoProcessor.is_supported('noextension')


# ============================================================================
# Métodos estáticos / utilitários
# ============================================================================

class TestStaticUtilities:
    def test_normalize_plate_strips_non_alnum(self):
        assert VideoProcessor._normalize_plate('ABC-1234') == 'ABC1234'
        assert VideoProcessor._normalize_plate('abc 1d23') == 'ABC1D23'
        assert VideoProcessor._normalize_plate('!!!') == ''

    def test_normalize_plate_empty(self):
        assert VideoProcessor._normalize_plate('') == ''

    def test_format_duration_seconds_only(self):
        assert VideoProcessor._format_duration(45) == '00:45'

    def test_format_duration_minutes(self):
        assert VideoProcessor._format_duration(125) == '02:05'

    def test_format_duration_hours(self):
        assert VideoProcessor._format_duration(3661) == '01:01:01'

    def test_format_duration_negative(self):
        assert VideoProcessor._format_duration(-5) == '00:00'

    def test_compute_sharpness_returns_positive_float(self):
        # Imagem com textura (ruído)
        img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        sharpness = VideoProcessor._compute_sharpness(img)
        assert isinstance(sharpness, float)
        assert sharpness > 0

    def test_compute_sharpness_flat_image_near_zero(self):
        # Imagem lisa → baixa nitidez
        flat = _make_bgr(100, 200, val=128)
        sharpness = VideoProcessor._compute_sharpness(flat)
        assert sharpness < 1.0  # Praticamente zero

    def test_compute_sharpness_grayscale_accepted(self):
        gray = np.random.randint(0, 255, (100, 200), dtype=np.uint8)
        sharpness = VideoProcessor._compute_sharpness(gray)
        assert sharpness > 0


# ============================================================================
# _get_output_path
# ============================================================================

class TestGetOutputPath:
    def test_keeps_supported_extension(self, processor):
        out = processor._get_output_path('input.mp4')
        assert out.suffix == '.mp4'
        assert 'input_alpr_' in out.name

    def test_fallback_for_unsupported_codec(self, processor):
        # .dav usa mp4v, mas se alguém passar algo não em OUTPUT_CODECS
        out = processor._get_output_path('input.xyz')
        assert out.suffix == '.mp4'

    def test_output_in_configured_dir(self, processor):
        out = processor._get_output_path('input.mp4')
        assert out.parent == processor.output_dir


# ============================================================================
# _aggregate_plates — agregação de unique_plates
# ============================================================================

class TestAggregatePlates:
    def test_new_plate_added(self, processor):
        vr = VideoResult()
        fr = FrameResult(
            frame_number=1,
            timestamp_ms=100.0,
            plates_found=1,
            plate_texts=['ABC1D23'],
            confidences=[0.9],
            bboxes=[(10, 20, 100, 80)],
        )
        processor._aggregate_plates(vr, fr)
        assert 'ABC1D23' in vr.unique_plates
        entry = vr.unique_plates['ABC1D23']
        assert entry['total_detections'] == 1
        assert entry['best_confidence'] == pytest.approx(0.9)

    def test_same_plate_updates_count_and_best_conf(self, processor):
        vr = VideoResult()
        fr1 = FrameResult(
            frame_number=1, timestamp_ms=100.0,
            plates_found=1, plate_texts=['ABC1D23'],
            confidences=[0.7],
        )
        fr2 = FrameResult(
            frame_number=5, timestamp_ms=500.0,
            plates_found=1, plate_texts=['ABC1D23'],
            confidences=[0.95],
        )
        processor._aggregate_plates(vr, fr1)
        processor._aggregate_plates(vr, fr2)

        entry = vr.unique_plates['ABC1D23']
        assert entry['total_detections'] == 2
        assert entry['best_confidence'] == pytest.approx(0.95)
        assert entry['last_seen_frame'] == 5

    def test_lower_conf_does_not_override_best(self, processor):
        vr = VideoResult()
        fr1 = FrameResult(frame_number=1, timestamp_ms=0, plates_found=1,
                          plate_texts=['ABC1D23'], confidences=[0.9])
        fr2 = FrameResult(frame_number=2, timestamp_ms=0, plates_found=1,
                          plate_texts=['ABC1D23'], confidences=[0.5])
        processor._aggregate_plates(vr, fr1)
        processor._aggregate_plates(vr, fr2)
        assert vr.unique_plates['ABC1D23']['best_confidence'] == pytest.approx(0.9)

    def test_normalization_unifies_variants(self, processor):
        vr = VideoResult()
        fr1 = FrameResult(frame_number=1, timestamp_ms=0, plates_found=1,
                          plate_texts=['ABC-1D23'], confidences=[0.9])
        fr2 = FrameResult(frame_number=2, timestamp_ms=0, plates_found=1,
                          plate_texts=['abc1d23'], confidences=[0.8])
        processor._aggregate_plates(vr, fr1)
        processor._aggregate_plates(vr, fr2)
        # Normalizado para 'ABC1D23'
        assert len(vr.unique_plates) == 1
        assert 'ABC1D23' in vr.unique_plates
        assert vr.unique_plates['ABC1D23']['total_detections'] == 2

    def test_empty_plate_text_skipped(self, processor):
        vr = VideoResult()
        fr = FrameResult(frame_number=1, timestamp_ms=0, plates_found=1,
                         plate_texts=[''], confidences=[0.9])
        processor._aggregate_plates(vr, fr)
        assert len(vr.unique_plates) == 0


# ============================================================================
# _annotate_frame
# ============================================================================

class TestAnnotateFrame:
    def test_annotation_returns_image_same_shape(self, processor):
        img = _make_bgr(300, 400)
        result = processor._annotate_frame(img, (50, 50, 250, 150), 'ABC1D23', 0.9)
        assert result.shape == img.shape

    def test_high_conf_uses_green(self, processor):
        img = _make_bgr(300, 400)
        # Sanity check: não lança com conf alta
        processor._annotate_frame(img, (50, 50, 250, 150), 'ABC1D23', 0.95)

    def test_low_conf_does_not_raise(self, processor):
        img = _make_bgr(300, 400)
        processor._annotate_frame(img, (50, 50, 250, 150), 'ABC1D23', 0.3)

    def test_annotate_from_last_with_no_previous(self, processor):
        img = _make_bgr(300, 400)
        result = processor._annotate_frame_from_last(img, [])
        # Deve retornar frame sem modificação
        assert np.array_equal(result, img)

    def test_annotate_from_last_uses_last_result(self, processor):
        img = _make_bgr(300, 400)
        prev = [
            FrameResult(
                frame_number=1, timestamp_ms=0, plates_found=1,
                plate_texts=['ABC1D23'], confidences=[0.9],
                bboxes=[(50, 50, 250, 150)],
            )
        ]
        result = processor._annotate_frame_from_last(img, prev)
        # Deve ter modificado (bbox desenhado)
        assert not np.array_equal(result, img)


# ============================================================================
# build_confirmed_reading
# ============================================================================

class TestBuildConfirmedReading:
    def test_all_above_threshold_returns_full(self):
        info = {
            'plate_text': 'ABC1D23',
            'best_confidence': 0.9,
            'best_char_confidences': [(c, 0.95) for c in 'ABC1D23'],
        }
        assert VideoProcessor.build_confirmed_reading(info, threshold=0.6) == 'ABC1D23'

    def test_some_below_threshold_masked(self):
        info = {
            'plate_text': 'ABC1D23',
            'best_confidence': 0.7,
            'best_char_confidences': [
                ('A', 0.9), ('B', 0.9), ('C', 0.9),
                ('1', 0.3),  # baixo
                ('D', 0.9),
                ('2', 0.2),  # baixo
                ('3', 0.9),
            ],
        }
        result = VideoProcessor.build_confirmed_reading(info, threshold=0.5)
        assert result == 'ABC*D*3'

    def test_no_char_confidences_uses_best_confidence(self):
        # Sem char_confs, usa best_confidence para decidir tudo ou nada
        info_high = {'plate_text': 'ABC1234', 'best_confidence': 0.9,
                     'best_char_confidences': []}
        assert VideoProcessor.build_confirmed_reading(info_high, threshold=0.5) == 'ABC1234'

        info_low = {'plate_text': 'ABC1234', 'best_confidence': 0.3,
                    'best_char_confidences': []}
        assert VideoProcessor.build_confirmed_reading(info_low, threshold=0.5) == '*******'


# ============================================================================
# rank_unique_plates
# ============================================================================

class TestRankUniquePlates:
    def test_empty_dict(self):
        result = VideoProcessor.rank_unique_plates({}, max_plates=5)
        assert result == {}

    def test_ranks_by_composite_score(self):
        plates = {
            'LOW': {
                'total_detections': 1,
                'best_confidence': 0.5,
                'all_confidences': [0.5],
            },
            'HIGH': {
                'total_detections': 10,
                'best_confidence': 0.95,
                'all_confidences': [0.9, 0.9, 0.95],
            },
        }
        result = VideoProcessor.rank_unique_plates(plates, max_plates=2)
        # HIGH deve vir primeiro
        assert list(result.keys())[0] == 'HIGH'

    def test_max_plates_limits_output(self):
        plates = {
            f'PLATE{i}': {
                'total_detections': i + 1,
                'best_confidence': 0.5 + i * 0.05,
                'all_confidences': [0.5 + i * 0.05],
            }
            for i in range(10)
        }
        result = VideoProcessor.rank_unique_plates(plates, max_plates=3)
        assert len(result) == 3

    def test_composite_score_added(self):
        plates = {
            'ABC1234': {
                'total_detections': 5,
                'best_confidence': 0.8,
                'all_confidences': [0.8, 0.75, 0.82],
            }
        }
        result = VideoProcessor.rank_unique_plates(plates, max_plates=5)
        assert 'composite_score' in result['ABC1234']


# ============================================================================
# generate_timeline
# ============================================================================

class TestGenerateTimeline:
    def test_empty_video_result(self, processor):
        vr = VideoResult()
        assert processor.generate_timeline(vr) == []

    def test_only_frames_with_plates_included(self, processor):
        vr = VideoResult()
        vr.frame_results = [
            FrameResult(frame_number=1, timestamp_ms=100, plates_found=0),
            FrameResult(
                frame_number=5, timestamp_ms=500, plates_found=1,
                plate_texts=['ABC1D23'], confidences=[0.9],
            ),
            FrameResult(frame_number=10, timestamp_ms=1000, plates_found=0),
        ]
        timeline = processor.generate_timeline(vr)
        assert len(timeline) == 1
        assert timeline[0]['frame'] == 5
        assert timeline[0]['plates'] == ['ABC1D23']
        assert timeline[0]['max_confidence'] == pytest.approx(0.9)

    def test_timeline_has_formatted_time(self, processor):
        vr = VideoResult()
        vr.frame_results = [
            FrameResult(
                frame_number=30, timestamp_ms=65000, plates_found=1,
                plate_texts=['ABC1234'], confidences=[0.8],
            )
        ]
        timeline = processor.generate_timeline(vr)
        assert timeline[0]['time_s'] == pytest.approx(65.0)
        assert ':' in timeline[0]['time_formatted']


# ============================================================================
# VideoResult / FrameResult — propriedades
# ============================================================================

class TestVideoResultProperties:
    def test_avg_processing_time_zero_frames(self):
        vr = VideoResult()
        assert vr.avg_processing_time_per_frame == 0.0

    def test_avg_processing_time_computed(self):
        vr = VideoResult()
        vr.processed_frames = 10
        vr.total_processing_time_ms = 1000.0
        assert vr.avg_processing_time_per_frame == pytest.approx(100.0)

    def test_processing_fps_zero_time(self):
        vr = VideoResult()
        assert vr.processing_fps == 0.0

    def test_processing_fps_computed(self):
        vr = VideoResult()
        vr.processed_frames = 30
        vr.total_processing_time_ms = 1000.0  # 1 segundo
        assert vr.processing_fps == pytest.approx(30.0)


# ============================================================================
# get_video_info — com VideoCapture mockado
# ============================================================================

class TestGetVideoInfo:
    def test_invalid_video_raises(self, processor):
        with pytest.raises(ValueError, match="Não foi possível abrir"):
            processor.get_video_info('/invalid/does/not/exist.mp4')
