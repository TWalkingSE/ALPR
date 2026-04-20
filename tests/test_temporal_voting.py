# tests/test_temporal_voting.py
"""
Testes para o TemporalVotingEngine (src/temporal_voting.py).

Cobre:
  - Inicialização, reset, get_status
  - _compute_iou / _text_similarity (estáticos)
  - add_observation — associação por IoU e criação de tracks
  - Estratégias de votação: positional / majority / hybrid
  - get_consolidated_results com diferentes strategies
  - max_tracks + decay
  - TemporalPlateTrack — propriedades
"""

from __future__ import annotations

import pytest

from src.temporal_voting import (
    TemporalVotingEngine,
    TemporalPlateTrack,
)


# ============================================================================
# Helpers
# ============================================================================

def _add_obs(engine, frame, text, conf=0.8, bbox=(100, 100, 200, 150),
             char_confs=None):
    engine.add_observation(
        frame_number=frame,
        plate_text=text,
        confidence=conf,
        bbox=bbox,
        char_confidences=char_confs,
    )


# ============================================================================
# Inicialização e status
# ============================================================================

class TestInitialization:
    def test_default_init(self):
        e = TemporalVotingEngine()
        assert e.enabled is True
        assert e.iou_threshold == 0.4
        assert e.min_observations == 2
        assert e.strategy == 'hybrid'
        assert e._tracks == []

    def test_custom_init(self):
        e = TemporalVotingEngine(
            enabled=False, iou_threshold=0.6,
            min_observations=3, strategy='positional',
        )
        assert e.enabled is False
        assert e.iou_threshold == 0.6
        assert e.min_observations == 3
        assert e.strategy == 'positional'

    def test_reset_clears_tracks(self):
        e = TemporalVotingEngine()
        _add_obs(e, 1, 'ABC1D23')
        assert len(e._tracks) > 0
        e.reset()
        assert e._tracks == []
        assert e._next_track_id == 0

    def test_status_reports_tracks(self):
        e = TemporalVotingEngine()
        _add_obs(e, 1, 'ABC1D23')
        _add_obs(e, 2, 'ABC1D23')
        status = e.get_status()
        assert status['enabled'] is True
        assert status['active_tracks'] >= 1
        assert status['total_observations'] >= 2


# ============================================================================
# _compute_iou — método estático
# ============================================================================

class TestComputeIoU:
    def test_identical_boxes_iou_1(self):
        iou = TemporalVotingEngine._compute_iou((10, 10, 100, 100), (10, 10, 100, 100))
        assert iou == pytest.approx(1.0)

    def test_non_overlapping_boxes_iou_0(self):
        iou = TemporalVotingEngine._compute_iou((0, 0, 50, 50), (100, 100, 200, 200))
        assert iou == pytest.approx(0.0)

    def test_partial_overlap(self):
        # Boxes 100x100 com overlap 50x50
        iou = TemporalVotingEngine._compute_iou((0, 0, 100, 100), (50, 50, 150, 150))
        # intersection = 50*50 = 2500, union = 2*10000 - 2500 = 17500
        assert iou == pytest.approx(2500 / 17500, rel=1e-3)

    def test_containing_box(self):
        # Box pequena dentro de grande
        iou = TemporalVotingEngine._compute_iou((25, 25, 75, 75), (0, 0, 100, 100))
        # intersection = 2500, union = 10000
        assert iou == pytest.approx(0.25)


# ============================================================================
# _text_similarity — método estático
# ============================================================================

class TestTextSimilarity:
    def test_identical_texts_similarity_1(self):
        assert TemporalVotingEngine._text_similarity('ABC1D23', 'ABC1D23') == 1.0

    def test_completely_different(self):
        # 7 chars totalmente diferentes
        assert TemporalVotingEngine._text_similarity('ABC1234', 'XYZ9876') == 0.0

    def test_partial_match(self):
        # 4/7 matches
        sim = TemporalVotingEngine._text_similarity('ABC1234', 'ABC9876')
        assert sim == pytest.approx(3 / 7, rel=1e-3)

    def test_empty_returns_0(self):
        assert TemporalVotingEngine._text_similarity('', 'ABC') == 0.0
        assert TemporalVotingEngine._text_similarity('ABC', '') == 0.0

    def test_different_lengths(self):
        # 'ABC' (3) vs 'ABC1D23' (7): 3 matches of 7 max
        sim = TemporalVotingEngine._text_similarity('ABC', 'ABC1D23')
        assert sim == pytest.approx(3 / 7, rel=1e-3)


# ============================================================================
# add_observation — associação por IoU e criação de tracks
# ============================================================================

class TestAddObservation:
    def test_disabled_engine_noop(self):
        e = TemporalVotingEngine(enabled=False)
        _add_obs(e, 1, 'ABC1D23')
        assert len(e._tracks) == 0

    def test_short_text_ignored(self):
        e = TemporalVotingEngine()
        _add_obs(e, 1, 'ABC')  # < 5 chars
        assert len(e._tracks) == 0

    def test_first_observation_creates_track(self):
        e = TemporalVotingEngine()
        _add_obs(e, 1, 'ABC1D23', conf=0.9)
        assert len(e._tracks) == 1
        assert e._tracks[0].best_text == 'ABC1D23'
        assert e._tracks[0].total_observations == 1

    def test_overlapping_bbox_merges_to_track(self):
        e = TemporalVotingEngine(iou_threshold=0.3)
        _add_obs(e, 1, 'ABC1D23', bbox=(100, 100, 200, 150))
        _add_obs(e, 2, 'ABC1D23', bbox=(105, 102, 205, 152))  # quase idêntico
        # Mesma track
        assert len(e._tracks) == 1
        assert e._tracks[0].total_observations == 2

    def test_non_overlapping_creates_new_track(self):
        e = TemporalVotingEngine(iou_threshold=0.3)
        _add_obs(e, 1, 'ABC1D23', bbox=(100, 100, 200, 150))
        _add_obs(e, 2, 'XYZ9P87', bbox=(500, 500, 600, 550))  # distante
        # Duas tracks
        assert len(e._tracks) == 2

    def test_best_confidence_updated(self):
        e = TemporalVotingEngine()
        _add_obs(e, 1, 'ABC1D23', conf=0.5)
        _add_obs(e, 2, 'ABC1D23', conf=0.9)
        assert e._tracks[0].best_confidence == pytest.approx(0.9)

    def test_max_tracks_enforces_limit(self):
        e = TemporalVotingEngine(max_tracks=3, iou_threshold=0.3)
        # 4 tracks distintas (bboxes bem separados)
        for i in range(4):
            _add_obs(
                e, frame=i, text=f'ABC{i}D23',
                bbox=(i * 500, 0, i * 500 + 100, 100),
            )
        assert len(e._tracks) <= 3

    def test_text_hyphen_stripped_and_uppercased(self):
        e = TemporalVotingEngine()
        _add_obs(e, 1, 'abc-1d23')
        assert e._tracks[0].best_text == 'ABC1D23'


# ============================================================================
# Votação posicional
# ============================================================================

class TestVotePositional:
    def test_consistent_readings_converge(self):
        e = TemporalVotingEngine(strategy='positional', min_observations=2)
        for f in range(3):
            _add_obs(e, f, 'ABC1D23', conf=0.9)
        results = e.get_consolidated_results()
        assert len(results) == 1
        assert results[0]['text'] == 'ABC1D23'
        assert results[0]['voting_applied'] is True

    def test_majority_position_wins(self):
        e = TemporalVotingEngine(strategy='positional', min_observations=2)
        # 4 leituras: 3x 'ABC1D23', 1x 'ABC1D24' (última posição diverge)
        _add_obs(e, 1, 'ABC1D23', conf=0.9)
        _add_obs(e, 2, 'ABC1D23', conf=0.9)
        _add_obs(e, 3, 'ABC1D23', conf=0.9)
        _add_obs(e, 4, 'ABC1D24', conf=0.5)
        results = e.get_consolidated_results()
        assert results[0]['text'] == 'ABC1D23'


# ============================================================================
# Votação por maioria
# ============================================================================

class TestVoteMajority:
    def test_most_frequent_wins(self):
        e = TemporalVotingEngine(strategy='majority', min_observations=2)
        _add_obs(e, 1, 'ABC1D23', conf=0.85)
        _add_obs(e, 2, 'ABC1D23', conf=0.85)
        _add_obs(e, 3, 'XYZ9P87', conf=0.90)  # uma leitura diferente
        results = e.get_consolidated_results()
        # ABC1D23 deve vencer (2 vs 1)
        top = [r for r in results if r['text'] == 'ABC1D23']
        assert top


# ============================================================================
# Votação híbrida
# ============================================================================

class TestVoteHybrid:
    def test_clear_majority_uses_majority(self):
        e = TemporalVotingEngine(strategy='hybrid', min_observations=2)
        for _ in range(5):
            _add_obs(e, frame=len(e._tracks) + 1, text='ABC1D23', conf=0.9)
        results = e.get_consolidated_results()
        assert results[0]['text'] == 'ABC1D23'

    def test_invalid_format_falls_back_to_best(self):
        e = TemporalVotingEngine(strategy='hybrid', min_observations=2)
        # Duas leituras de mesmo track com formato inválido (strings de 7 chars)
        _add_obs(e, 1, '1234567', conf=0.5)  # 7 dígitos — inválido
        _add_obs(e, 2, '1234567', conf=0.7)
        results = e.get_consolidated_results()
        # Deve cair no fallback (best_text)
        assert results[0]['text'] == '1234567'


# ============================================================================
# get_consolidated_results
# ============================================================================

class TestConsolidatedResults:
    def test_insufficient_obs_no_voting(self):
        e = TemporalVotingEngine(min_observations=3)
        _add_obs(e, 1, 'ABC1D23')
        results = e.get_consolidated_results()
        assert len(results) == 1
        assert results[0]['voting_applied'] is False

    def test_results_sorted_by_confidence(self):
        e = TemporalVotingEngine(min_observations=2, iou_threshold=0.3)
        # Track A: alta confiança
        _add_obs(e, 1, 'ABC1D23', conf=0.95, bbox=(0, 0, 100, 100))
        _add_obs(e, 2, 'ABC1D23', conf=0.93, bbox=(0, 0, 100, 100))
        # Track B: baixa confiança, bbox distante
        _add_obs(e, 3, 'XYZ9P87', conf=0.4, bbox=(500, 500, 600, 600))
        _add_obs(e, 4, 'XYZ9P87', conf=0.4, bbox=(500, 500, 600, 600))
        results = e.get_consolidated_results()
        # Ordenado desc
        for i in range(len(results) - 1):
            assert results[i]['confidence'] >= results[i + 1]['confidence']

    def test_empty_no_results(self):
        e = TemporalVotingEngine()
        assert e.get_consolidated_results() == []


# ============================================================================
# get_track_for_plate
# ============================================================================

class TestGetTrackForPlate:
    def test_finds_existing_track(self):
        e = TemporalVotingEngine()
        _add_obs(e, 1, 'ABC1D23')
        track = e.get_track_for_plate('ABC1D23')
        assert track is not None
        assert track.best_text == 'ABC1D23'

    def test_handles_hyphen_in_query(self):
        e = TemporalVotingEngine()
        _add_obs(e, 1, 'ABC1D23')
        track = e.get_track_for_plate('ABC-1D23')
        assert track is not None

    def test_not_found_returns_none(self):
        e = TemporalVotingEngine()
        _add_obs(e, 1, 'ABC1D23')
        assert e.get_track_for_plate('XYZ9876') is None


# ============================================================================
# TemporalPlateTrack — propriedades
# ============================================================================

class TestTemporalPlateTrack:
    def test_has_enough_observations_false_for_one(self):
        track = TemporalPlateTrack(track_id=0)
        track.observations = [{'text': 'ABC1D23'}]
        assert track.has_enough_observations is False

    def test_has_enough_observations_true_for_two(self):
        track = TemporalPlateTrack(track_id=0)
        track.observations = [{'text': 'ABC1D23'}, {'text': 'ABC1D23'}]
        assert track.has_enough_observations is True

    def test_duration_frames(self):
        track = TemporalPlateTrack(track_id=0)
        track.observations = [{'text': 'x'}]
        track.first_frame = 5
        track.last_frame = 20
        assert track.duration_frames == 16

    def test_duration_frames_no_obs(self):
        track = TemporalPlateTrack(track_id=0)
        assert track.duration_frames == 0
