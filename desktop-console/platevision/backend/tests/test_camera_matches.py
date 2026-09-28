"""Cross-camera association gates are tested without Re-ID weights or GPU."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from app.services.camera_matches import CameraMatchGallery


def vector(cosine=1.):
    value = np.zeros(2048, np.float32)
    value[0], value[1] = cosine, np.sqrt(max(0., 1-cosine*cosine))
    return value


def observation(camera='A', at=100., track='track-1', cosine=1., **extra):
    return dict(camera_id=camera, observed_at=at, track_id=track,
                descriptor=vector(cosine), category='car', **extra)


def gallery(**kwargs):
    value = CameraMatchGallery(**kwargs)
    value.set_routes([dict(source_camera='A', target_camera='B', min_seconds=10, max_seconds=60)])
    return value


def test_matching_requires_explicit_direction_and_travel_window():
    value = CameraMatchGallery()
    value.register_batch([observation()])
    assert value.register_batch([observation('B', 120.)])[0]['status'] == 'new'
    value.set_routes([dict(source_camera='A', target_camera='B', min_seconds=10, max_seconds=60)])
    assert value.register_batch([observation('A', 130., 'different-local-track')])[0]['status'] == 'new'
    too_soon = value.register_batch([observation('B', 105., 'early')])[0]
    too_late = value.register_batch([observation('B', 300., 'late')])[0]
    assert too_soon['status'] == too_late['status'] == 'new'
    assert too_soon['rejected_candidates']['chronological_or_travel'] > 0


def test_appearance_candidate_has_raw_evidence_and_no_identity_claim():
    value = gallery()
    source = value.register_batch([observation()])[0]
    result = value.register_batch([observation('B', 120., cosine=.9)])[0]
    assert result['status'] == 'candidate'
    assert result['requires_review'] is True
    candidate = result['selected_candidate']
    assert candidate['source_observation_id'] == source['observation_id']
    assert candidate['appearance_similarity'] == pytest.approx(.9)
    assert candidate['score'] == candidate['review_score'] == pytest.approx(.9)
    assert candidate['travel_seconds'] == 20.
    assert candidate['min_seconds'] == 10 and candidate['max_seconds'] == 60
    assert result['matches'] == result['candidates']
    assert 'verified' in result['note']


def test_interval_travel_uses_source_exit_and_target_entry():
    value = gallery()
    value.register_batch([observation(at=70., first_seen_at=40., last_seen_at=100.)])
    result = value.register_batch([observation('B', 150., first_seen_at=115., last_seen_at=170.)])[0]
    assert result['selected_candidate']['travel_seconds'] == 15.
    assert result['selected_candidate']['source_last_seen_at'] == 100.
    assert result['first_seen_at'] == 115.
    # Crop timestamps are chronologically ordered but track intervals overlap.
    overlap = value.register_batch([observation('B', 155., 'overlap', first_seen_at=90., last_seen_at=180.)])[0]
    assert overlap['status'] == 'new'


@pytest.mark.parametrize('elapsed,expected', [(9.99, 'new'), (10., 'candidate'),
                                             (60., 'candidate'), (60.01, 'new')])
def test_directed_travel_window_includes_both_endpoints(elapsed, expected):
    value = gallery()
    value.register_batch([observation()])
    result = value.register_batch([observation('B', 100.+elapsed)])[0]
    assert result['status'] == expected


@pytest.mark.parametrize('plate', [None, '', '??????????'])
def test_unreadable_plate_matches_three_minutes_later_without_plate_boost(plate):
    value = CameraMatchGallery()
    value.set_routes([dict(source_camera='A', target_camera='B', min_seconds=120, max_seconds=240)])
    value.register_batch([observation(plate_text=plate, plate_confidence=.95)])
    result = value.register_batch([observation('B', 280., cosine=.91,
                                              plate_text='MH12AB1234', plate_confidence=.95)])[0]
    assert result['status'] == 'candidate'
    candidate = result['selected_candidate']
    assert candidate['travel_seconds'] == 180.
    assert candidate['review_score'] == pytest.approx(.91)
    assert candidate['plate_evidence']['boost'] == 0.
    assert candidate['plate_evidence']['known_matching_characters'] == 0


@pytest.mark.parametrize('a,b,confidence,kind,boost', [
    ('MH12????', 'MH12AB34', .9, 'agreement', .04),
    ('MH1?????', 'MH12AB34', .9, 'insufficient_known_characters', 0.),
    ('????MH12', 'MH12AB34', .9, 'partial_conflict', 0.),
    ('MH12AB34', 'MH12AB34', .7, 'low_confidence', 0.),
    ('MH12AB34', 'MH12AB35', .7, 'low_confidence', 0.),
    ('MH12?B34', 'MH12AB34', .9, 'agreement', .04),
    ('mh-12 ?b34', 'MH12AB34', .9, 'agreement', .04),
])
def test_plate_boost_requires_four_positionally_matching_known_characters(a,b,confidence,kind,boost):
    value = gallery()
    value.register_batch([observation(plate_text=a, plate_confidence=confidence)])
    result = value.register_batch([observation('B', 120., cosine=.9, plate_text=b, plate_confidence=confidence)])[0]
    candidate = result['selected_candidate']
    assert candidate['plate_evidence']['kind'] == kind
    assert candidate['plate_evidence']['boost'] == boost
    assert candidate['score'] == pytest.approx(.9+boost)


def test_full_high_confidence_plate_conflict_vetoes_identical_appearance():
    value = gallery()
    value.register_batch([observation(plate_text='MH12AB1234', plate_confidence=.95)])
    result = value.register_batch([observation('B', 120., plate_text='MH12AB1235', plate_confidence=.95)])[0]
    assert result['status'] == 'new' and result['selected_candidate'] is None
    assert result['rejected_candidates']['conflicting_full_plate'] == 1


def test_plate_agreement_cannot_bypass_appearance_threshold():
    value = gallery()
    value.register_batch([observation(plate_text='MH12AB1234', plate_confidence=1.)])
    result = value.register_batch([observation('B', 120., cosine=.79, plate_text='MH12AB1234', plate_confidence=1.)])[0]
    assert result['status'] == 'new'
    assert result['rejected_candidates']['below_appearance_threshold'] == 1


def test_impossible_travel_does_not_make_a_plausible_appearance_match_ambiguous():
    value = gallery()
    value.register_batch([observation(track='plausible', cosine=.91),
                          observation(at=115., track='too-close', cosine=.99)])
    result = value.register_batch([observation('B', 120.)])[0]
    assert result['status'] == 'candidate'
    assert result['selected_candidate']['source_track_id'] == 'plausible'
    assert result['rejected_candidates']['chronological_or_travel'] == 1


def test_category_disagreement_does_not_discard_an_appearance_candidate():
    value = gallery()
    value.register_batch([observation()])
    incoming = observation('B', 120., cosine=.9)
    incoming['category'] = 'bus'
    result = value.register_batch([incoming])[0]
    assert result['status'] == 'candidate'
    assert result['selected_candidate']['category_agreement'] is False


def test_close_top_two_sources_require_review_without_assignment():
    value = gallery()
    value.register_batch([observation(track='one'), observation(track='two', cosine=.99)])
    result = value.register_batch([observation('B', 120.)])[0]
    assert result['status'] == 'ambiguous'
    assert len(result['candidates']) == 2
    assert result['selected_candidate'] is None


def test_batch_sources_are_assigned_once_and_stronger_proposal_wins():
    value = gallery()
    value.register_batch([observation()])
    results = value.register_batch([observation('B', 120., 'weaker', cosine=.9),
                                    observation('B', 120., 'stronger', cosine=.97)])
    assert [result['status'] for result in results] == ['ambiguous', 'candidate']
    assert results[0]['reason'] == 'source_already_assigned_in_batch'
    assert results[0]['selected_candidate'] is None
    assert results[1]['selected_candidate']['appearance_similarity'] == pytest.approx(.97)


def test_batch_observations_cannot_become_each_others_sources():
    value = gallery()
    results = value.register_batch([observation(), observation('B', 120.)])
    assert [result['status'] for result in results] == ['new', 'new']


def test_out_of_order_utc_arrival_cannot_match_future_observation():
    value = gallery()
    value.register_batch([observation(at=150.)])
    result = value.register_batch([observation('B', 120.)])[0]
    assert result['status'] == 'new'
    assert result['rejected_candidates']['chronological_or_travel'] == 1
    stale = value.register_batch([observation(at=100.)])[0]
    assert stale['recorded'] is False
    assert stale['reason'] == 'out_of_order_local_track_observation'


def test_historical_uploads_expire_by_monotonic_ingestion_and_clear_removes_matches():
    now = [5000.]
    value = gallery(ttl_seconds=20, clock=lambda:now[0])
    value.register_batch([observation(at=100.)])
    now[0] += 5
    assert value.register_batch([observation('B', 120.)])[0]['status'] == 'candidate'
    now[0] += 16
    assert value.stats()['observations'] == 1
    assert len(value.recent_matches()) == 1
    assert value.register_batch([observation('B', 130., 'new')])[0]['status'] == 'new'
    value.clear()
    assert value.stats()['observations'] == 0
    assert value.recent_matches() == []
    assert len(value.get_routes()) == 1


def test_bounded_gallery_evicts_oldest_ingested_observation():
    value = gallery(max_entries=2)
    value.register_batch([observation(track='oldest')])
    value.register_batch([observation('X', 101., 'second')])
    value.register_batch([observation('Y', 102., 'third')])
    assert value.stats()['observations'] == 2
    assert value.register_batch([observation('B', 120.)])[0]['status'] == 'new'


def test_route_replacement_is_atomic_and_has_no_implicit_reverse():
    value = gallery()
    before = value.get_routes()
    with pytest.raises(ValueError):
        value.set_routes([dict(source_camera='A', target_camera='B', min_seconds=60, max_seconds=10)])
    assert value.get_routes() == before
    value.register_batch([observation('B', 100.)])
    assert value.register_batch([observation('A', 120.)])[0]['status'] == 'new'
    value.set_routes([])
    assert value.get_routes() == [] and value.recent_matches() == []


@pytest.mark.parametrize('field,bad', [('descriptor', np.zeros(2048)), ('descriptor', np.zeros(512)),
                                      ('descriptor', np.full(2048, np.nan)), ('observed_at', float('inf')),
                                      ('camera_id', ''), ('plate_confidence', 2.), ('first_seen_at', 101.),
                                      ('plate_confidence', False), ('plate_confidence', ''),
                                      ('plate_confidence', []), ('plate_confidence', {})])
def test_invalid_batches_are_rejected_without_partial_mutation(field, bad):
    value = gallery()
    invalid = observation(track='invalid')
    invalid[field] = bad
    with pytest.raises(ValueError):
        value.register_batch([observation(), invalid])
    assert value.stats()['observations'] == 0
    assert value.recent_matches() == []


@pytest.mark.parametrize('confidence', [None, 0.])
def test_absent_or_zero_plate_confidence_does_not_veto_a_match(confidence):
    value = gallery()
    value.register_batch([observation(plate_text='MH12AB1234', plate_confidence=confidence)])
    result = value.register_batch([observation('B', 120., plate_text='MH12AB1235', plate_confidence=.95)])[0]
    assert result['status'] == 'candidate'
    assert result['selected_candidate']['plate_evidence']['kind'] == 'low_confidence'


def test_thread_safe_registration_and_public_results_do_not_expose_descriptors():
    value = gallery()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda index:value.register_batch([observation(track=str(index))]), range(20)))
    assert value.stats()['observations'] == 20
    assert len({result[0]['observation_id'] for result in results}) == 20
    public = value.recent_matches()
    assert all('descriptor' not in item for item in public)
    public[0]['status'] = 'mutated outside lock'
    assert value.recent_matches()[0]['status'] == 'new'
