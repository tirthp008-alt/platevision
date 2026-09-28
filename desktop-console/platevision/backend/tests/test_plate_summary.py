from copy import deepcopy

import pytest

from app.services.plate_summary import summarize_recognized_plates


def plate(identifier, text='GJ01AB1234', confidence=.9, first=0., last=1., **extra):
    return dict(id=identifier, text=text, ocr_confidence=confidence, format_status='valid',
                first_seen=first, last_seen=last, observations=5, **extra)


def test_exact_separate_sightings_group_using_best_crop_without_changing_tracks():
    tracks = [plate(1, confidence=.82), plate(2, confidence=.96, first=3., last=5.),
              plate(3, text='MH12CD4321', first=0., last=4.)]
    original = deepcopy(tracks)
    summary = summarize_recognized_plates(tracks)
    assert summary['unique_registrations'] == 2
    assert summary['recognized_track_count'] == 3 and summary['suppressed_repeat_count'] == 1
    group = summary['groups'][0]
    assert group['track_ids'] == [1, 2] and group['representative_track_id'] == 2
    assert group['ocr_confidence'] == .96 and group['observations'] == 10
    assert group['first_seen'] == 0. and group['last_seen'] == 5.
    assert group['status'] == 'recognized' and tracks == original


def test_simultaneous_repeated_text_stays_separate_including_later_sightings():
    summary = summarize_recognized_plates([plate(1), plate(2, first=.5, last=2.), plate(3, first=5., last=6.)])
    assert len(summary['groups']) == summary['ambiguous_track_count'] == 3
    assert summary['unique_registrations'] == summary['suppressed_repeat_count'] == 0
    assert all(group['status'] == 'ambiguous' and group['track_count'] == 1 for group in summary['groups'])
    assert all(group['ambiguity_reason'] == 'simultaneous_tracks_share_text' for group in summary['groups'])


def test_actual_frame_observations_detect_collisions_without_assuming_gap_visibility():
    tracks = [plate(1, first=0, last=10), plate(2, first=4, last=6)]
    summary = summarize_recognized_plates(tracks, frames_by_track={1: [0, 10], 2: [4, 6]})
    assert summary['unique_registrations'] == 1 and summary['groups'][0]['track_ids'] == [1, 2]
    collision = summarize_recognized_plates(tracks, frames_by_track={1: [0, 4, 10], 2: [4, 6]})
    assert collision['ambiguous_track_count'] == 2 and collision['unique_registrations'] == 0


@pytest.mark.parametrize('extra', [
    {'ocr_confidence': .79}, {'ocr_confidence': None}, {'ocr_confidence': float('nan')},
    {'ocr_confidence': 0., 'confidence': .99}, {'format_status': 'uncertain'},
    {'experimental_review_only': True}, {'ocr_review': {'requires_review': True}},
    {'ocr_review': {'conflicting_readings': True}}, {'text': 'BLUR'},
    {'raw_text': 'GJ01AB?1234'}, {'text': 'GJ01??1234'},
])
def test_unresolved_readings_never_merge_or_disappear(extra):
    first = plate(1); first.update(extra)
    second = plate(2, first=3., last=4.); second.update(extra)
    summary = summarize_recognized_plates([first, second])
    assert summary['groups'] == [] and summary['unique_registrations'] == 0
    assert summary['unresolved_track_ids'] == [1, 2] and summary['unresolved_track_count'] == 2


def test_frame_scope_preserves_each_same_text_detection_with_uuid_ids():
    detections = [dict(id=identifier, normalized_text='GJ01AB1234', ocr_confidence=.9, format_status='valid')
                  for identifier in ('photo-a', 'photo-b')]
    summary = summarize_recognized_plates(detections, scope='frame')
    assert summary['unique_registrations'] == 0 and summary['ambiguous_track_count'] == 2
    assert [group['representative_track_id'] for group in summary['groups']] == ['photo-a', 'photo-b']
    assert all(group['first_seen'] is None and group['observations'] == 1 for group in summary['groups'])


def test_ocr_off_disables_summary_without_losing_raw_track_references():
    summary = summarize_recognized_plates([plate(1), plate(2)], read_text=False)
    assert summary['enabled'] is False and summary['groups'] == []
    assert summary['unresolved_track_ids'] == [1, 2]


def test_normalization_allows_formatting_but_not_fuzzy_registration_merges():
    tracks = [plate(1, text='gj 01-ab 1234'), plate(2, text='GJ01AB1234', first=3., last=4.),
              plate(3, text='GJ01AB1235', first=6., last=7.)]
    summary = summarize_recognized_plates(tracks)
    assert [group['text'] for group in summary['groups']] == ['GJ01AB1234', 'GJ01AB1235']
    assert summary['groups'][0]['track_ids'] == [1, 2]


def test_unknown_recording_times_cannot_establish_distinct_sightings():
    tracks = [dict(id=identifier, text='GJ01AB1234', ocr_confidence=.9, format_status='valid') for identifier in (1, 2)]
    summary = summarize_recognized_plates(tracks)
    assert summary['ambiguous_track_count'] == 2 and summary['unique_registrations'] == 0
