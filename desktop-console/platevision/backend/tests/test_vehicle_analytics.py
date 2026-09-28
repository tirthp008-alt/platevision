import pytest

from app.services.vehicles import VehicleTracker, attach_vehicles, category_for


def test_indian_model_taxonomy_groups_classes_without_calling_every_van_a_tempo():
    names = ['Hatchback', 'Sedan', 'SUV', 'MUV', 'Bus', 'Truck', 'Three-wheeler',
             'Two-wheeler', 'LCV', 'Mini-bus', 'tempo-traveller', 'bicycle', 'Van', 'Others']
    assert [category_for(name) for name in names] == [
        'car', 'car', 'car', 'car', 'bus', 'truck', 'three_wheeler', 'two_wheeler',
        'lcv', 'bus', 'tempo', 'bicycle', 'van', 'other']
    objects = [(index * 20, 10, 18, 20, .8, name) for index, name in enumerate(names)]
    vehicles, summary = attach_vehicles([], objects, 300, 100, taxonomy='uvh')
    assert summary['total_vehicles'] == 14
    assert summary['counts_by_category']['car'] == 4
    assert summary['counts_by_category']['bus'] == 2
    assert summary['unsupported_categories'] == []
    assert [vehicle['label'] for vehicle in vehicles if vehicle['category'] == 'tempo'] == ['Tempo Traveller']


def test_coco_summary_reports_unsupported_categories_and_unavailable_as_unknown():
    _, summary = attach_vehicles([], [(0, 0, 30, 30, .9, 2)], 100, 100)
    assert summary['total_vehicles'] == 1
    assert summary['counts_by_category']['car'] == 1
    assert summary['unsupported_categories'] == ['tempo', 'three_wheeler', 'lcv', 'van']
    _, unavailable = attach_vehicles([], [], 100, 100, available=False)
    assert unavailable['total_vehicles'] is None
    assert all(value is None for value in unavailable['counts_by_category'].values())


def test_box_clipping_and_smallest_vehicle_association_does_not_drop_unmatched_plate():
    plates = [dict(id='p1', bounding_box=dict(x=0, y=0, width=20, height=20),
                   tight_plate_box=dict(x=30, y=30, width=10, height=10)),
              dict(id='p2', bounding_box=dict(x=150, y=150, width=10, height=10))]
    vehicles, summary = attach_vehicles(plates, [(-10, -20, 130, 140, .8, 2),
                                                (20, 20, 40, 40, .9, 3),
                                                (1000, 1000, 20, 20, .9, 7)], 200, 200)
    assert summary['total_vehicles'] == 2
    assert vehicles[0]['box'] == [0, 0, 120, 120]
    assert plates[0]['vehicle_id'] == 'vehicle-2'
    assert plates[1]['vehicle_id'] is None
    assert vehicles[1]['plate_ids'] == ['p1']
    assert len(plates) == 2


def test_motion_tracking_keeps_ids_through_crossing_and_one_to_one_association():
    tracker = VehicleTracker(fps=12)
    first = tracker.update([(10, 10, 30, 20, .9, 2), (110, 10, 30, 20, .8, 2)], 0, 200, 100)
    assert [item['track_id'] for item in first] == [1, 2]
    for frame in range(1, 6):
        current = tracker.update([(10 + frame * 12, 10, 30, 20, .9, 2),
                                  (110 - frame * 12, 10, 30, 20, .8, 2)], frame, 200, 100)
        assert [item['track_id'] for item in current] == [1, 2]
    summary = tracker.summary()
    assert summary['total_vehicles'] == summary['confirmed_vehicle_tracks'] == 2
    assert summary['max_visible_vehicles'] == 2


def test_expiry_reentry_and_disappearance_do_not_emit_ghost_boxes():
    tracker = VehicleTracker(fps=12)
    tracker.update([(10, 10, 30, 20, .9, 2)], 0, 200, 100)
    assert tracker.update([], 1, 200, 100) == []
    # Longer than .75 seconds: a local tracker cannot claim the same identity.
    current = tracker.update([(10, 10, 30, 20, .9, 2)], 12, 200, 100)
    assert current[0]['track_id'] == 2
    summary = tracker.summary()
    assert summary['unique_vehicle_tracks'] == 2
    assert summary['confirmed_vehicle_tracks'] == 0
    assert summary['max_visible_vehicles'] == 1


def test_repeated_frame_does_not_inflate_counts_and_category_vote_resists_one_weak_frame():
    tracker = VehicleTracker(fps=12, taxonomy='uvh')
    tracker.update([(10, 10, 60, 30, .95, 'LCV')], 0, 200, 100)
    tracker.update([(12, 10, 60, 30, .95, 'LCV')], 1, 200, 100)
    current = tracker.update([(14, 10, 60, 30, .3, 'tempo-traveller')], 2, 200, 100)
    assert current[0]['category'] == 'lcv'
    tracker.update([(14, 10, 60, 30, .9, 'LCV')], 2, 200, 100)
    assert tracker.results()[0]['observations'] == 3
    assert tracker.summary()['counts_by_category']['lcv'] == 1
    assert tracker.summary()['unsupported_categories'] == []
    with pytest.raises(ValueError):
        tracker.update([], 1, 200, 100)


def test_invalid_vehicle_geometry_cannot_inflate_analytics():
    vehicles, summary = attach_vehicles([], [
        (float('nan'), 0, 30, 30, .9, 2), (10, 10, -1, 20, .9, 2),
        (10, 10, 30, 20, .9, 2)], 200, 100)
    assert len(vehicles) == summary['total_vehicles'] == 1


def test_recording_summary_aliases_distinguish_visible_from_all_tracks():
    tracker = VehicleTracker(fps=12)
    for frame in range(3):
        tracker.update([(10, 10, 30, 20, .9, 2)], frame, 200, 100)
    tracker.update([], 3, 200, 100)
    summary = tracker.summary()
    assert summary['scope'] == 'recording'
    assert summary['total_tracks'] == summary['unique_vehicle_tracks'] == 1
    assert summary['confirmed_tracks'] == summary['confirmed_vehicle_tracks'] == 1
    assert summary['tentative_tracks'] == summary['tentative_vehicle_tracks'] == 0
    assert summary['visible_now'] == 0
    assert summary['max_visible'] == summary['max_visible_vehicles'] == 1
    assert summary['counting_note'] == summary['tracking_note']
    unknown = tracker.summary(available=False)
    assert all(unknown[key] is None for key in ('total_tracks', 'confirmed_tracks',
                                              'tentative_tracks', 'visible_now', 'max_visible'))


def test_vehicle_track_limit_bounds_recording_history():
    tracker = VehicleTracker(fps=1)
    for index in range(5000):
        tracker.update([(10, 10, 30, 20, .9, 2)], index * 2, 200, 100)
    assert tracker.summary()['total_tracks'] == 5000
    with pytest.raises(ValueError, match='5,000 vehicle tracks'):
        tracker.update([(10, 10, 30, 20, .9, 2)], 10000, 200, 100)
    assert len(tracker.results()) == 5000


def test_bus_track_never_inherits_a_car_predictions_higher_category_score():
    tracker = VehicleTracker(fps=12, taxonomy='uvh')
    tracker.update([(10, 10, 80, 60, .99, 'Sedan')], 0, 200, 100)
    for frame in range(1, 4):
        current = tracker.update([(10+frame, 10, 80, 60, .8, 'Bus')], frame, 200, 100)
    result = tracker.results()[0]
    assert result['category'] == 'bus' and result['confidence'] == .8
    assert result['classification_confidence'] == .8 and result['detection_confidence'] == .99
    assert current[0]['classification_confidence'] == .8
    review = result['classification_review']
    assert review['requires_review'] is True
    assert 'conflicting_category_predictions' in review['reasons']
    assert [candidate['category'] for candidate in review['candidates']] == ['bus', 'car']
    assert review['candidates'][0]['observations'] == 3 and review['candidates'][1]['observations'] == 1
    summary = tracker.summary()
    assert summary['total_tracks'] == summary['counts_by_category']['bus'] == 1
    assert summary['counts_by_category']['car'] == 0
    assert summary['classification_review_count'] == summary['classification_review_by_category']['bus'] == 1


def test_latest_car_prediction_disagrees_with_stable_bus_without_changing_total():
    tracker = VehicleTracker(fps=12, taxonomy='uvh')
    for frame in range(10):
        tracker.update([(10, 10, 80, 60, .9, 'Bus')], frame, 200, 100)
    observation = tracker.update([(10, 10, 80, 60, .98, 'SUV')], 10, 200, 100)[0]
    assert observation['category'] == 'bus' and observation['observed_category'] == 'car'
    assert observation['confidence'] == observation['detection_confidence'] == .98
    assert observation['classification_confidence'] == .9
    assert 'latest_category_disagrees' in observation['classification_review']['reasons']
    assert tracker.summary()['total_tracks'] == 1


def test_one_weak_type_flip_does_not_trigger_a_false_strong_classification_conflict():
    tracker = VehicleTracker(fps=12, taxonomy='uvh')
    for frame, name, score in [(0, 'Bus', .95), (1, 'Bus', .95), (2, 'Sedan', .3)]:
        tracker.update([(10+frame, 10, 80, 60, score, name)], frame, 200, 100)
    result = tracker.results()[0]
    assert result['category'] == 'bus' and result['confidence'] == .95
    assert result['classification_review']['requires_review'] is False
    assert tracker.summary()['classification_review_count'] == 0


def test_distinct_bus_and_car_stay_separate_when_confidence_order_changes():
    tracker = VehicleTracker(fps=12, taxonomy='uvh')
    first = tracker.update([(10, 10, 60, 60, .91, 'Bus'), (120, 10, 60, 60, .87, 'Sedan')], 0, 240, 100)
    second = tracker.update([(122, 10, 60, 60, .96, 'Sedan'), (12, 10, 60, 60, .83, 'Bus')], 1, 240, 100)
    assert [vehicle['track_id'] for vehicle in first] == [1, 2]
    assert [vehicle['track_id'] for vehicle in second] == [2, 1]
    assert tracker.summary()['counts_by_category']['bus'] == tracker.summary()['counts_by_category']['car'] == 1
    assert tracker.summary()['total_tracks'] == 2


def test_low_category_scores_are_reported_for_review_without_discarding_counts():
    vehicles, summary = attach_vehicles([], [(10, 10, 60, 60, .4, 'Bus'),
                                            (100, 10, 60, 60, .92, 'Sedan')], 240, 100, taxonomy='uvh')
    assert summary['total_vehicles'] == 2
    assert summary['counts_by_category']['bus'] == summary['counts_by_category']['car'] == 1
    assert summary['classification_review_count'] == summary['classification_review_by_category']['bus'] == 1
    assert vehicles[0]['classification_review']['reasons'] == ['low_category_score']
    assert vehicles[1]['classification_review']['requires_review'] is False
    _, unavailable = attach_vehicles([], [], 240, 100, available=False)
    assert unavailable['classification_review_count'] is None
    assert all(value is None for value in unavailable['classification_review_by_category'].values())
