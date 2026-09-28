"""Vehicle boxes, frame counts and short-lived geometry tracking.

Plate detection is independent of vehicle classification. Tracking is a local
motion association, not cross-camera identity recognition or Re-ID.
"""

import math
from collections import Counter
from numbers import Integral


CATEGORIES = ('car', 'two_wheeler', 'bus', 'truck', 'bicycle', 'three_wheeler',
              'lcv', 'tempo', 'van', 'other')
LABELS = dict(car='Car', two_wheeler='Two wheeler', bus='Bus', truck='Truck',
              bicycle='Bicycle', three_wheeler='Three wheeler', lcv='Light commercial vehicle',
              tempo='Tempo Traveller', van='Van', other='Other vehicle')
COCO_CLASSES = {1: 'bicycle', 2: 'car', 3: 'two_wheeler', 5: 'bus', 7: 'truck'}
UNSUPPORTED_COCO = ['tempo', 'three_wheeler', 'lcv', 'van']
CLASSIFICATION_REVIEW_SCORE = .6


def category_for(class_id):
    if isinstance(class_id, Integral):
        return COCO_CLASSES.get(int(class_id), 'other')
    name = ''.join(character for character in str(class_id).lower() if character.isalnum())
    aliases = {
        'car': 'car', 'hatchback': 'car', 'sedan': 'car', 'suv': 'car', 'muv': 'car',
        'motorcycle': 'two_wheeler', 'motorbike': 'two_wheeler', 'bike': 'two_wheeler',
        'scooter': 'two_wheeler', 'moped': 'two_wheeler', 'twowheeler': 'two_wheeler',
        '2wheeler': 'two_wheeler', 'bus': 'bus', 'minibus': 'bus', 'buses': 'bus',
        'truck': 'truck', 'heavytruck': 'truck', 'bicycle': 'bicycle', 'cycle': 'bicycle',
        'threewheeler': 'three_wheeler', '3wheeler': 'three_wheeler',
        'autorickshaw': 'three_wheeler', 'rickshaw': 'three_wheeler', 'auto': 'three_wheeler',
        'erickshaw': 'three_wheeler', 'lcv': 'lcv', 'lightcommercialvehicle': 'lcv',
        'lightcommercial': 'lcv', 'cargotempo': 'lcv', 'tempo': 'tempo',
        'tempotraveller': 'tempo', 'tempotraveler': 'tempo', 'van': 'van', 'minivan': 'van',
        'other': 'other', 'others': 'other', 'othervehicle': 'other',
    }
    return aliases.get(name, 'other')


def _taxonomy(taxonomy):
    specialized = taxonomy == 'uvh'
    unsupported = [] if specialized else list(UNSUPPORTED_COCO)
    return dict(supported_categories=[category for category in CATEGORIES if category not in unsupported],
                unsupported_categories=unsupported,
                classification_note=(
                    'Categories are model predictions. LCV denotes a light commercial cargo vehicle; '
                    'Tempo Traveller denotes a passenger vehicle.' if specialized else
                    'The general vehicle model does not distinguish tempo, three wheeler, LCV or van; '
                    'those categories require a specialized model.'))


def _box_dict(box):
    return dict(zip(('x', 'y', 'width', 'height'), box))


def _classification_review(category, votes, maxima, counts, latest_category=None, latest_score=0.):
    """Expose model disagreement without guessing a replacement vehicle type."""
    total = sum(votes.values())
    candidates = [dict(category=name, label=LABELS[name], observations=counts[name],
                       max_confidence=maxima[name], evidence_weight=round(weight, 4),
                       vote_share=round(weight/total, 4) if total > 0 else 0.)
                  for name, weight in sorted(votes.items(), key=lambda item: -item[1])]
    reasons = []
    if maxima.get(category, 0.) < CLASSIFICATION_REVIEW_SCORE:
        reasons.append('low_category_score')
    if category == 'other':
        reasons.append('unknown_vehicle_category')
    credible_competitors = [candidate for candidate in candidates if candidate['category'] != category
                           and candidate['max_confidence'] >= CLASSIFICATION_REVIEW_SCORE
                           and candidate['vote_share'] >= .2]
    if credible_competitors:
        reasons.append('conflicting_category_predictions')
    if latest_category is not None and latest_category != category and latest_score >= CLASSIFICATION_REVIEW_SCORE:
        reasons.append('latest_category_disagrees')
    return dict(requires_review=bool(reasons), reasons=reasons, selected_category=category,
                candidates=candidates, review_score_threshold=CLASSIFICATION_REVIEW_SCORE,
                competing_vote_share_threshold=.2,
                note='Categories and scores are model predictions. Temporal vote shares and review thresholds '
                     'are heuristics, not calibrated classification accuracy.')


def _review_counts(items, available):
    reviews = [item for item in items if item['classification_review']['requires_review']]
    counts = Counter(item['category'] for item in reviews)
    return dict(classification_review_count=len(reviews) if available else None,
                classification_review_by_category={category: counts[category] if available else None
                                                   for category in CATEGORIES})


def _observations(objects, width, height):
    observations = []
    for item in objects:
        if len(item) < 6:
            continue
        x, y, w, h, confidence, class_id = item[:6]
        try:
            x, y, w, h, confidence = map(float, (x, y, w, h, confidence))
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (x, y, w, h, confidence)) or w <= 0 or h <= 0:
            continue
        x1, y1 = max(0, min(width, x)), max(0, min(height, y))
        x2, y2 = max(0, min(width, x + w)), max(0, min(height, y + h))
        if x2 <= x1 or y2 <= y1:
            continue
        box = [x1, y1, x2 - x1, y2 - y1]
        category = category_for(class_id)
        confidence = max(0, min(1, confidence))
        observations.append(dict(box=box, bounding_box=_box_dict(box), confidence=confidence,
                                 detection_confidence=confidence, classification_confidence=confidence,
                                 category=category, observed_category=category, label=LABELS[category], plate_ids=[],
                                 classification_review=_classification_review(category, {category: confidence},
                                                                              {category: confidence}, {category: 1})))
    return observations


def attach_plate_ids(detections, vehicles):
    """Associate each plate with one smallest containing vehicle, never filter it."""
    for vehicle in vehicles:
        vehicle['plate_ids'] = []
    for detection in detections:
        box = detection.get('tight_plate_box') or detection.get('bounding_box')
        if not box:
            continue
        cx, cy = box['x'] + box['width'] / 2, box['y'] + box['height'] / 2
        owners = [vehicle for vehicle in vehicles
                  if vehicle['bounding_box']['x'] <= cx <= vehicle['bounding_box']['x'] + vehicle['bounding_box']['width']
                  and vehicle['bounding_box']['y'] <= cy <= vehicle['bounding_box']['y'] + vehicle['bounding_box']['height']]
        owner = min(owners, key=lambda vehicle: vehicle['box'][2] * vehicle['box'][3]) if owners else None
        detection['vehicle_id'] = owner['id'] if owner else None
        if owner:
            owner['plate_ids'].append(detection['id'])


def attach_vehicles(detections, objects, image_width, image_height, available=True, taxonomy='coco'):
    vehicles = _observations(objects, image_width, image_height) if available else []
    for index, vehicle in enumerate(vehicles):
        vehicle['id'] = f'vehicle-{index + 1}'
    attach_plate_ids(detections, vehicles)
    counts = Counter(vehicle['category'] for vehicle in vehicles)
    summary = dict(available=bool(available), scope='frame', total_vehicles=len(vehicles) if available else None,
                   counts_by_category={category: counts[category] if available else None for category in CATEGORIES},
                   **_review_counts(vehicles, available),
                   **_taxonomy(taxonomy))
    return vehicles, summary


def _iou(a, b):
    intersection = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])) * max(
        0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return intersection / max(1e-9, a[2] * a[3] + b[2] * b[3] - intersection)


def _centre(box):
    return box[0] + box[2] / 2, box[1] + box[3] / 2


class VehicleTracker:
    """One-to-one observation matching with short constant-velocity prediction.

    Only boxes actually observed in the current frame are returned. Historical
    tracks are counted separately from the peak simultaneously visible count.
    """

    def __init__(self, fps=30.0, taxonomy='coco', max_gap_seconds=.75):
        self.fps = fps if math.isfinite(fps) and fps > 0 else 30.0
        self.taxonomy = taxonomy
        self.max_gap_frames = max(1, math.ceil(self.fps * max_gap_seconds))
        self._tracks = []
        self._active = {}
        self._last_frame = -1
        self._last_observations = []
        self.max_visible = 0

    def update(self, objects, frame_index, width, height):
        if frame_index < self._last_frame:
            raise ValueError('Vehicle frames must be processed in chronological order.')
        if frame_index == self._last_frame:
            return self._last_observations
        observations = _observations(objects, width, height)
        self._active = {track_id: track for track_id, track in self._active.items()
                        if frame_index - track['last_frame'] <= self.max_gap_frames}
        active = self._active.values()
        pairs = []
        for track in active:
            delta = frame_index - track['last_frame']
            previous = track['box']
            predicted = [previous[0] + track['velocity'][0] * delta,
                         previous[1] + track['velocity'][1] * delta, previous[2], previous[3]]
            pcx, pcy = _centre(predicted)
            for index, observation in enumerate(observations):
                box = observation['box']
                area_ratio = (box[2] * box[3]) / (previous[2] * previous[3])
                if not .3 <= area_ratio <= 3.3:
                    continue
                cx, cy = _centre(box)
                gate = max(30.0, .8 * max(previous[2], previous[3], box[2], box[3]))
                distance = math.hypot(cx - pcx, cy - pcy)
                overlap = _iou(predicted, box)
                if distance > gate or (overlap < .02 and distance > .6 * gate):
                    continue
                category_penalty = .08 if observation['category'] != track['category'] else 0
                score = .65 * overlap + .35 * (1 - distance / gate) - category_penalty
                pairs.append((score, track['track_id'], index, track))
        matched_tracks, matched_observations, assignments = set(), set(), {}
        for _, track_id, index, track in sorted(pairs, key=lambda pair: (-pair[0], pair[1], pair[2])):
            if track_id not in matched_tracks and index not in matched_observations:
                matched_tracks.add(track_id)
                matched_observations.add(index)
                assignments[index] = track
        for index, observation in enumerate(observations):
            track = assignments.get(index)
            if track is None:
                if len(self._tracks) >= 5000:
                    raise ValueError('Recording exceeds 5,000 vehicle tracks. Split it into shorter clips.')
                track = dict(track_id=len(self._tracks) + 1, box=list(observation['box']), velocity=[0., 0.],
                             first_frame=frame_index, last_frame=frame_index, observations=0,
                             confidence=0., category=observation['category'], votes=Counter(),
                             category_counts=Counter(), category_maxima=Counter())
                self._tracks.append(track)
                self._active[track['track_id']] = track
            else:
                delta = max(1, frame_index - track['last_frame'])
                old_centre, new_centre = _centre(track['box']), _centre(observation['box'])
                measured = [(new_centre[axis] - old_centre[axis]) / delta for axis in (0, 1)]
                track['velocity'] = [measured[axis] if track['observations'] < 2 else
                                     .7 * measured[axis] + .3 * track['velocity'][axis] for axis in (0, 1)]
            track['box'] = list(observation['box'])
            track['last_frame'] = frame_index
            track['observations'] += 1
            track['confidence'] = max(track['confidence'], observation['confidence'])
            track['votes'][observation['category']] += observation['confidence']
            track['category_counts'][observation['category']] += 1
            track['category_maxima'][observation['category']] = max(
                track['category_maxima'][observation['category']], observation['confidence'])
            track['latest_category'] = observation['category']
            track['latest_score'] = observation['confidence']
            track['category'] = max(track['votes'], key=lambda category: (track['votes'][category], category == track['category']))
            observation.update(id=f"vehicle-{track['track_id']}", track_id=track['track_id'],
                               category=track['category'], label=LABELS[track['category']],
                               classification_confidence=track['category_maxima'][track['category']],
                               classification_review=_classification_review(track['category'], track['votes'],
                                                                            track['category_maxima'], track['category_counts'],
                                                                            track['latest_category'], track['latest_score']))
            track['classification_review'] = observation['classification_review']
        self.max_visible = max(self.max_visible, len(observations))
        self._last_frame, self._last_observations = frame_index, observations
        return observations

    def results(self):
        return [dict(id=f"vehicle-{track['track_id']}", track_id=track['track_id'],
                     box=list(track['box']), bounding_box=_box_dict(track['box']),
                     category=track['category'], label=LABELS[track['category']],
                     confidence=track['category_maxima'][track['category']], detection_confidence=track['confidence'],
                     classification_confidence=track['category_maxima'][track['category']],
                     observed_category=track['latest_category'], classification_review=track['classification_review'],
                     first_frame=track['first_frame'], last_frame=track['last_frame'],
                     observations=track['observations'], confirmed=track['observations'] >= 3,
                     plate_ids=[]) for track in self._tracks]

    def summary(self, available=True):
        counts = Counter(track['category'] for track in self._tracks)
        confirmed = [track for track in self._tracks if track['observations'] >= 3]
        confirmed_counts = Counter(track['category'] for track in confirmed)
        note = ('Local geometry and motion tracks; occlusion or re-entry can create a new track. '
                'Counts are not cross-camera identities or line-crossing traffic flow.')
        return dict(available=bool(available), scope='recording',
                    total_vehicles=len(self._tracks) if available else None,
                    total_tracks=len(self._tracks) if available else None,
                    confirmed_tracks=len(confirmed) if available else None,
                    tentative_tracks=len(self._tracks) - len(confirmed) if available else None,
                    visible_now=len(self._last_observations) if available else None,
                    max_visible=self.max_visible if available else None,
                    unique_vehicle_tracks=len(self._tracks) if available else None,
                    confirmed_vehicle_tracks=len(confirmed) if available else None,
                    tentative_vehicle_tracks=len(self._tracks) - len(confirmed) if available else None,
                    max_visible_vehicles=self.max_visible if available else None,
                    counts_by_category={category: counts[category] if available else None for category in CATEGORIES},
                    confirmed_counts_by_category={category: confirmed_counts[category] if available else None for category in CATEGORIES},
                    **_review_counts(self._tracks, available),
                    counting_note=note, tracking_note=note,
                    **_taxonomy(self.taxonomy))
