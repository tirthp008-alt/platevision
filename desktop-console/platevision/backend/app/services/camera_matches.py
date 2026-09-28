"""Bounded, in-memory cross-camera review candidates, not identity decisions.

Appearance descriptors are matched only along explicitly configured directed
routes. UTC observation times gate travel; monotonic ingestion times govern
expiry, so historical uploaded recordings do not expire immediately.
"""
from collections import OrderedDict, deque
from copy import deepcopy
from dataclasses import dataclass
import math
from threading import RLock
import time
import uuid

import numpy as np


DESCRIPTOR_DIMENSION = 2048


def _number(value, name, minimum=None, maximum=None):
    if isinstance(value, (bool, str)):
        raise ValueError(f'{name} must be a finite number.')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{name} must be a finite number.') from error
    if not math.isfinite(value) or (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError(f'{name} is outside its supported range.')
    return value


def _identifier(value, name, maximum=240):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f'{name} must be a nonempty string of at most {maximum} characters.')
    if any(ord(character) < 32 for character in value):
        raise ValueError(f'{name} cannot contain control characters.')
    return value.strip()


def _plate(value):
    if value is None or value == '':
        return None
    if not isinstance(value, str):
        raise ValueError('plate_text must be a string or null.')
    normalized = value.upper().replace(' ', '').replace('-', '')
    if not 1 <= len(normalized) <= 20 or any(character not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789?' for character in normalized):
        raise ValueError('plate_text must contain only letters, digits and positional ? wildcards.')
    return normalized


def _plate_evidence(source, target):
    a, b = source.plate_text, target.plate_text
    evidence = dict(kind='none', known_matching_characters=0, known_conflicting_characters=0,
                    strong_confidence=False, boost=0., full_plate_conflict=False)
    if not a or not b:
        return evidence
    strong = min(source.plate_confidence, target.plate_confidence) >= .8
    evidence['strong_confidence'] = strong
    if len(a) == len(b):
        known = [(x, y) for x, y in zip(a, b) if x != '?' and y != '?']
        evidence['known_matching_characters'] = sum(x == y for x, y in known)
        evidence['known_conflicting_characters'] = sum(x != y for x, y in known)
    if not strong:
        evidence['kind'] = 'low_confidence'
    elif '?' not in a and '?' not in b and a != b:
        evidence.update(kind='full_plate_conflict', full_plate_conflict=True)
    elif (len(a) == len(b) and evidence['known_matching_characters'] >= 4
          and evidence['known_conflicting_characters'] == 0):
        evidence.update(kind='agreement', boost=.04)
    elif evidence['known_conflicting_characters']:
        evidence['kind'] = 'partial_conflict'
    else:
        evidence['kind'] = 'insufficient_known_characters'
    return evidence


@dataclass
class _Observation:
    observation_id: str
    camera_id: str
    observed_at: float
    first_seen_at: float
    last_seen_at: float
    track_id: str
    descriptor: np.ndarray
    category: str
    plate_text: str | None
    plate_confidence: float
    ingested_at: float = 0.

    @property
    def key(self):
        return self.camera_id, self.track_id


class CameraMatchGallery:
    """Thread-safe prototype with explicit route gates and conservative review.

    Each batch matches the gallery as it existed before the batch. Entries in
    one batch cannot become each other's evidence. Source assignments are
    one-to-one within a batch; no permanent/global identity is established.
    """

    def __init__(self, max_entries=2000, ttl_seconds=21600,
                 appearance_threshold=.82, ambiguity_margin=.03, *, clock=None):
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or not 1 <= max_entries <= 2000:
            raise ValueError('max_entries must be between 1 and 2000.')
        self.max_entries = max_entries
        self.ttl_seconds = _number(ttl_seconds, 'ttl_seconds', 0.001, 21600)
        self.appearance_threshold = _number(appearance_threshold, 'appearance_threshold', -1, 1)
        self.ambiguity_margin = _number(ambiguity_margin, 'ambiguity_margin', 0, 1)
        self._clock = clock or time.monotonic
        self._lock = RLock()
        self._entries = OrderedDict()
        self._routes = {}
        self._recent = deque(maxlen=max_entries)

    def set_routes(self, routes):
        """Replace all routes atomically; reverse routes must be supplied too."""
        if not isinstance(routes, (list, tuple)) or len(routes) > 1000:
            raise ValueError('routes must be a list of at most 1000 directed camera routes.')
        validated = {}
        for item in routes:
            if not isinstance(item, dict):
                raise ValueError('Each route must be a mapping.')
            source = _identifier(item.get('source_camera'), 'source_camera', 100)
            target = _identifier(item.get('target_camera'), 'target_camera', 100)
            if source == target:
                raise ValueError('A cross-camera route needs two different cameras.')
            minimum = _number(item.get('min_seconds'), 'min_seconds', 0)
            maximum = _number(item.get('max_seconds'), 'max_seconds', minimum)
            if (source, target) in validated:
                raise ValueError('Duplicate directed camera route.')
            validated[(source, target)] = dict(source_camera=source, target_camera=target,
                                              min_seconds=minimum, max_seconds=maximum)
        with self._lock:
            self._routes = validated
            # Previous review candidates used previous gates; do not display
            # stale associations as if they used the new route configuration.
            self._recent.clear()
        return self.get_routes()

    def get_routes(self):
        with self._lock:
            return deepcopy(list(self._routes.values()))

    def clear(self):
        """Erase descriptors, retained plate strings and review results."""
        with self._lock:
            removed = len(self._entries)
            self._entries.clear()
            self._recent.clear()
        return dict(cleared=removed, retained=0)

    def _expire(self, now):
        expired = [key for key, entry in self._entries.items() if now-entry.ingested_at >= self.ttl_seconds]
        for key in expired:
            del self._entries[key]
        while self._recent and now-self._recent[0][0] >= self.ttl_seconds:
            self._recent.popleft()

    def recent_matches(self, limit=50):
        if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 2000:
            raise ValueError('limit must be between 0 and 2000.')
        with self._lock:
            self._expire(self._clock())
            return deepcopy([item[1] for item in list(self._recent)[-limit:]][::-1]) if limit else []

    def stats(self):
        with self._lock:
            self._expire(self._clock())
            return dict(observations=len(self._entries), routes=len(self._routes),
                        max_entries=self.max_entries, ttl_seconds=self.ttl_seconds,
                        appearance_threshold=self.appearance_threshold, ambiguity_margin=self.ambiguity_margin,
                        descriptor_dimension=DESCRIPTOR_DIMENSION, storage='local_memory',
                        note='Review candidates, not verified identities. Appearance scores are uncalibrated; VeRi mainly covers four-wheeled vehicles.')

    @staticmethod
    def _validate(item):
        if not isinstance(item, dict):
            raise ValueError('Each observation must be a mapping.')
        try:
            descriptor = np.array(item.get('descriptor'), dtype=np.float32, copy=True)
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError('descriptor must be a finite 2048-element vector.') from error
        if descriptor.shape != (DESCRIPTOR_DIMENSION,) or not np.isfinite(descriptor).all():
            raise ValueError('descriptor must be a finite 2048-element vector.')
        norm = float(np.linalg.norm(descriptor.astype(np.float64)))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError('descriptor must have a nonzero finite norm.')
        descriptor = (descriptor.astype(np.float64) / norm).astype(np.float32)
        observed_at = _number(item.get('observed_at'), 'observed_at', 0)
        first_seen_at = _number(item.get('first_seen_at', observed_at), 'first_seen_at', 0)
        last_seen_at = _number(item.get('last_seen_at', observed_at), 'last_seen_at', first_seen_at)
        if not first_seen_at <= observed_at <= last_seen_at:
            raise ValueError('observed_at must fall inside first_seen_at and last_seen_at.')
        plate_confidence = item.get('plate_confidence')
        if plate_confidence is None:
            plate_confidence = 0.
        return _Observation(
            observation_id=uuid.uuid4().hex,
            camera_id=_identifier(item.get('camera_id'), 'camera_id', 100),
            observed_at=observed_at, first_seen_at=first_seen_at, last_seen_at=last_seen_at,
            track_id=_identifier(item.get('track_id'), 'track_id'), descriptor=descriptor,
            category=_identifier(item.get('category') or 'unknown', 'category', 80),
            plate_text=_plate(item.get('plate_text')),
            plate_confidence=_number(plate_confidence, 'plate_confidence', 0, 1))

    def _candidates(self, target):
        candidates = []
        rejected = dict(no_route=0, chronological_or_travel=0, below_appearance_threshold=0, conflicting_full_plate=0)
        for source in self._entries.values():
            if source.camera_id == target.camera_id:
                continue
            route = self._routes.get((source.camera_id, target.camera_id))
            if route is None:
                rejected['no_route'] += 1
                continue
            elapsed = target.first_seen_at-source.last_seen_at
            if elapsed <= 0 or not route['min_seconds'] <= elapsed <= route['max_seconds']:
                rejected['chronological_or_travel'] += 1
                continue
            similarity = float(np.clip(np.dot(source.descriptor, target.descriptor), -1., 1.))
            if similarity < self.appearance_threshold:
                rejected['below_appearance_threshold'] += 1
                continue
            plate = _plate_evidence(source, target)
            if plate['full_plate_conflict']:
                rejected['conflicting_full_plate'] += 1
                continue
            score = min(1., similarity+plate['boost'])
            candidates.append(dict(source_observation_id=source.observation_id,
                                   source_camera=source.camera_id, source_track_id=source.track_id,
                                   source_observed_at=source.observed_at, appearance_similarity=similarity,
                                   source_first_seen_at=source.first_seen_at, source_last_seen_at=source.last_seen_at,
                                   review_score=score, score=score, plate_evidence=plate,
                                   category_agreement=(source.category == target.category
                                                       if 'unknown' not in (source.category, target.category) else None),
                                   travel_seconds=elapsed, min_seconds=route['min_seconds'], max_seconds=route['max_seconds'],
                                   requires_review=True))
        candidates.sort(key=lambda candidate:(-candidate['review_score'], candidate['source_observation_id']))
        return candidates, rejected

    def register_batch(self, observations):
        if not isinstance(observations, (list, tuple)) or len(observations) > self.max_entries:
            raise ValueError(f'observations must be a list of at most {self.max_entries} records.')
        batch = [self._validate(item) for item in observations]
        if len({item.key for item in batch}) != len(batch):
            raise ValueError('A batch cannot contain the same camera/local track twice.')
        with self._lock:
            now = self._clock()
            self._expire(now)
            results, proposed = [], []
            for index, target in enumerate(batch):
                previous = self._entries.get(target.key)
                stale = previous is not None and target.last_seen_at <= previous.last_seen_at
                candidates, rejected = self._candidates(target) if not stale else ([], {})
                result = dict(observation_id=target.observation_id, camera_id=target.camera_id,
                              track_id=target.track_id, observed_at=target.observed_at,
                              first_seen_at=target.first_seen_at, last_seen_at=target.last_seen_at,
                              category=target.category, status='new', reason='no_eligible_prior_observation',
                              candidates=candidates[:3], matches=candidates[:3], selected_candidate=None,
                              appearance_threshold=self.appearance_threshold, ambiguity_margin=self.ambiguity_margin,
                              rejected_candidates=rejected, recorded=not stale, requires_review=True,
                              note='Candidate association only; not a verified cross-camera identity. Scores are uncalibrated.')
                if stale:
                    result['reason'] = 'out_of_order_local_track_observation'
                elif candidates:
                    if len(candidates) > 1 and candidates[0]['review_score']-candidates[1]['review_score'] < self.ambiguity_margin:
                        result.update(status='ambiguous', reason='appearance_candidates_too_close')
                    else:
                        proposed.append((candidates[0]['review_score'], index, candidates[0]))
                results.append(result)
            # Globally order the strongest unambiguous proposals. A losing
            # observation remains ambiguous rather than being silently forced
            # onto its second-choice source after a collision.
            used_sources = set()
            for _, index, candidate in sorted(proposed, key=lambda proposal:(-proposal[0], proposal[1])):
                result = results[index]
                if candidate['source_observation_id'] in used_sources:
                    result.update(status='ambiguous', reason='source_already_assigned_in_batch')
                else:
                    used_sources.add(candidate['source_observation_id'])
                    result.update(status='candidate', reason='appearance_and_route_candidate',
                                  selected_candidate=candidate)
            for target, result in zip(batch, results):
                if result['recorded']:
                    target.ingested_at = now
                    self._entries.pop(target.key, None)
                    self._entries[target.key] = target
                    while len(self._entries) > self.max_entries:
                        self._entries.popitem(last=False)
                self._recent.append((now, deepcopy(result)))
            return deepcopy(results)

    def match_batch(self, observations):
        """Alias: match against prior observations, then register this batch."""
        return self.register_batch(observations)
