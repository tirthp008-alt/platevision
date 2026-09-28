"""Group trustworthy registration readings without collapsing geometric evidence."""
import math
from collections import OrderedDict

from app.services.normalization import normalize_plate


def _number(value, default=None):
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def _registration(plate):
    confidence = _number(plate.get('ocr_confidence', plate.get('confidence')))
    review = plate.get('ocr_review') or {}
    raw = plate.get('raw_text')
    text = plate.get('normalized_text') or plate.get('text') or ''
    if (plate.get('format_status') != 'valid' or confidence is None or not .8 <= confidence <= 1.
            or plate.get('experimental_review_only') or plate.get('requires_review')
            or plate.get('localization_requires_review')
            or (isinstance(review, dict) and (review.get('requires_review') or review.get('conflicting_readings')))
            or not isinstance(text, str) or '?' in text or (isinstance(raw, str) and '?' in raw)):
        return None
    normalized, formatted, status = normalize_plate(text)
    return (normalized, formatted, confidence) if status == 'valid' else None


def _simultaneous(members, scope, frames_by_track):
    if len(members) < 2:
        return False
    if scope == 'frame':
        return True
    if frames_by_track is not None and all(member['id'] in frames_by_track for member in members):
        seen = set()
        for member in members:
            frames = set(frames_by_track[member['id']])
            if seen.intersection(frames):
                return True
            seen.update(frames)
        return False
    intervals = [(_number(member['plate'].get('first_seen')), _number(member['plate'].get('last_seen')))
                 for member in members]
    # Missing timing cannot establish that sightings happened separately.
    if any(start is None or end is None or end < start for start, end in intervals):
        return True
    intervals.sort()
    last_end = intervals[0][1]
    for start, end in intervals[1:]:
        if start <= last_end:
            return True
        last_end = max(last_end, end)
    return False


def summarize_recognized_plates(plates, *, read_text=True, scope='recording', frames_by_track=None):
    """Return review rows; input tracks, detections and counts remain untouched.

    Only exact valid readings with OCR confidence at least .8 are eligible.
    Simultaneous tracks sharing a reading remain separate ambiguous rows.
    """
    if scope not in ('frame', 'recording'):
        raise ValueError('Plate summary scope must be frame or recording.')
    grouped = OrderedDict()
    unresolved = []
    for index, plate in enumerate(plates):
        identifier = plate.get('id', plate.get('track_id', index + 1))
        registration = _registration(plate) if read_text else None
        if registration is None:
            unresolved.append(identifier)
            continue
        text, formatted, confidence = registration
        grouped.setdefault(text, []).append(dict(id=identifier, plate=plate, text=text,
                                                 formatted=formatted, confidence=confidence))
    groups = []
    recognized_count = sum(len(members) for members in grouped.values())
    for text, members in grouped.items():
        ambiguous = _simultaneous(members, scope, frames_by_track)
        for occurrences in ([member] for member in members) if ambiguous else [members]:
            representative = max(occurrences, key=lambda member: member['confidence'])
            starts = [_number(member['plate'].get('first_seen')) for member in occurrences]
            ends = [_number(member['plate'].get('last_seen')) for member in occurrences]
            starts = [value for value in starts if value is not None]
            ends = [value for value in ends if value is not None]
            groups.append(dict(id=f'recognized-{len(groups)+1}', text=text, normalized_text=text,
                               formatted_text=representative['formatted'],
                               ocr_confidence=representative['confidence'],
                               representative_track_id=representative['id'],
                               track_ids=[member['id'] for member in occurrences], track_count=len(occurrences),
                               first_seen=min(starts) if starts else None, last_seen=max(ends) if ends else None,
                               observations=sum(max(0, int(_number(member['plate'].get('observations'), 1)))
                                                for member in occurrences),
                               status='ambiguous' if ambiguous else 'recognized',
                               ambiguity_reason='simultaneous_tracks_share_text' if ambiguous else None))
    ambiguous_count = sum(group['track_count'] for group in groups if group['status'] == 'ambiguous')
    return dict(enabled=bool(read_text), scope=scope, minimum_ocr_confidence=.8,
                unique_registrations=sum(group['status'] == 'recognized' for group in groups),
                recognized_track_count=recognized_count, ambiguous_track_count=ambiguous_count,
                unresolved_track_count=len(unresolved), unresolved_track_ids=unresolved,
                suppressed_repeat_count=recognized_count-len(groups), groups=groups,
                note='Exact high-confidence registration readings are grouped only across separate sightings. '
                     'Simultaneous repeated text remains ambiguous. These are OCR summaries, not verified vehicle identities; '
                     'all geometric tracks, boxes and counts are retained.')
