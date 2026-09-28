"""Optional, once-per-observation appearance matching for review only."""
import logging
import math
import time

logger = logging.getLogger(__name__)
MODEL_NAME = 'FastReID VeRi SBS (R50-IBN)'
REVIEW_NOTE = ('Cross-camera candidates require review and are not verified identities. '
               'VeRi mainly covers four-wheeled vehicles; scores are uncalibrated.')


def strongest_plate(plates):
    """Keep actual OCR evidence separate from appearance-only observations."""
    usable = []
    for plate in plates:
        if not isinstance(plate, dict) or plate.get('experimental_review_only') or plate.get('localization_requires_review'):
            continue
        review = plate.get('ocr_review') or {}
        if plate.get('requires_review') or (isinstance(review, dict) and (review.get('requires_review') or review.get('conflicting_readings'))):
            continue
        # Normalization can remove positional ? wildcards or replace visually
        # ambiguous characters. Matching must use the actual reading if kept.
        text = plate.get('raw_text') if 'raw_text' in plate else plate.get('normalized_text') or plate.get('text')
        if not isinstance(text, str):
            continue
        text = text.upper().replace(' ', '').replace('-', '')
        if not 1 <= len(text) <= 20 or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789?' for c in text):
            continue
        value = plate.get('ocr_confidence') if 'ocr_confidence' in plate else plate.get('confidence', 0.)
        try:
            confidence = float(value) if value is not None else 0.
        except (TypeError, ValueError, OverflowError):
            continue
        if isinstance(value, bool) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            continue
        usable.append((text, confidence))
    return max(usable, key=lambda pair:pair[1]) if usable else (None, 0.)


def match_observations(observations, context, checkpoint=None):
    """Embed one crop at a time; never expose images or descriptors in results.

    Gallery mutation happens only after every embedding succeeded. An optional
    model failure cannot invalidate the main plate-detection response.
    Videos may stage descriptors during export to avoid retaining vehicle
    images or decoding the recording again. The gallery validates the complete
    descriptor batch before mutating its state.
    """
    started = time.perf_counter()
    result = dict(status='unavailable', model=MODEL_NAME, observations=[], observation_count=0,
                  processing_time_ms=0., note=REVIEW_NOTE)
    engine = context.get('engine') if context else None
    gallery = context.get('gallery') if context else None
    if engine is None or gallery is None:
        result['warning'] = 'Cross-camera appearance matching is unavailable; plate and vehicle results remain available.'
        return result
    try:
        prepared = []
        for observation in observations:
            if checkpoint:
                checkpoint()
            embedding = (observation['descriptor'] if 'descriptor' in observation
                         else engine.embed([observation['crop']])[0])
            if checkpoint:
                checkpoint()
            prepared.append(dict(camera_id=context['camera_id'], track_id=observation['track_id'],
                                 observed_at=observation['observed_at'],
                                 first_seen_at=observation.get('first_seen_at', observation['observed_at']),
                                 last_seen_at=observation.get('last_seen_at', observation['observed_at']),
                                 category=observation['category'], descriptor=embedding,
                                 plate_text=observation.get('plate_text'),
                                 plate_confidence=observation.get('plate_confidence', 0.)))
        if checkpoint:
            checkpoint()
        matched = gallery.register_batch(prepared)
        result.update(status='complete', observations=matched, observation_count=len(matched))
        if not matched:
            result['note'] = 'No eligible vehicle crops were available for cross-camera matching. ' + REVIEW_NOTE
    except InterruptedError:
        raise
    except Exception:
        logger.exception('Optional cross-camera appearance matching failed')
        result.update(status='failed', warning='Cross-camera appearance matching failed; plate and vehicle results remain available.')
    result['processing_time_ms'] = round((time.perf_counter()-started)*1000, 2)
    return result
