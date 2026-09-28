"""User-requested green-plate comparison; original readings always win display."""
import logging
import time
from app.services.green_preprocessing import prepare_green_emboss
from app.services.normalization import normalize_plate

logger = logging.getLogger(__name__)


def green_alternative(ocr, crop, original):
    enhanced, metadata = prepare_green_emboss(crop, original)
    metadata['ocr_ms'] = 0.0
    if enhanced is not None and ocr is not None:
        started = time.perf_counter()
        try:
            text, confidence = ocr.read(enhanced)
            metadata.update(candidate_text=normalize_plate(text)[0],
                            candidate_raw_text=text, candidate_ocr_confidence=float(confidence),
                            candidate_status='experimental_review_required')
        except Exception:
            logger.exception('Experimental green-plate OCR failed; preserving original reading')
            metadata.update(candidate_text='', candidate_ocr_confidence=0.,
                            candidate_status='unavailable', reason='experimental_ocr_failed')
        metadata['ocr_ms']=round((time.perf_counter()-started)*1000, 2)
    return enhanced, metadata
