import cv2
import numpy as np
import pytest

from app.services.green_preprocessing import (
    emboss_morphological_gradient, green_emboss_candidate, green_plate_features,
    needs_green_emboss, prepare_green_emboss,
)


def plate(colour=(45, 110, 25), text_colour=None):
    image = np.full((80, 320, 3), colour, np.uint8)
    if text_colour:
        cv2.putText(image, 'KA25AB1234', (8, 54), cv2.FONT_HERSHEY_SIMPLEX,
                    .85, text_colour, 2, cv2.LINE_AA)
    return image


def test_faint_green_gate_preserves_normal_plate_path():
    assert needs_green_emboss(plate(text_colour=(50, 120, 30)))
    assert not needs_green_emboss(plate(text_colour=(255, 255, 255)))
    assert green_emboss_candidate(plate((235, 235, 235), (20, 20, 20))) is None
    assert green_emboss_candidate(plate((10, 220, 230), (20, 20, 20))) is None


def test_enhancement_preserves_original_and_is_bounded():
    original = cv2.resize(plate(text_colour=(50, 120, 30)), (2000, 500))
    snapshot = original.copy()
    enhanced = emboss_morphological_gradient(original)
    assert enhanced.dtype == np.uint8 and enhanced.ndim == 3
    assert enhanced.shape[1] <= 640 and enhanced.shape[0] <= 160
    assert np.array_equal(original, snapshot)
    assert enhanced.max() - enhanced.min() > 100


def test_blank_green_plate_does_not_invent_edges():
    enhanced = emboss_morphological_gradient(plate())
    assert np.all(enhanced == 255)


def test_extreme_crop_aspect_never_rounds_dimension_to_zero():
    image = np.full((8192, 8, 3), (45, 110, 25), np.uint8)
    assert needs_green_emboss(image)
    enhanced = emboss_morphological_gradient(image)
    assert enhanced.shape == (160, 1, 3)


def test_green_gate_ignores_white_frame():
    image = plate(text_colour=(50, 120, 30))
    cv2.rectangle(image, (0, 0), (319, 79), (255, 255, 255), 5)
    assert needs_green_emboss(image)
    assert green_plate_features(image)['white_fraction'] < .04


def test_opt_in_candidate_never_replaces_original_reading_or_confidence():
    reading = ('KA25AB1234', .97)
    enhanced, metadata = prepare_green_emboss(plate(text_colour=(50, 120, 30)), reading)
    assert enhanced is not None and metadata['requires_review']
    assert metadata['original_text'] == reading[0]
    assert metadata['original_ocr_confidence'] == reading[1]
    assert metadata['original_preserved'] and not metadata['original_uncertain']
    assert metadata['preprocessing_ms'] >= 0
    assert 'candidate_confidence' not in metadata  # Only OCR may supply that score.


def test_opt_in_skips_normal_plates_with_an_explicit_reason():
    enhanced, metadata = prepare_green_emboss(plate((240, 240, 240)), ('BLUR', .2))
    assert enhanced is None and not metadata['applied']
    assert metadata['reason'] == 'insufficient_green_background'
    assert metadata['original_uncertain']


@pytest.mark.parametrize('crop', [None, np.zeros((0, 20, 3), np.uint8),
                                     np.zeros((20, 20), np.uint8),
                                     np.zeros((20, 20, 3), np.float32)])
def test_invalid_crops_are_skipped_by_gate_and_rejected_by_filter(crop):
    assert green_emboss_candidate(crop) is None
    with pytest.raises(ValueError):
        emboss_morphological_gradient(crop)
