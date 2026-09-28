"""Unit tests for plate OCR helpers: normalization, aggregation, similarity."""

import pytest

from app.grid.plate_ocr import (
    PlateHypothesis,
    PlateReading,
    character_similarity,
    fuse_plate_observations,
    plate_agreement_score,
)


# --------------------------------------------------------------------------
# 1. Plate normalization (delegates to the existing PlateVision normalizer)
# --------------------------------------------------------------------------

def test_plate_normalization_removes_spaces_and_uppercases():
    from app.services.normalizer import normalize_indian_plate

    norm, formatted, status = normalize_indian_plate("gj 01 ab 1234")
    assert norm == "GJ01AB1234"
    assert formatted == "GJ 01 AB 1234"
    assert status == "valid"


def test_plate_normalization_corrects_common_ocr_confusions():
    from app.services.normalizer import normalize_indian_plate

    # O/0 confusion in the numeric district position.
    norm, _, _ = normalize_indian_plate("GJO1AB1234")
    assert norm.startswith("GJ0")


def test_plate_normalization_bh_series():
    from app.services.normalizer import normalize_indian_plate

    norm, _, status = normalize_indian_plate("22BH1234AA")
    assert norm == "22BH1234AA"
    assert status == "valid"


# --------------------------------------------------------------------------
# 3. Character-level similarity
# --------------------------------------------------------------------------

def test_character_similarity_exact():
    assert character_similarity("GJ01AB1234", "GJ01AB1234") == 1.0


def test_character_similarity_full_mismatch_is_low():
    # Y and 9 are not in any confusable group.
    assert character_similarity("GJ01AB1234", "GJ01XY9999") < 0.85


def test_character_similarity_confusable_substitution_partially_credited():
    # 0 <-> O are confusable, so the penalty should be smaller than a random change.
    confusable = character_similarity("GJ01AB1234", "GJO1AB1234")
    unrelated = character_similarity("GJ01AB1234", "GJXYAB1234")
    assert confusable > unrelated


# --------------------------------------------------------------------------
# 2. Multi-frame OCR aggregation
# --------------------------------------------------------------------------

def test_ocr_aggregation_prefers_repeated_high_confidence():
    obs = [
        PlateReading("GJ01AB12?8", "GJ01AB12", 0.72, 100.0),
        PlateReading("GJ01AB1234", "GJ01AB1234", 0.91, 105.0),
        PlateReading("GJ01AB1234", "GJ01AB1234", 0.94, 110.0),
    ]
    hyp = fuse_plate_observations(obs)
    assert hyp.plate_normalized == "GJ01AB1234"
    assert hyp.status == "confident"
    assert hyp.observation_count == 3
    assert hyp.confidence > 0.5


def test_ocr_aggregation_ambiguous_when_disagreement():
    obs = [
        PlateReading("GJ01AB1234", "GJ01AB1234", 0.80, 100.0),
        PlateReading("GJ99ZZ4321", "GJ99ZZ4321", 0.80, 105.0),
    ]
    hyp = fuse_plate_observations(obs)
    assert hyp.status == "ambiguous"
    assert hyp.plate_normalized == ""
    # Alternative readings must be preserved.
    assert len(hyp.candidates) >= 2


def test_ocr_aggregation_discards_degraded_readings():
    obs = [
        PlateReading("GJ01AB1234", "GJ01AB1234", 0.95, 100.0),
        PlateReading("ZZZZ", "ZZZZ", 0.20, 105.0),  # below min reading confidence
    ]
    hyp = fuse_plate_observations(obs)
    assert hyp.status == "confident"
    assert hyp.plate_normalized == "GJ01AB1234"


def test_ocr_aggregation_empty_observations():
    hyp = fuse_plate_observations([])
    assert hyp.status == "unknown"
    assert hyp.plate_normalized == ""


# --------------------------------------------------------------------------
# 11. Plate agreement score
# --------------------------------------------------------------------------

def test_plate_agreement_exact_match_high():
    a = PlateHypothesis(plate_normalized="GJ01AB1234", confidence=0.9, status="confident")
    b = PlateHypothesis(plate_normalized="GJ01AB1234", confidence=0.9, status="confident")
    assert plate_agreement_score(a, b) > 0.8


def test_plate_agreement_different_plate_near_zero():
    a = PlateHypothesis(plate_normalized="GJ01AB1234", confidence=0.9, status="confident")
    b = PlateHypothesis(plate_normalized="MH12DE1433", confidence=0.9, status="confident")
    assert plate_agreement_score(a, b) <= 0.15


def test_plate_agreement_partial_match_is_moderate():
    # Matching state/district but a different series must stay clearly below an exact match.
    a = PlateHypothesis(plate_normalized="GJ01AB1234", confidence=0.9, status="confident")
    b = PlateHypothesis(plate_normalized="GJ01DL1234", confidence=0.9, status="confident")
    score = plate_agreement_score(a, b)
    assert 0.05 < score <= 0.45
    assert score < plate_agreement_score(a, a)


def test_plate_agreement_single_confusable_typo_stays_high():
    # B and 8 are a known confusable pair; a single such substitution is a
    # plausible OCR slip so agreement stays high (but below an exact match).
    a = PlateHypothesis(plate_normalized="GJ01AB1234", confidence=0.9, status="confident")
    b = PlateHypothesis(plate_normalized="GJ01A81234", confidence=0.9, status="confident")
    score = plate_agreement_score(a, b)
    assert 0.6 < score < plate_agreement_score(a, a)


def test_plate_agreement_unrelated_confusable_pair_is_capped():
    # Exactly one confusable substitution but otherwise unrelated → moderate cap.
    a = PlateHypothesis(plate_normalized="GJ01AB1234", confidence=0.9, status="confident")
    b = PlateHypothesis(plate_normalized="MH12D81234", confidence=0.9, status="confident")
    score = plate_agreement_score(a, b)
    assert score <= 0.45


def test_plate_agreement_small_ocr_discrepancy_moderate():
    a = PlateHypothesis(plate_normalized="GJ01AB1234", confidence=0.9, status="confident")
    b = PlateHypothesis(plate_normalized="GJ01A?1234", confidence=0.7, status="confident")
    score = plate_agreement_score(a, b)
    assert score is not None
    assert 0.15 < score < 0.95


def test_plate_agreement_unknown_is_none():
    a = PlateHypothesis(status="unknown")
    b = PlateHypothesis(plate_normalized="GJ01AB1234", confidence=0.9, status="confident")
    assert plate_agreement_score(a, b) is None
