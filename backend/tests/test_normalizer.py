"""Unit tests for Indian license plate OCR normalization and position-aware character substitution."""

import pytest
from app.services.normalizer import clean_raw_ocr, normalize_indian_plate


def test_clean_raw_ocr():
    assert clean_raw_ocr("IND  GJ-01-AB-1234 ") == "GJ01AB1234"
    assert clean_raw_ocr("IN MH 12 DE 1433") == "MH12DE1433"
    assert clean_raw_ocr("  DL.01.CA.1234  ") == "DL01CA1234"
    assert clean_raw_ocr("") == ""


def test_normalize_standard_plates():
    # Direct valid inputs
    norm, fmt, status = normalize_indian_plate("GJ 01 AB 1234")
    assert norm == "GJ01AB1234"
    assert fmt == "GJ 01 AB 1234"
    assert status == "valid"

    norm, fmt, status = normalize_indian_plate("MH12DE1433")
    assert norm == "MH12DE1433"
    assert fmt == "MH 12 DE 1433"
    assert status == "valid"

    norm, fmt, status = normalize_indian_plate("DL01CA1234")
    assert norm == "DL01CA1234"
    assert status == "valid"

    norm, fmt, status = normalize_indian_plate("KA03MN4567")
    assert norm == "KA03MN4567"
    assert status == "valid"


def test_position_aware_character_substitution():
    # State code digits to letters: 6J -> GJ
    norm, fmt, status = normalize_indian_plate("6J01AB1234")
    assert norm == "GJ01AB1234"
    assert status == "valid"

    # District letters to digits: GJ O1 AB 1234 -> GJ 01 AB 1234
    norm, fmt, status = normalize_indian_plate("GJO1AB1234")
    assert norm == "GJ01AB1234"
    assert status == "valid"

    # Series digits to letters: GJ 01 A8 1234 -> GJ 01 AB 1234
    norm, fmt, status = normalize_indian_plate("GJ01A81234")
    assert norm == "GJ01AB1234"
    assert status == "valid"

    # Number letters to digits: GJ 01 AB I234 -> GJ 01 AB 1234
    norm, fmt, status = normalize_indian_plate("GJ01ABI234")
    assert norm == "GJ01AB1234"
    assert status == "valid"

    # Multiple confusions combined: 6J O1 A8 I234 -> GJ01AB1234
    norm, fmt, status = normalize_indian_plate("6JO1A8I234")
    assert norm == "GJ01AB1234"
    assert status == "valid"


def test_bh_series_normalization():
    norm, fmt, status = normalize_indian_plate("22 BH 1234 AA")
    assert norm == "22BH1234AA"
    assert fmt == "22 BH 1234 AA"
    assert status == "valid"

    # Disambiguation in BH series: 22 8H I234 AA -> 22 BH 1234 AA
    norm, fmt, status = normalize_indian_plate("228HI234AA")
    assert norm == "22BH1234AA"
    assert status == "valid"
