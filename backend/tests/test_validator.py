"""Unit tests for Indian license plate validation patterns."""

import pytest
from app.services.validator import validate_indian_plate_format


def test_valid_indian_states():
    status, comp = validate_indian_plate_format("GJ01AB1234")
    assert status == "valid"
    assert comp["state"] == "GJ"
    assert comp["district"] == "01"
    assert comp["series"] == "AB"
    assert comp["number"] == "1234"

    status, comp = validate_indian_plate_format("MH12DE1433")
    assert status == "valid"
    assert comp["state"] == "MH"

    status, comp = validate_indian_plate_format("DL01CA1234")
    assert status == "valid"
    assert comp["state"] == "DL"


def test_bh_series_validation():
    status, comp = validate_indian_plate_format("22BH1234AA")
    assert status == "valid"
    assert comp["type"] == "BH_SERIES"
    assert comp["year"] == "22"
    assert comp["code"] == "BH"
    assert comp["number"] == "1234"
    assert comp["series"] == "AA"


def test_unknown_state_possible():
    # ZZ is not a recognized state code, but pattern matches
    status, comp = validate_indian_plate_format("ZZ01AB1234")
    assert status == "possible"


def test_invalid_arbitrary_string():
    status, comp = validate_indian_plate_format("HELLO-WORLD")
    assert status == "uncertain"

    status, comp = validate_indian_plate_format("")
    assert status == "uncertain"
