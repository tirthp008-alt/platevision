"""Crop adapter regressions without loading OCR weights or running inference."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def load_engine(monkeypatch):
    def load(rapid):
        monkeypatch.setitem(
            sys.modules, "rapidocr_onnxruntime", SimpleNamespace(RapidOCR=lambda: rapid)
        )
        path = Path(__file__).parents[1] / "app/services/ocr/engine.py"
        spec = importlib.util.spec_from_file_location("_ocr_engine_adapter_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.ocr_engine

    return load


@pytest.mark.parametrize("attribute", ["text_rec", "text_recognizer"])
def test_direct_recognizer_receives_one_crop_batch(load_engine, attribute):
    crop = np.zeros((48, 192, 3), dtype=np.uint8)
    calls = []

    def recognize(crops):
        assert isinstance(crops, list)
        assert len(crops) == 1 and crops[0] is crop
        calls.append(crops)
        return [(" MH12AB1234 ", 0.93)], 0.02

    engine = load_engine(SimpleNamespace(**{attribute: recognize}))
    assert engine._infer_recognizer(crop) == ("MH12AB1234", 0.93)
    assert len(calls) == 1


def test_installed_attribute_is_preferred_over_legacy_alias(load_engine):
    def wrong_alias(crops):
        pytest.fail("text_rec must be preferred when both attributes exist")

    engine = load_engine(SimpleNamespace(
        text_rec=lambda crops: ([("MH12AB1234", 0.9)], 0.01),
        text_recognizer=wrong_alias,
    ))
    assert engine._infer_recognizer(np.zeros((48, 192, 3))) == ("MH12AB1234", 0.9)


def test_wrapper_fallback_explicitly_disables_detection_and_classification(load_engine):
    crop = np.zeros((48, 192, 3), dtype=np.uint8)

    def rapid(image, **options):
        assert image is crop
        assert options == {"use_det": False, "use_cls": False, "use_rec": True}
        return [["KA01AB1234", 0.82]], [0.02]

    engine = load_engine(rapid)
    assert engine._infer_recognizer(crop) == ("KA01AB1234", 0.82)


@pytest.mark.parametrize("readings", [
    None, [], [("", 0.99)], [("   ", 0.99)], [("MH12AB1234",)],
    [([[0, 0], [10, 0], [10, 10], [0, 10]], "MH12AB1234", 0.99)],
    [([[0, 0], [10, 10]], 0.99)],
    [("MH12AB1234", float("nan"))], [("MH12AB1234", float("inf"))],
    [("MH12AB1234", -0.1)], [("MH12AB1234", 1.1)],
    [("MH12AB1234", 0.9), ("KA01AB1234", 0.8)],
])
def test_empty_or_malformed_outputs_never_become_confident_text(load_engine, readings):
    engine = load_engine(SimpleNamespace(text_rec=lambda crops: (readings, 0.01)))
    assert engine._infer_recognizer(np.zeros((48, 192, 3))) == ("", 0.0)


def test_recognizer_failure_is_uncertain(load_engine):
    def fail(crops):
        raise RuntimeError("recognizer unavailable")

    engine = load_engine(SimpleNamespace(text_rec=fail))
    result = engine.recognize(np.zeros((30, 120, 3), dtype=np.uint8))
    assert result.raw_text == ""
    assert result.normalized_text == ""
    assert result.confidence == 0.0
    assert result.format_status == "uncertain"


def test_empty_result_does_not_receive_fabricated_confidence(load_engine):
    engine = load_engine(SimpleNamespace(text_rec=lambda crops: ([], 0.01)))
    result = engine.recognize(np.zeros((30, 120, 3), dtype=np.uint8))
    assert result.raw_text == ""
    assert result.confidence == 0.0


def test_model_zero_confidence_is_preserved(load_engine):
    engine = load_engine(SimpleNamespace(
        text_rec=lambda crops: ([("MH12AB1234", 0.0)], 0.01)
    ))
    result = engine.recognize(np.zeros((30, 120, 3), dtype=np.uint8))
    assert result.raw_text == "MH12AB1234"
    assert result.confidence == 0.0


def test_existing_original_crop_retry_is_preserved(load_engine):
    responses = iter([([], 0.01), ([("MH12AB1234", 0.91)], 0.01)])
    engine = load_engine(SimpleNamespace(text_rec=lambda crops: next(responses)))
    result = engine.recognize(np.zeros((30, 120, 3), dtype=np.uint8))
    assert result.raw_text == "MH12AB1234"
    assert result.confidence == 0.91


def test_two_line_join_and_confidence_aggregation_are_preserved(load_engine):
    responses = iter([([("MH12", 0.9)], 0.01), ([("AB1234", 0.8)], 0.01)])
    engine = load_engine(SimpleNamespace(text_rec=lambda crops: next(responses)))
    result = engine.recognize(np.zeros((48, 60, 3), dtype=np.uint8))
    assert result.raw_text == "MH12 AB1234"
    assert result.normalized_text == "MH12AB1234"
    assert result.confidence == 0.85
