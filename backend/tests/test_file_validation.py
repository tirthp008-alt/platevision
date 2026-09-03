"""Unit tests for image file validation and safe decoding."""

import io
import numpy as np
from PIL import Image
import pytest
from app.utils.image_ops import decode_image_safely, validate_image_bytes


def test_validate_empty_image():
    valid, err = validate_image_bytes(b"")
    assert not valid
    assert "empty" in err.lower()


def test_validate_oversized_image():
    # 11 MB dummy payload with 10MB limit
    fake_large_data = b"\xff\xd8" + b"0" * (11 * 1024 * 1024)
    valid, err = validate_image_bytes(fake_large_data, max_size_mb=10)
    assert not valid
    assert "exceeds" in err.lower()


def test_validate_corrupted_header():
    valid, err = validate_image_bytes(b"NON_IMAGE_DATA_CORRUPTED")
    assert not valid


def test_valid_jpeg_png_validation_and_decode():
    # Generate real in-memory JPEG
    img = Image.new("RGB", (200, 100), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    jpeg_bytes = buf.getvalue()

    valid, err = validate_image_bytes(jpeg_bytes)
    assert valid
    assert err is None

    decoded = decode_image_safely(jpeg_bytes)
    assert decoded is not None
    assert decoded.shape == (100, 200, 3)
