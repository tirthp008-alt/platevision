"""Unit tests for bounding box math, padding expansion, coordinate clamping, and cropping."""

import numpy as np
import pytest
from app.schemas.detection import BoundingBox, NormalizedBox
from app.services.cropper import (
    clamp_coordinate,
    expand_and_clamp_bbox,
    extract_plate_crop,
    to_normalized_box,
)


def test_clamp_coordinate():
    assert clamp_coordinate(10, 0, 100) == 10
    assert clamp_coordinate(-5, 0, 100) == 0
    assert clamp_coordinate(150, 0, 100) == 100


def test_expand_and_clamp_bbox_normal():
    # Image 1000x1000, Box: x=200, y=300, w=200, h=100
    # 10% padding -> pad_x = 20, pad_y = 10
    bbox = BoundingBox(x=200, y=300, width=200, height=100)
    expanded = expand_and_clamp_bbox(bbox, img_width=1000, img_height=1000, padding_ratio=0.10)

    assert expanded.x == 180
    assert expanded.y == 290
    assert expanded.width == 240
    assert expanded.height == 120


def test_expand_and_clamp_bbox_boundary():
    # Box touching top-left corner
    bbox = BoundingBox(x=5, y=5, width=100, height=50)
    expanded = expand_and_clamp_bbox(bbox, img_width=500, img_height=500, padding_ratio=0.10)

    assert expanded.x == 0  # clamped to 0
    assert expanded.y == 0  # clamped to 0
    assert expanded.width > 100
    assert expanded.height > 50

    # Box touching bottom-right corner
    bbox_br = BoundingBox(x=450, y=470, width=50, height=30)
    expanded_br = expand_and_clamp_bbox(bbox_br, img_width=500, img_height=500, padding_ratio=0.10)
    assert expanded_br.x + expanded_br.width <= 500
    assert expanded_br.y + expanded_br.height <= 500


def test_to_normalized_box():
    bbox = BoundingBox(x=250, y=500, width=500, height=250)
    norm = to_normalized_box(bbox, img_width=1000, img_height=1000)

    assert norm.x == 0.25
    assert norm.y == 0.5
    assert norm.width == 0.5
    assert norm.height == 0.25


def test_extract_plate_crop():
    # Create synthetic test canvas
    canvas = np.zeros((400, 600, 3), dtype=np.uint8)
    canvas[100:200, 150:350] = (255, 255, 255)  # White rectangle

    bbox = BoundingBox(x=150, y=100, width=200, height=100)
    crop, adjusted = extract_plate_crop(canvas, bbox, apply_padding=False)

    assert crop.shape == (100, 200, 3)
    assert np.all(crop == 255)
