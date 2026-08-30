import pytest
from nifti2dicom.crop import compute_crop_box, global_max_fov


def test_global_max_fov_picks_largest_axis():
    size = (64, 64, 64)
    spacing = (2.0, 1.0, 3.0)
    # extents: 128, 64, 192 -> max is 192
    assert global_max_fov(size, spacing) == 192.0


def test_compute_crop_box_square_png_matches_fov():
    # PNG is exactly the min-dimension square used for zoom; content should
    # fill proportionally to FOV vs global max FOV.
    box = compute_crop_box(
        png_w=256, png_h=256, h_fov=128, v_fov=128, global_max_fov_mm=128
    )
    # scale = min(256, 256) / 128 = 2.0 px/mm
    assert box.scale == pytest.approx(2.0)
    assert box.content_w == 256
    assert box.content_h == 256
    assert box.crop_left == 0
    assert box.crop_right == 256
    assert box.crop_top == 0
    assert box.crop_bot == 256


def test_compute_crop_box_smaller_fov_than_global_max():
    # global_max_fov bigger than this plane's own FOV -> content smaller than PNG,
    # reproducing mrview's single fixed zoom across all 3 planes.
    box = compute_crop_box(
        png_w=256, png_h=256, h_fov=64, v_fov=128, global_max_fov_mm=128
    )
    assert box.scale == pytest.approx(2.0)
    assert box.content_w == 128
    assert box.content_h == 256
    assert box.crop_top == 0
    assert box.crop_bot == 256


def test_compute_crop_box_non_square_png():
    box = compute_crop_box(
        png_w=320, png_h=240, h_fov=100, v_fov=100, global_max_fov_mm=100
    )
    # scale = min(320, 240) / 100 = 2.4
    assert box.scale == pytest.approx(2.4)
    assert box.content_w == 240
    assert box.content_h == 240


def test_compute_crop_box_indices_are_slice_ready():
    box = compute_crop_box(
        png_w=100, png_h=100, h_fov=50, v_fov=50, global_max_fov_mm=100
    )
    # crop_right/crop_bot are exclusive end indices, width should match content
    assert box.crop_right - box.crop_left == box.content_w
    assert box.crop_bot - box.crop_top == box.content_h
