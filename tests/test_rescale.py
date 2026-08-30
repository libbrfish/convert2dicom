import numpy as np
import pytest
from convert2dicom.rescale import (
    RescaleError,
    clean_nonfinite,
    compute_label_rescale,
    compute_legacy_rescale,
    compute_quantitative_rescale,
    compute_window,
)


def test_clean_nonfinite_no_nans():
    arr = np.array([1.0, 2.0, -3.0])
    out, vmin, vmax, n, fill = clean_nonfinite(arr)
    assert n == 0
    assert fill is None
    assert vmin == -3.0 and vmax == 2.0
    assert np.array_equal(out, arr)


def test_clean_nonfinite_fills_with_zero_when_in_range():
    arr = np.array([-1.0, np.nan, 2.0, np.inf])
    out, vmin, vmax, n, fill = clean_nonfinite(arr)
    assert n == 2
    assert fill == 0.0
    assert vmin == -1.0 and vmax == 2.0
    assert out[1] == 0.0 and out[3] == 0.0


def test_clean_nonfinite_fills_with_vmin_when_zero_out_of_range():
    arr = np.array([5.0, np.nan, 10.0])
    out, _vmin, _vmax, _n, fill = clean_nonfinite(arr)
    assert fill == 5.0  # vmin, since 0 is not within [5, 10]
    assert out[1] == 5.0


def test_clean_nonfinite_does_not_mutate_input():
    arr = np.array([1.0, np.nan, 3.0])
    original = arr.copy()
    clean_nonfinite(arr)
    assert np.array_equal(arr, original, equal_nan=True)


def test_clean_nonfinite_all_nonfinite_raises():
    arr = np.array([np.nan, np.inf, -np.inf])
    with pytest.raises(RescaleError):
        clean_nonfinite(arr)


# --- label mode -------------------------------------------------------


def test_label_rescale_integers_always_converts():
    # Regression test for the original bug: img_int16/slope/intercept were
    # only assigned inside the "non-integer, rounding" warning branch, so an
    # already-integer label map (the normal case) never got them set at all.
    arr = np.array([0.0, 1.0, 2.0, 5.0])
    img_int16, slope, intercept, was_non_integer = compute_label_rescale(arr, 0.0, 5.0)
    assert was_non_integer is False
    assert slope == 1.0 and intercept == 0.0
    assert img_int16.dtype == np.int16
    assert np.array_equal(img_int16, arr.astype(np.int16))


def test_label_rescale_rounds_non_integers_and_flags_it():
    arr = np.array([0.0, 1.4, 2.6])
    img_int16, _slope, _intercept, was_non_integer = compute_label_rescale(arr, 0.0, 2.6)
    assert was_non_integer is True
    assert np.array_equal(img_int16, np.array([0, 1, 3], dtype=np.int16))


def test_label_rescale_out_of_int16_range_raises():
    with pytest.raises(RescaleError):
        compute_label_rescale(np.array([0.0, 40000.0]), 0.0, 40000.0)


# --- quantitative mode --------------------------------------------------


def test_quantitative_rescale_scales_about_zero():
    arr = np.array([-10.0, 0.0, 20.0])
    img_int16, slope, intercept = compute_quantitative_rescale(arr, -10.0, 20.0)
    assert intercept == 0.0
    peak = 20.0
    assert slope == pytest.approx(peak / 32767.0)
    # value 20.0 should map close to +32767
    assert img_int16[2] == pytest.approx(32767, abs=1)
    # value -10.0 should map to roughly -32767/2
    assert img_int16[0] == pytest.approx(round(-10.0 / slope), abs=1)


def test_quantitative_rescale_all_zeros_uses_slope_one():
    arr = np.zeros(5)
    img_int16, slope, intercept = compute_quantitative_rescale(arr, 0.0, 0.0)
    assert slope == 1.0 and intercept == 0.0
    assert np.all(img_int16 == 0)


def test_quantitative_rescale_clips_to_int16_range():
    # Construct data whose /slope value would exceed int16 due to rounding;
    # should be clipped, not overflow/wrap.
    arr = np.array([-20.0, 20.0])
    img_int16, _slope, _intercept = compute_quantitative_rescale(arr, -20.0, 20.0)
    i16 = np.iinfo(np.int16)
    assert img_int16.min() >= i16.min and img_int16.max() <= i16.max


# --- legacy mode ---------------------------------------------------------


def test_legacy_rescale_fills_int16_range():
    arr = np.array([0.0, 50.0, 100.0])
    img_int16, _slope, intercept = compute_legacy_rescale(arr, 100.0)
    i16 = np.iinfo(np.int16)
    assert intercept == 0.0
    assert img_int16[2] == pytest.approx(i16.max, abs=1)
    assert img_int16[0] == 0


def test_legacy_rescale_all_zero_raises():
    with pytest.raises(RescaleError):
        compute_legacy_rescale(np.zeros(3), 0.0)


def test_legacy_rescale_negative_values_clipped_not_wrapped():
    # Historically this used to wrap modulo 2**16 on overflow; must clip instead.
    arr = np.array([-1000.0, 0.0, 100.0])
    img_int16, _slope, _intercept = compute_legacy_rescale(arr, 100.0)
    i16 = np.iinfo(np.int16)
    assert img_int16.min() >= i16.min
    assert img_int16[0] < 0  # still negative, not wrapped to a huge positive


# --- window computation ----------------------------------------------------


def test_compute_window_basic_percentiles():
    # Foreground values 1..100 (nonzero), background zeros should be excluded.
    fg_vals = np.arange(1, 101, dtype=np.int16)
    bg = np.zeros(
        1000, dtype=np.int16
    )  # forces "masked" branch (fg.size >= 100 after filtering nonzero)
    img_int16 = np.concatenate([fg_vals, bg])
    wc, ww = compute_window(
        img_int16, slope=1.0, intercept=0.0, is_label=False, headroom=1.0
    )
    assert wc is not None and ww is not None
    assert ww > 0


def test_compute_window_label_mode_spans_full_range():
    img_int16 = np.array([0, 1, 2, 3, 4] * 30, dtype=np.int16)  # >=100 elements
    _wc, ww = compute_window(img_int16, slope=1.0, intercept=0.0, is_label=True)
    # label mode uses min/max of foreground (nonzero) directly, no percentile clipping
    assert ww == pytest.approx(4.0 - 1.0)


def test_compute_window_headroom_widens_top():
    fg_vals = np.arange(1, 101, dtype=np.int16)
    bg = np.zeros(1000, dtype=np.int16)
    img_int16 = np.concatenate([fg_vals, bg])
    _wc1, ww1 = compute_window(img_int16, 1.0, 0.0, is_label=False, headroom=1.0)
    _wc2, ww2 = compute_window(img_int16, 1.0, 0.0, is_label=False, headroom=1.5)
    assert ww2 > ww1


def test_compute_window_no_dynamic_range_returns_none():
    img_int16 = np.zeros(10, dtype=np.int16)
    wc, ww = compute_window(img_int16, 1.0, 0.0, is_label=False)
    assert wc is None and ww is None


def test_compute_window_uses_real_units_not_stored_units():
    # slope far from 1: window must be computed in real (rescaled) units.
    fg_vals = np.arange(1, 101, dtype=np.int16)
    bg = np.zeros(1000, dtype=np.int16)
    img_int16 = np.concatenate([fg_vals, bg])
    slope = 2e-5
    _wc, ww = compute_window(img_int16, slope=slope, intercept=0.0, is_label=False)
    # width should scale down by slope relative to the stored-unit width
    _wc_stored, ww_stored = compute_window(
        img_int16, slope=1.0, intercept=0.0, is_label=False
    )
    assert ww == pytest.approx(ww_stored * slope, rel=1e-6)
