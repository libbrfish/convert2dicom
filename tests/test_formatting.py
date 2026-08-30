from convert2dicom.formatting import ds, ds16


def test_ds_fixed_six_decimals():
    assert ds(0.9999999999999998) == "1.000000"
    assert ds(-1.0) == "-1.000000"
    assert ds(0) == "0.000000"


def test_ds_result_fits_dicom_ds_limit():
    assert len(ds(-123.456789012345)) <= 16


def test_ds16_keeps_significant_digits_for_small_slope():
    # A tiny slope must not be rounded to 6 decimals (0.000112), which would
    # be a large relative error; it should keep more significant digits.
    result = ds16(0.00011174462045)
    assert len(result) <= 16
    assert abs(float(result) - 0.00011174462045) / 0.00011174462045 < 1e-4


def test_ds16_never_exceeds_16_bytes():
    values = [0.00011174462045097997, 123456789.123456789, -0.5, 1.0, 0.0, 1e-10, 1e10]
    for v in values:
        result = ds16(v)
        assert len(result) <= 16, (v, result)


def test_ds16_zero():
    assert float(ds16(0.0)) == 0.0
    assert len(ds16(0.0)) <= 16


def test_ds16_round_trips_reasonably_for_ordinary_values():
    for v in [1.0, -1.0, 0.5, 3.14159, -2.71828]:
        assert abs(float(ds16(v)) - v) < 1e-6
