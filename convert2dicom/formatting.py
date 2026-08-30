"""Formatting helpers for DICOM DS (Decimal String) elements.

DS is limited to 16 bytes, which ordinary float formatting/repr routinely
overflows.
"""


def ds(v):
    """Format a float for a DICOM DS element with fixed 6-decimal precision.

    Python's repr (what `str()` gives) can emit 18+ characters for an ordinary
    oblique direction cosine -- '0.9999999999999998' -- which strict PACS and
    validators reject. Six decimals is well inside the 16-byte limit and far
    finer than any geometry this tool computes.
    """
    return f"{v:.6f}"


def ds16(v):
    """Format a float as DS keeping as many significant digits as fit in 16 bytes.

    Used for RescaleSlope, where `ds()`'s fixed 6 decimals would be ruinous: a
    slope of 0.00011174462045 would round to 0.000112, a 0.2% error on every
    voxel. Left to itself Python emits 0.00011174462045097997 (22 chars), which
    overflows DS.
    """
    v = float(v)
    for prec in range(15, 0, -1):
        s = f"{v:.{prec}g}"
        if len(s) <= 16:
            return s
    return f"{v:.6g}"[:16]
