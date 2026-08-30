"""Post-hoc Modality LUT attachment for quantitative (-q/--label) series.

RescaleSlope/RescaleIntercept/RescaleType (and an optional default window)
cannot go through SimpleITK's writer: GDCM interprets their presence in the
metadata dictionary as a request to inverse-rescale the pixel data while
writing, and rejects a non-integer slope or intercept outright. So the series
is written first with the stored int16 values it already has (see
writer.write_slice), and this module attaches the LUT afterwards as a pure
pydicom metadata edit.
"""

import glob
import os

import pydicom

from .formatting import ds16


def attach_modality_lut(
    out_dir, slope, intercept, units, window_center=None, window_width=None
):
    """Add RescaleSlope/Intercept/Type (+ optional default window) to every
    *.dcm file already written in `out_dir`.

    Without these tags a PACS ROI reports raw stored integers instead of
    real-world units (rCBV / ALFF / FA / ...), which is the entire point of
    quantitative mode -- so a missing pydicom import is left to raise rather
    than being silently swallowed.
    """
    slope_s, intercept_s = ds16(slope), ds16(intercept)
    files = sorted(glob.glob(os.path.join(out_dir, "*.dcm")))
    for f in files:
        ds_obj = pydicom.dcmread(f)
        ds_obj.RescaleSlope = slope_s
        ds_obj.RescaleIntercept = intercept_s
        ds_obj.RescaleType = units
        if window_center is not None:
            ds_obj.WindowCenter = ds16(window_center)
            ds_obj.WindowWidth = ds16(window_width)
        ds_obj.save_as(f)
    return len(files)
