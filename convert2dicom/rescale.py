"""Voxel-value -> int16 rescaling and default-window computation.

These are pure numpy functions (no SimpleITK/pydicom/file I/O) so they can be
unit tested directly on small arrays.

NOTE on bugs fixed vs. the original script:
  * Label mode: in the original, `img_int16`/`rescale_slope`/`rescale_intercept`
    were only assigned *inside* the "values are not integers, rounding" warning
    branch. For any label map whose values were already integers (the normal
    case), those variables were simply never set, so writing would crash with
    a NameError deeper in the pipeline. `compute_label_rescale` below always
    rounds/casts and always returns slope=1.0/intercept=0.0; the "not integer"
    condition only controls whether a warning is reported.
  * Quantitative mode: the informational
    "Quantitative mode: [...] -> int16, slope=... intercept=..." print was
    nested under the `_peak == 0` (all-zero) branch only, so it never printed
    for the normal non-zero case. `compute_quantitative_rescale` returns
    the values; the caller is responsible for logging, once, unconditionally.
"""

import numpy as np


class RescaleError(Exception):
    """Raised for a voxel-data problem that should abort conversion with a
    clear message, in place of the original script's ad hoc sys.exit(1)."""


def clean_nonfinite(img_data):
    """Replace NaN/Inf voxels with a sane fill value; report vmin/vmax.

    Non-finite voxels are mapped to 0 when 0 is inside the finite value range
    (normal for a masked brain map), otherwise to the finite minimum, so they
    don't propagate through later arithmetic and cast to arbitrary integers.

    Returns (cleaned_array, vmin, vmax, n_replaced, fill_used). `fill_used` is
    None when n_replaced is 0. Does not mutate the input.
    """
    nonfinite = ~np.isfinite(img_data)
    n_nonfinite = int(nonfinite.sum())
    finite = img_data[~nonfinite]
    if finite.size == 0:
        raise RescaleError("input image has no finite voxels")
    vmin, vmax = float(finite.min()), float(finite.max())
    fill = None
    if n_nonfinite:
        fill = 0.0 if vmin <= 0.0 <= vmax else vmin
        img_data = img_data.copy()
        img_data[nonfinite] = fill
    return img_data, vmin, vmax, n_nonfinite, fill


def compute_label_rescale(img_data, vmin, vmax):
    """Integer label/segmentation map: values kept unchanged (slope 1,
    intercept 0) so each label keeps its identity.

    Returns (img_int16, slope, intercept, was_non_integer).
    Raises RescaleError if [vmin, vmax] doesn't fit in int16.
    """
    i16 = np.iinfo(np.int16)
    if vmin < i16.min or vmax > i16.max:
        raise RescaleError(f"label values [{vmin}, {vmax}] do not fit in int16")
    was_non_integer = not np.allclose(img_data, np.round(img_data))
    img_int16 = np.round(img_data).astype(np.int16)
    return img_int16, 1.0, 0.0, was_non_integer


def compute_quantitative_rescale(img_data, vmin, vmax):
    """Scale about zero: stored = value / slope, intercept always 0.

    The intercept is deliberately kept at exactly 0 rather than shifted to
    pack the values into the full int16 range. GDCM's Rescaler asserts
    `intercept == (int)intercept` when writing integer pixel data, so a
    fractional intercept aborts the write outright. Anchoring at zero also
    keeps zero meaning zero, which matters for the masked background of a
    brain map. The cost is at most one bit of precision (1 part in 32767 of
    the largest magnitude), far below the noise of any map this handles.

    Returns (img_int16, slope, intercept).
    """
    i16 = np.iinfo(np.int16)
    peak = max(abs(vmin), abs(vmax))
    if peak > 0:
        slope = peak / 32767.0
        intercept = 0.0
        img_int16 = np.clip(np.round(img_data / slope), i16.min, i16.max).astype(
            np.int16
        )
    else:
        # All zeros: nothing to scale, and slope must stay non-zero.
        slope, intercept = 1.0, 0.0
        img_int16 = np.zeros(img_data.shape, dtype=np.int16)
    return img_int16, slope, intercept


def compute_legacy_rescale(img_data, vmax):
    """Historical behaviour: fill the int16 range from 0..max, rounded and
    clipped (not truncated/wrapped like the very first version of this
    script), with the scale factor recorded instead of discarded.

    Returns (img_int16, slope, intercept). Raises RescaleError if vmax == 0.
    """
    if vmax == 0:
        raise RescaleError("input image is all zeros; cannot rescale to int16")
    i16 = np.iinfo(np.int16)
    scale = i16.max / vmax
    img_int16 = np.clip(np.round(img_data * scale), i16.min, i16.max).astype(np.int16)
    return img_int16, 1.0 / scale, 0.0


def compute_window(img_int16, slope, intercept, is_label=False, headroom=1.0):
    """Default DICOM Window Center/Width from robust percentiles of the
    FOREGROUND real-world (rescaled) values.

    Background is excluded on purpose: these maps are often brain-masked, so
    a large fraction of every volume is exactly zero, which would otherwise
    drag the low percentile into the zero mass and the high percentile down
    with it.

    Percentiles/headroom are computed in REAL units (after slope/intercept),
    not stored int16 units, because Window Center/Width are applied *after*
    the Modality LUT (DICOM PS3.3 C.11.2) -- computing them from raw stored
    values makes the error scale with the slope.

    Returns (window_center, window_width), or (None, None) if the data has no
    dynamic range at all.
    """
    real = img_int16.astype(np.float64) * slope + intercept
    fg = real[img_int16 != 0]
    if fg.size < 100:  # not a masked map (or nearly empty): use everything
        fg = real

    if is_label:
        # Labels are categorical: span them all rather than clipping the tails.
        lo, hi = float(fg.min()), float(fg.max())
    else:
        # Asymmetric on purpose. The top is p99.5, not p98: on a DSC map the
        # choroid plexus (genuinely very vascular) and CSF (deconvolution
        # garbage) reach ~46x the cortical median, and a p98 top left the
        # ventricles as blown-out white blobs. p99.5 cuts the saturated voxels
        # from 2.0% to 0.5% of tissue.
        lo, hi = float(np.percentile(fg, 2)), float(np.percentile(fg, 99.5))
        # Optional extra headroom above p99.5, a per-map display preference
        # (there is no statistic that reliably separates maps that want it
        # from ones that don't), so it's passed in rather than inferred.
        if headroom and headroom != 1.0 and hi > 0:
            hi *= headroom

    if hi <= lo:
        lo, hi = float(real.min()), float(real.max())
    if hi > lo:
        return (lo + hi) / 2.0, hi - lo
    return None, None
