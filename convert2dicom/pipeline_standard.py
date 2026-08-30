"""Loading + int16 rescaling for the standard write path (NIfTI / 3D-TIFF /
a directory of PNG slices with no underlay-based geometry correction).
"""

import glob
import os

import numpy as np
import SimpleITK as sitk
from PIL import Image

from .rescale import (
    RescaleError,
    clean_nonfinite,
    compute_label_rescale,
    compute_legacy_rescale,
    compute_quantitative_rescale,
)


class UnsupportedDimensionError(RescaleError):
    """Raised for >3-D NIfTI input (e.g. a 4-D timeseries)."""


def load_png_dir_as_volume(directory):
    """Stack a directory of RGB PNG slices into a SimpleITK vector image.

    Returns (image, n_slices). Raises FileNotFoundError if the directory has
    no PNGs.
    """
    pngs = sorted(glob.glob(os.path.join(directory, "*.png")))
    if not pngs:
        raise FileNotFoundError(f"No PNG files found in {directory}")
    frames = [np.array(Image.open(p).convert("RGB")) for p in pngs]
    volume = np.stack(frames, axis=0)  # (Z, H, W, 3)
    img = sitk.GetImageFromArray(volume, isVector=True)
    img.SetSpacing([1.0, 1.0, 1.0])
    return img, len(pngs)


def convert_nifti_to_int16(
    nii_img, *, label=False, quantitative=False, window_headroom=1.0, log=print
):
    """Rescale a NIfTI volume's voxel values to int16 for DICOM storage.

    `label` and `quantitative` select the rescale mode (label implies
    quantitative, same as the CLI's `--label` -> `args.quantitative = True`).
    Mirrors the original script's three branches (label / quantitative /
    legacy fill-the-range), but fixes two bugs present there:
      * label mode's int16 conversion used to be nested under the
        "non-integer, rounding" warning branch, so it silently never ran for
        the common case of already-integer label values.
      * quantitative mode's informational print used to be nested under the
        "all zero" branch only, so it never printed for ordinary data.
    Both fixes only affect logging/control-flow correctness, not the numeric
    result for data that previously "worked".

    Returns (new_img, rescale_slope, rescale_intercept, is_label).
    Raises UnsupportedDimensionError for >3-D input.
    """
    if quantitative is False and label:
        quantitative = True  # --label implies -q, same as the CLI layer

    if nii_img.GetDimension() > 3:
        sz = nii_img.GetSize()
        raise UnsupportedDimensionError(
            f"{nii_img.GetDimension()}-D {sz}; only 3-D volumes are supported. "
            "For a timeseries (e.g. a 4-D PWI), split it first with "
            "`mrconvert <in> -coord 3 <index> vol.nii.gz` and convert each "
            "volume as its own series."
        )

    img_data = sitk.GetArrayFromImage(nii_img).astype(np.float64)
    img_data, vmin, vmax, n_nonfinite, fill = clean_nonfinite(img_data)
    if n_nonfinite:
        log(f"Note: {n_nonfinite} non-finite voxel(s) set to {fill}")

    if label:
        img_int16, slope, intercept, was_non_integer = compute_label_rescale(
            img_data, vmin, vmax
        )
        if was_non_integer:
            log("Warning: --label given but values are not integers; rounding")
        log(
            f"Label mode: {len(np.unique(img_int16))} distinct value(s), written unscaled"
        )
    elif quantitative:
        img_int16, slope, intercept = compute_quantitative_rescale(img_data, vmin, vmax)
        log(
            f"Quantitative mode: [{vmin:.6g}, {vmax:.6g}] -> int16, "
            f"slope={slope:.6g} intercept={intercept:.6g}"
        )
    else:
        log("Converting the nifti to 16bit")
        img_int16, slope, intercept = compute_legacy_rescale(img_data, vmax)

    new_img = sitk.GetImageFromArray(img_int16)
    new_img.CopyInformation(nii_img)
    new_img = sitk.DICOMOrient(new_img, "LPS")
    return new_img, slope, intercept, label


def apply_pixel_spacing_override(img, pixelspacing, log=print):
    """Override only the in-plane spacing (x, y), leaving slice spacing (z)
    untouched.

    `-p` describes a rendered screenshot's pixel size and says nothing about
    slice separation; overwriting the third spacing component as an earlier
    version of this tool did corrupted every slice position downstream.
    """
    if pixelspacing is None:
        return img
    sp = img.GetSpacing()
    z = sp[2] if len(sp) > 2 else 1.0
    img.SetSpacing([pixelspacing, pixelspacing, z])
    log(f"Pixel spacing set to {pixelspacing:.4f} mm (slice spacing kept at {z})")
    return img
