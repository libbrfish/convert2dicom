"""SAG donor-match mode: match each rendered PNG to the spatially closest
donor frame and copy its exact IPP/IOP/PixelSpacing lineage, guaranteeing
PACS alignment without computing geometry from scratch for the donor side.

The in-plane field of view, output size and pixel spacing all come from the
NIfTI *underlay* (the volume the PNGs were rendered from), not the donor:
the donor only supplies identity (patient/study), Frame of Reference, and
the frame count/positions to match against. Using the donor's own in-plane
extent here is a documented past bug (mismatched FOV between donor and
underlay stretches/clips the image) -- see the comment on `compute_output_geometry`.
"""

import glob
import os

import numpy as np
import SimpleITK as sitk
from PIL import Image

from .crop import global_max_fov
from .donor_match_geom import compute_donor_frame_locs, compute_png_locs
from .geometry import orthonormal_row_col

LANCZOS = Image.Resampling.LANCZOS


class DonorMatchError(Exception):
    pass


def load_donor_3d(donor_dcm_path):
    """Read the donor as a full 3-D series (needed for per-frame IPP), forced
    to exactly 3 dimensions.

    A single *multi-frame* (enhanced) DICOM -- e.g. a Philips 100-frame
    SmartBrain localiser in one file -- reads back as 4-D (cols, rows,
    frames, 1), whose 3x3-indexed direction-matrix columns would silently
    read zeros. SAG is the only orientation routed through donor-match mode,
    so this used to present as "sagittal DICOMs fail, the others are fine".

    Returns a 3-D SimpleITK image (origin/spacing/direction/size all 3-D).
    """
    donor_dir = os.path.dirname(os.path.abspath(donor_dcm_path))
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(donor_dir)
    if series_ids:
        series_files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            donor_dir, series_ids[0]
        )
        donor_3d = sitk.ReadImage(series_files)
    else:
        donor_3d = sitk.ReadImage(donor_dcm_path)

    while donor_3d.GetDimension() > 3:
        donor_3d = donor_3d[..., 0]
    if donor_3d.GetDimension() == 2:
        donor_3d = sitk.JoinSeries(donor_3d)
    return donor_3d


def donor_basis(donor_3d):
    """Extract an orthonormal (row_dir, col_dir, slice_dir) basis from a
    donor's own direction matrix, tolerating small numerical noise via
    Gram-Schmidt. Raises DonorMatchError on a degenerate (zero) direction."""
    d3 = donor_3d.GetDirection()
    donor_row_raw = [d3[0], d3[3], d3[6]]
    donor_col_raw = [d3[1], d3[4], d3[7]]
    try:
        return orthonormal_row_col(donor_row_raw, donor_col_raw)
    except ValueError as e:
        raise DonorMatchError(
            f"donor has a degenerate direction matrix ({donor_3d.GetDimension()}-D, "
            f"direction {d3}). Cannot derive geometry from this donor; use a different one."
        ) from e


def compute_output_geometry(underlay_geom):
    """In-plane FOV/size/spacing for SAG output slices, taken from the
    underlay (not the donor).

    The PNGs are renders of the underlay, so their in-plane extent is the
    underlay's: for SAG, j (A-P) horizontally and k (S-I) vertically. Using
    the donor's own in-plane extent instead (donor_cols * donor_pixel_spacing)
    is only correct when the donor happens to share the underlay's field of
    view -- with, say, a 320x320 @ 1.09mm SmartBrain localiser (350mm FOV)
    donating for a 64x64x64 @ 2mm underlay (128mm FOV), every SAG series used
    to claim a 350mm in-plane extent for 128mm of anatomy, a 2.7x stretch
    clipped at the frame edge.

    Returns dict with j_fov, k_fov, out_w, out_h, out_sp_x, out_sp_y,
    out_slice_sp, global_max_fov_mm.
    """
    _origin, spacing, _direction, size = underlay_geom
    return {
        "j_fov": size[1] * spacing[1],
        "k_fov": size[2] * spacing[2],
        "out_w": size[1],
        "out_h": size[2],
        "out_sp_x": spacing[1],
        "out_sp_y": spacing[2],
        "out_slice_sp": spacing[0],  # sagittal step = underlay i spacing
        "global_max_fov_mm": global_max_fov(size, spacing),
    }


def build_frame_and_png_tables(underlay_geom, donor_geom, n_pngs):
    """Build (donor_frames, png_table) for matching, per compute_output_geometry's
    doc. `donor_geom` is (origin, spacing, direction, size) for the *donor* 3-D
    volume (used only for slice-normal direction / spacing / frame count).
    """
    donor_origin, donor_spacing, donor_direction, donor_size = donor_geom
    _, _donor_col_dir, donor_slice_dir = orthonormal_row_col(
        [donor_direction[0], donor_direction[3], donor_direction[6]],
        [donor_direction[1], donor_direction[4], donor_direction[7]],
    )
    n_donor_frames = donor_size[2]
    donor_frames = compute_donor_frame_locs(
        n_donor_frames, donor_origin, donor_slice_dir, donor_spacing[2]
    )

    nifti_origin, nifti_spacing, nifti_direction, _ = underlay_geom
    nifti_i_dir = [nifti_direction[0], nifti_direction[3], nifti_direction[6]]
    png_locs = compute_png_locs(
        n_pngs, nifti_origin, nifti_i_dir, nifti_spacing[0], donor_slice_dir
    )

    if len(png_locs) != n_donor_frames:
        # Caller decides whether/how to surface this warning; still proceeds
        # with nearest-neighbour-by-order matching, matching original behavior.
        pass

    return donor_frames, png_locs, n_donor_frames


def crop_and_resize_png(png_path, crop_box, out_w, out_h):
    """Load a PNG, crop to `crop_box`, resize to (out_w, out_h) with Lanczos."""
    img_arr = np.array(Image.open(png_path).convert("RGB"))
    img_arr = img_arr[
        crop_box.crop_top : crop_box.crop_bot,
        crop_box.crop_left : crop_box.crop_right,
        :,
    ]
    return np.array(Image.fromarray(img_arr).resize((out_w, out_h), LANCZOS))


def list_pngs(directory):
    pngs = sorted(glob.glob(os.path.join(directory, "*.png")))
    if not pngs:
        raise FileNotFoundError(f"No PNG files found in {directory}")
    return pngs
