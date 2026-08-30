"""TRA/COR PNG mode: correct spatial geometry taken directly from the NIfTI
underlay (no donor-frame matching needed, unlike SAG donor-match mode --
TRA/COR PNGs map 1:1 onto underlay slice index already).
"""

import glob
import os

import numpy as np
from PIL import Image

from .crop import global_max_fov

LANCZOS = Image.Resampling.LANCZOS


def list_pngs(directory):
    pngs = sorted(glob.glob(os.path.join(directory, "*.png")))
    if not pngs:
        raise FileNotFoundError(f"No PNG files found in {directory}")
    return pngs


def plane_output_geometry(underlay_geom, plane):
    """In-plane FOV/size/spacing for TRA or COR output, from the underlay.

    Returns dict with h_fov, v_fov, out_w, out_h, out_sp_x, out_sp_y,
    global_max_fov_mm.
    """
    _origin, spacing, _direction, size = underlay_geom
    N_i, N_j, N_k = size[0], size[1], size[2]
    if plane == "TRA":
        h_fov, v_fov = N_i * spacing[0], N_j * spacing[1]
        out_w, out_h = N_i, N_j
        out_sp_x, out_sp_y = spacing[0], spacing[1]
    elif plane == "COR":
        h_fov, v_fov = N_i * spacing[0], N_k * spacing[2]
        out_w, out_h = N_i, N_k
        out_sp_x, out_sp_y = spacing[0], spacing[2]
    else:
        raise ValueError(
            f"Invalid plane {plane}; TRA/COR only (SAG uses donor-match mode)."
        )
    return {
        "h_fov": h_fov,
        "v_fov": v_fov,
        "out_w": out_w,
        "out_h": out_h,
        "out_sp_x": out_sp_x,
        "out_sp_y": out_sp_y,
        "global_max_fov_mm": global_max_fov(size, spacing),
    }


def crop_and_resize_png(png_path, crop_box, out_w, out_h):
    img_arr = np.array(Image.open(png_path).convert("RGB"))
    img_arr = img_arr[
        crop_box.crop_top : crop_box.crop_bot,
        crop_box.crop_left : crop_box.crop_right,
        :,
    ]
    return np.array(Image.fromarray(img_arr).resize((out_w, out_h), LANCZOS))
