"""Geometric crop/resize math for rendered PNG screenshots (mrview captures).

Pixel-content thresholds are unreliable for finding the true anatomy region
inside a rendered screenshot: both the rendering margin and the dark brain
exterior are near-zero with a black background (or both near-white with
white). Instead the crop is derived purely from geometry: the underlay's
field of view, and mrview's fixed single-zoom-for-the-whole-volume rendering
convention (based on the single largest physical extent across all 3 volume
axes, not just the 2 in-plane dimensions of the current view). This has been
verified empirically against mrview with synthetic phantoms.

This module contains only the pure arithmetic; PNG loading/cropping/resizing
with PIL stays in the calling pipeline code.
"""

from collections import namedtuple

CropBox = namedtuple(
    "CropBox",
    [
        "crop_left",
        "crop_right",
        "crop_top",
        "crop_bot",
        "scale",
        "content_w",
        "content_h",
    ],
)


def global_max_fov(size, spacing):
    """Largest physical extent (mm) across all 3 volume axes.

    `size` and `spacing` are each 3-tuples (Ni, Nj, Nk) / (sx, sy, sz).
    """
    return max(size[i] * spacing[i] for i in range(3))


def compute_crop_box(png_w, png_h, h_fov, v_fov, global_max_fov_mm):
    """Compute the pixel crop box that isolates the rendered anatomy within a
    PNG screenshot, given the in-plane field of view (h_fov x v_fov, mm) and
    the volume-wide max FOV that determines mrview's fixed zoom.

    Returns a CropBox. `crop_right`/`crop_bot` are exclusive end indices,
    suitable for direct numpy/PIL slicing: arr[crop_top:crop_bot, crop_left:crop_right].
    """
    scale = min(png_w, png_h) / global_max_fov_mm
    content_w = round(h_fov * scale)
    content_h = round(v_fov * scale)
    crop_left = (png_w - content_w) // 2
    crop_right = crop_left + content_w
    crop_top = (png_h - content_h) // 2
    crop_bot = crop_top + content_h
    return CropBox(
        crop_left, crop_right, crop_top, crop_bot, scale, content_w, content_h
    )
