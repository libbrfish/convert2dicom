"""Writes one 2-D slice of a 3-D SimpleITK image out as a single DICOM file.

This is the one piece of the pipeline that legitimately needs SimpleITK for
its own sake (SetMetaData / ImageFileWriter), so it isn't a pure function --
but the geometry math it calls into (geometry.compute_slice_geometry) is
pure and independently tested.
"""

import os
import time

from .formatting import ds
from .geometry import compute_slice_geometry


def write_slice(
    series_tag_values,
    new_img,
    out_dir,
    i,
    writer,
    underlay_geom=None,
    plane=None,
    series_uid=None,
):
    """Write slice index `i` of `new_img` (a SimpleITK image with depth >= i)
    to `out_dir` as a zero-padded-numbered .dcm file.

    If `underlay_geom` and `plane` are both given, per-slice geometry
    (IPP/IOP/SliceLocation/Thickness) is computed from that underlay using
    mrview's screenshot display convention (see geometry.compute_slice_geometry)
    -- the correct path when `new_img` is a rendered PNG screenshot stack.
    Otherwise geometry is read directly from `new_img`'s own direction matrix
    -- the correct path for actual voxel data (NIfTI/TIFF).
    """
    image_slice = new_img[:, :, i]

    for tag, value in series_tag_values:
        image_slice.SetMetaData(tag, value)

    # Slice-specific date/time
    image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
    image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))
    # Instance Number is 1-based by convention; 0 makes some viewers mislabel or
    # mis-sort the first slice.
    image_slice.SetMetaData("0020|0013", str(i + 1))

    # SOP Instance UID, derived from the series UID so it is unique and stable.
    # writer.KeepOriginalImageUIDOn() has nothing to keep on a freshly created
    # image, and the writer's own generation is not guaranteed unique across a
    # tight loop of many slices -- duplicates make PACS silently drop slices.
    if series_uid is not None:
        image_slice.SetMetaData("0008|0018", f"{series_uid}.{i + 1}")

    if underlay_geom is not None and plane is not None:
        pos, row_dir, col_dir, normal, thick = compute_slice_geometry(
            underlay_geom, plane, i
        )
        image_slice.SetMetaData("0020|0032", "\\".join(ds(v) for v in pos))
        image_slice.SetMetaData(
            "0020|0037", "\\".join(ds(v) for v in row_dir + col_dir)
        )
        slice_loc = sum(pos[x] * normal[x] for x in range(3))
        image_slice.SetMetaData("0020|1041", ds(slice_loc))
        image_slice.SetMetaData("0018|0050", ds(thick))  # Slice Thickness
    else:
        # Geometry straight from the volume's own direction matrix. This is the
        # correct path for actual voxel data: compute_slice_geometry reproduces
        # mrview's *display* flip convention, which is right for screenshots
        # and wrong for a scalar map.
        sp = new_img.GetSpacing()
        d = new_img.GetDirection()
        row_dir = (d[0], d[3], d[6])  # direction of increasing column index
        col_dir = (d[1], d[4], d[7])  # direction of increasing row index
        normal = (d[2], d[5], d[8])
        pos = new_img.TransformIndexToPhysicalPoint((0, 0, i))
        image_slice.SetMetaData("0020|0032", "\\".join(ds(v) for v in pos))
        image_slice.SetMetaData(
            "0020|0037", "\\".join(ds(v) for v in list(row_dir) + list(col_dir))
        )
        image_slice.SetMetaData(
            "0020|1041", ds(sum(pos[x] * normal[x] for x in range(3)))
        )
        # PixelSpacing is [between rows, between columns] = [dy, dx], the
        # opposite order to SimpleITK's (x, y, z) spacing.
        image_slice.SetMetaData("0028|0030", f"{ds(sp[1])}\\{ds(sp[0])}")
        image_slice.SetMetaData("0018|0050", ds(sp[2]))  # Slice Thickness
        image_slice.SetMetaData("0018|0088", ds(sp[2]))  # Spacing Between Slices

    writer.SetFileName(os.path.join(out_dir, str(i).rjust(6, "0") + ".dcm"))
    writer.Execute(image_slice)
