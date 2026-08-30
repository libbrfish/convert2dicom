"""Per-slice DICOM geometry (IPP/IOP/SliceLocation/Thickness) derived from a
volume's origin/spacing/direction/size, reproducing mrview's fixed anatomical
display convention for rendered-screenshot PNG input.

All functions here are pure: they take plain tuples/lists in and return plain
tuples/lists out, so they can be unit tested without SimpleITK.

A "geometry" tuple is (origin, spacing, direction, size):
  origin:    (ox, oy, oz)
  spacing:   (sx, sy, sz)
  direction: 9-tuple, row-major 3x3 direction cosine matrix
  size:      (Ni, Nj, Nk) voxel counts
"""


def axis_offset_and_dir(vec, flip, spacing, size):
    """Return (corner_offset, display_dir) for one in-plane voxel axis.

    `flip` says whether pixel index 0 along this screen axis sits at voxel
    index N-1 rather than 0 (i.e. whether display order is reversed relative
    to voxel-index order for this axis).
    """
    if flip:
        offset = [(size - 1) * spacing * v for v in vec]
        direction = [-v for v in vec]
    else:
        offset = [0.0, 0.0, 0.0]
        direction = list(vec)
    return offset, direction


def compute_slice_geometry(underlay_geom, plane, slice_idx):
    """Return (image_position, row_dir, col_dir, slice_normal, slice_thickness)
    for a given slice index in a given orthogonal plane.

    mrview renders with a fixed anatomical display convention (superior/anterior
    up, patient-right on the left) regardless of how a volume's own voxel axes
    are signed or how oblique its direction matrix is. Per axis, whether pixel
    index 0 sits at voxel index 0 or N-1 depends only on that axis's own
    dominant anatomical sign — verified empirically against mrview by rendering
    synthetic phantoms with anisotropic spacing, oblique (gantry-tilt-like)
    rotation, and each axis's sign independently flipped:
      i (L-R):  flips when i_dir[0] < 0
      j (A-P):  flips when j_dir[1] < 0
      k (S-I):  flips when k_dir[2] > 0
    This one rule holds unchanged across TRA/COR/SAG since each plane just
    picks 2 of the 3 axes for its row/col roles and the third as its normal.
    """
    origin, spacing, d, sz = underlay_geom
    i_dir = [d[0], d[3], d[6]]
    j_dir = [d[1], d[4], d[7]]
    k_dir = [d[2], d[5], d[8]]
    N_i, N_j, N_k = sz[0], sz[1], sz[2]

    i_off, i_disp = axis_offset_and_dir(i_dir, i_dir[0] < 0, spacing[0], N_i)
    j_off, j_disp = axis_offset_and_dir(j_dir, j_dir[1] < 0, spacing[1], N_j)
    k_off, k_disp = axis_offset_and_dir(k_dir, k_dir[2] > 0, spacing[2], N_k)

    if plane == "TRA":
        row_off, row_dir = i_off, i_disp
        col_off, col_dir = j_off, j_disp
        slice_off = [spacing[2] * slice_idx * v for v in k_dir]
        normal, thick = list(k_dir), spacing[2]
    elif plane == "COR":
        row_off, row_dir = i_off, i_disp
        col_off, col_dir = k_off, k_disp
        slice_off = [spacing[1] * slice_idx * v for v in j_dir]
        normal, thick = list(j_dir), spacing[1]
    elif plane == "SAG":
        row_off, row_dir = j_off, j_disp
        col_off, col_dir = k_off, k_disp
        slice_off = [spacing[0] * slice_idx * v for v in i_dir]
        normal, thick = list(i_dir), spacing[0]
    else:
        raise ValueError(f"Plane {plane} is invalid.")

    position = [origin[x] + row_off[x] + col_off[x] + slice_off[x] for x in range(3)]
    return position, row_dir, col_dir, normal, thick


def norm3(v):
    """Normalize a 3-vector. Raises ValueError on a degenerate (zero) vector
    instead of dividing by zero, so callers can produce an actionable error."""
    n = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
    if n == 0:
        raise ValueError("cannot normalize a zero-length vector")
    return [x / n for x in v]


def dot3(a, b):
    return sum(a[i] * b[i] for i in range(3))


def cross3(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def orthonormal_row_col(row_raw, col_raw):
    """Gram-Schmidt: normalize `row_raw`, then orthogonalize+normalize
    `col_raw` against it. Returns (row_dir, col_dir, slice_dir) where
    slice_dir = row_dir x col_dir.

    Used to recover a clean orthonormal basis from a donor's direction matrix
    (which may carry small numerical noise) for the donor-match SAG path.
    """
    row_dir = norm3(row_raw)
    col_orth = [col_raw[i] - dot3(row_dir, col_raw) * row_dir[i] for i in range(3)]
    col_dir = norm3(col_orth)
    slice_dir = cross3(row_dir, col_dir)
    return row_dir, col_dir, slice_dir
