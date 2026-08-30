import pytest
from nifti2dicom.geometry import (
    axis_offset_and_dir,
    compute_slice_geometry,
    cross3,
    dot3,
    norm3,
    orthonormal_row_col,
)

IDENTITY = (1, 0, 0, 0, 1, 0, 0, 0, 1)  # row-major flattened 3x3 identity


def test_axis_offset_and_dir_no_flip():
    offset, direction = axis_offset_and_dir([1, 0, 0], flip=False, spacing=2.0, size=5)
    assert offset == [0.0, 0.0, 0.0]
    assert direction == [1, 0, 0]


def test_axis_offset_and_dir_flip():
    offset, direction = axis_offset_and_dir([1, 0, 0], flip=True, spacing=2.0, size=5)
    assert offset == pytest.approx([8.0, 0.0, 0.0])  # (5-1)*2.0
    assert direction == [-1, 0, 0]


def test_compute_slice_geometry_invalid_plane_raises():
    geom = ((0, 0, 0), (1, 1, 1), IDENTITY, (4, 4, 4))
    with pytest.raises(ValueError):
        compute_slice_geometry(geom, "AXIAL", 0)


def test_compute_slice_geometry_tra_identity():
    # Identity direction: i_dir=[1,0,0], j_dir=[0,1,0], k_dir=[0,0,1].
    # k flips (k_dir[2] > 0); i and j don't.
    geom = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), IDENTITY, (4, 4, 4))
    pos0, row_dir, col_dir, normal, thick = compute_slice_geometry(geom, "TRA", 0)
    assert pos0 == pytest.approx([0.0, 0.0, 0.0])
    assert row_dir == [1, 0, 0]
    assert col_dir == [0, 1, 0]
    assert normal == [0, 0, 1]
    assert thick == 1.0

    pos1, *_ = compute_slice_geometry(geom, "TRA", 1)
    assert pos1 == pytest.approx([0.0, 0.0, 1.0])


def test_compute_slice_geometry_cor_identity_reflects_k_flip():
    geom = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), IDENTITY, (4, 4, 4))
    pos0, row_dir, col_dir, normal, thick = compute_slice_geometry(geom, "COR", 0)
    # k (col axis here) flips: offset lands at (N_k-1)*spacing = 3.0 along z,
    # and the displayed column direction is reversed.
    assert pos0 == pytest.approx([0.0, 0.0, 3.0])
    assert row_dir == [1, 0, 0]
    assert col_dir == [0, 0, -1]
    assert normal == [0, 1, 0]
    assert thick == 1.0

    pos1, *_ = compute_slice_geometry(geom, "COR", 1)
    assert pos1 == pytest.approx([0.0, 1.0, 3.0])


def test_compute_slice_geometry_sag_identity():
    geom = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), IDENTITY, (4, 4, 4))
    pos0, row_dir, col_dir, normal, thick = compute_slice_geometry(geom, "SAG", 0)
    assert pos0 == pytest.approx([0.0, 0.0, 3.0])
    assert row_dir == [0, 1, 0]
    assert col_dir == [0, 0, -1]
    assert normal == [1, 0, 0]
    assert thick == 1.0

    pos1, *_ = compute_slice_geometry(geom, "SAG", 1)
    assert pos1 == pytest.approx([1.0, 0.0, 3.0])


def test_compute_slice_geometry_flips_with_mirrored_i_axis():
    # i_dir[0] < 0 now, so i flips.
    mirrored = (-1, 0, 0, 0, 1, 0, 0, 0, 1)
    geom = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), mirrored, (4, 4, 4))
    pos0, row_dir, _col_dir, _normal, _thick = compute_slice_geometry(geom, "TRA", 0)
    assert row_dir == [1, 0, 0]  # -(-1) = 1, direction is negated
    # offset = (N_i - 1) * spacing * i_dir, and i_dir itself is [-1, 0, 0]
    assert pos0 == pytest.approx([-3.0, 0.0, 0.0])


def test_compute_slice_geometry_respects_nonunit_spacing_and_origin():
    geom = ((10.0, -5.0, 2.0), (0.5, 2.0, 1.0), IDENTITY, (4, 4, 4))
    pos, _row_dir, _col_dir, _normal, thick = compute_slice_geometry(geom, "TRA", 2)
    # k flips: offset = (4-1)*1.0 = 3.0 along z contributes nothing to TRA's
    # slice_off (TRA uses raw k_dir, not the flipped k_off) -- slice position
    # along z is origin.z + spacing_z * slice_idx.
    assert pos == pytest.approx([10.0, -5.0, 2.0 + 1.0 * 2])
    assert thick == 1.0


def test_norm3():
    assert norm3([3, 0, 0]) == pytest.approx([1, 0, 0])
    assert norm3([0, 3, 4]) == pytest.approx([0, 0.6, 0.8])


def test_norm3_zero_vector_raises():
    with pytest.raises(ValueError):
        norm3([0, 0, 0])


def test_dot3_and_cross3():
    assert dot3([1, 0, 0], [0, 1, 0]) == 0
    assert dot3([1, 2, 3], [1, 2, 3]) == 14
    assert cross3([1, 0, 0], [0, 1, 0]) == pytest.approx([0, 0, 1])


def test_orthonormal_row_col_already_orthogonal():
    row_dir, col_dir, slice_dir = orthonormal_row_col([1, 0, 0], [0, 1, 0])
    assert row_dir == pytest.approx([1, 0, 0])
    assert col_dir == pytest.approx([0, 1, 0])
    assert slice_dir == pytest.approx([0, 0, 1])


def test_orthonormal_row_col_removes_skew():
    # col_raw has a component along row_dir; Gram-Schmidt should remove it.
    row_dir, col_dir, _slice_dir = orthonormal_row_col([1, 0, 0], [0.5, 1, 0])
    assert row_dir == pytest.approx([1, 0, 0])
    assert col_dir == pytest.approx([0, 1, 0])
    assert dot3(row_dir, col_dir) == pytest.approx(0.0, abs=1e-9)
