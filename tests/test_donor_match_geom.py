import pytest

from convert2dicom.donor_match_geom import compute_donor_frame_locs, compute_png_locs


def test_compute_donor_frame_locs_basic():
    frames = compute_donor_frame_locs(
        n_frames=3, origin=(0.0, 0.0, 0.0), slice_dir=(1.0, 0.0, 0.0), slice_spacing=2.0
    )
    assert len(frames) == 3
    locs = [f["loc"] for f in frames]
    assert locs == pytest.approx([0.0, 2.0, 4.0])
    assert frames[1]["ipp"] == pytest.approx([2.0, 0.0, 0.0])


def test_compute_donor_frame_locs_sorted_even_if_direction_negative():
    # Negative slice_dir means physical location decreases with frame index;
    # result must still come back sorted ascending by location.
    frames = compute_donor_frame_locs(
        n_frames=3,
        origin=(10.0, 0.0, 0.0),
        slice_dir=(-1.0, 0.0, 0.0),
        slice_spacing=2.0,
    )
    locs = [f["loc"] for f in frames]
    assert locs == sorted(locs)


def test_compute_png_locs_basic_ordering():
    # underlay i axis aligned with donor slice normal, positive step
    table = compute_png_locs(
        n_pngs=4,
        underlay_origin=(0.0, 0.0, 0.0),
        underlay_i_dir=(1.0, 0.0, 0.0),
        underlay_spacing_i=1.0,
        donor_slice_dir=(1.0, 0.0, 0.0),
    )
    # indices 0..3, ascending locations 0,1,2,3
    assert [idx for idx, loc in table] == [0, 1, 2, 3]
    assert [loc for idx, loc in table] == pytest.approx([0.0, 1.0, 2.0, 3.0])


def test_compute_png_locs_reverses_when_step_negative():
    table = compute_png_locs(
        n_pngs=4,
        underlay_origin=(0.0, 0.0, 0.0),
        underlay_i_dir=(-1.0, 0.0, 0.0),
        underlay_spacing_i=1.0,
        donor_slice_dir=(1.0, 0.0, 0.0),
    )
    # loc_step negative -> underlay index order is reversed in sorted output
    assert [idx for idx, loc in table] == [3, 2, 1, 0]


def test_compute_png_locs_length_matches_n_pngs():
    table = compute_png_locs(5, (0, 0, 0), (0, 1, 0), 2.0, (0, 1, 0))
    assert len(table) == 5
