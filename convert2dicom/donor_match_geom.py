"""Pure slice-location bookkeeping for donor-match SAG mode: matching each
rendered PNG to the spatially closest donor frame by projecting positions
onto the donor's own slice-normal axis.
"""


def compute_donor_frame_locs(n_frames, origin, slice_dir, slice_spacing):
    """Per-frame (IPP, SliceLocation) for a donor volume, sorted by location.

    `origin` is the donor's image origin (3-tuple), `slice_dir` its
    (normalized) slice-normal direction, `slice_spacing` the spacing along
    that normal. Returns a list of {'ipp': [x,y,z], 'loc': float}, ascending
    by 'loc'.
    """
    frames = []
    for j in range(n_frames):
        ipp = [origin[x] + slice_dir[x] * slice_spacing * j for x in range(3)]
        loc = sum(ipp[x] * slice_dir[x] for x in range(3))
        frames.append({"ipp": ipp, "loc": loc})
    frames.sort(key=lambda f: f["loc"])
    return frames


def compute_png_locs(
    n_pngs, underlay_origin, underlay_i_dir, underlay_spacing_i, donor_slice_dir
):
    """Slice location (projected onto the donor's slice-normal axis) for each
    of `n_pngs` underlay slices, in underlay slice order.

    Returns a list of (index, location) sorted ascending by location -- the
    order to write PNGs in so they land on the correct matched donor frame.
    """
    base_loc = sum(underlay_origin[x] * donor_slice_dir[x] for x in range(3))
    loc_step = (
        sum(underlay_i_dir[x] * donor_slice_dir[x] for x in range(3))
        * underlay_spacing_i
    )
    table = [(i, base_loc + loc_step * i) for i in range(n_pngs)]
    table.sort(key=lambda t: t[1])
    return table
