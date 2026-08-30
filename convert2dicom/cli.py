"""Command-line entry point: convert a NIfTI, 3D-TIFF, or PNG-slice directory
to DICOM given a donor DICOM image.

This module only does argument parsing + orchestration; all actual logic
lives in the other, independently-tested modules of this package.
"""

import argparse
import os
import shutil
import sys
import time

import SimpleITK as sitk

from .crop import compute_crop_box
from .donor import DonorHeader
from .geometry import compute_slice_geometry
from .modality_lut import attach_modality_lut
from .pipeline_donor_match import (
    DonorMatchError,
    build_frame_and_png_tables,
    compute_output_geometry,
    load_donor_3d,
)
from .pipeline_donor_match import (
    crop_and_resize_png as crop_and_resize_png_sag,
)
from .pipeline_donor_match import (
    list_pngs as list_pngs_sag,
)
from .pipeline_standard import (
    UnsupportedDimensionError,
    apply_pixel_spacing_override,
    convert_nifti_to_int16,
    load_png_dir_as_volume,
)
from .pipeline_tra_cor import (
    crop_and_resize_png as crop_and_resize_png_tc,
)
from .pipeline_tra_cor import (
    list_pngs as list_pngs_tc,
)
from .pipeline_tra_cor import (
    plane_output_geometry,
)
from .rescale import RescaleError, compute_window
from .series_tags import build_common_series_tags, copy_donor_tags, sop_class_for
from .uids import make_series_instance_uid, make_sop_instance_uid, resolve_study_uid
from .writer import write_slice


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Convert a nifti, 3d-tiff, or PNG directory to dicom given a donor dicom image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="increase verbosity"
    )
    parser.add_argument("-s", "--seriesdescription")
    parser.add_argument("-n", "--seriesnumber")
    parser.add_argument(
        "-p",
        "--pixelspacing",
        type=float,
        default=None,
        help="in-plane pixel spacing in mm (required for PNG input; corrects PACS distance measurements)",
    )
    parser.add_argument(
        "-u",
        "--underlay",
        default=None,
        help="NIfTI underlay used for mrview rendering; provides correct spatial metadata for PACS linking and MPR",
    )
    parser.add_argument(
        "-o",
        "--plane",
        choices=["TRA", "SAG", "COR"],
        default=None,
        help="orientation plane of the screenshots (TRA/SAG/COR); required when --underlay is given",
    )
    parser.add_argument(
        "-M",
        "--match-donor",
        action="store_true",
        help="SAG mode: match each PNG to the spatially closest donor frame and copy its exact "
        "IPP/IOP/PixelSpacing — guarantees PACS alignment without any geometry computation",
    )
    parser.add_argument(
        "-q",
        "--quantitative",
        action="store_true",
        help="NIfTI input: write a measurable MR Image Storage series. Voxel values are mapped "
        "to int16 and the mapping is recorded in RescaleSlope/RescaleIntercept, so an ROI "
        "drawn on PACS reads the original units (rCBV, ALFF, ReHo, FA, ...). Without this "
        "the values are rescaled to fill int16 and the scale factor is lost.",
    )
    parser.add_argument(
        "--label",
        action="store_true",
        help="NIfTI input: integer label/segmentation map. Values are written through unchanged "
        "(slope 1, intercept 0) so each label keeps its identity. Implies -q.",
    )
    parser.add_argument(
        "--units",
        default=None,
        help="value units for -q, written to RescaleType (0028,1054), e.g. 'ml/100g' or 'ratio'. "
        "Default 'US' (unspecified).",
    )
    parser.add_argument(
        "--window-headroom",
        type=float,
        default=1.0,
        help="multiply the top of the default display window by this factor (-q only). "
        "1.0 = the plain p99.5 of foreground. Raise it (e.g. 1.3) for maps whose "
        "bright tail is anatomy you want to keep out of saturation, such as the "
        "choroid plexus on a CBV map.",
    )
    parser.add_argument("nifti", help="nifti, 3d-tiff, or directory of PNG slices")
    parser.add_argument("donor", help="dicom donor image")
    parser.add_argument("dicomdir", help="dicom output directory")
    return parser


def classify_input(nifti_input):
    """Return 'png_dir', 'tiff', or 'nifti' for the given input path."""
    if os.path.isdir(nifti_input):
        return "png_dir"
    ext = os.path.splitext(nifti_input)[1].lower()
    if ext in (".tiff", ".tif"):
        return "tiff"
    return "nifti"


def clean_output_dir(dcm_output):
    if os.path.exists(dcm_output):
        shutil.rmtree(dcm_output)
    os.makedirs(dcm_output, exist_ok=True)


def _mod_stamp():
    return time.strftime("%Y%m%d"), time.strftime("%H%M%S")


def _resolve_study_uid_from_donor(donor, mod_date, mod_time, log):
    return resolve_study_uid(donor.get_tag, mod_date, mod_time, on_warning=log)


# --------------------------------------------------------------------------
# Mode: SAG donor-match
# --------------------------------------------------------------------------


def run_donor_match_mode(args, donor, underlay_geom, log=print):
    if underlay_geom is None:
        raise ValueError(
            "donor-match mode requires --underlay for correct SAG orientation"
        )

    log("Donor-match mode: reading donor 3D geometry...")
    donor_3d = load_donor_3d(args.donor)
    donor_geom = (
        donor_3d.GetOrigin(),
        donor_3d.GetSpacing(),
        donor_3d.GetDirection(),
        donor_3d.GetSize(),
    )

    pngs = list_pngs_sag(args.nifti)
    _, out_row_dir, out_col_dir, _, _ = compute_slice_geometry(underlay_geom, "SAG", 0)
    out_iop_str = "\\".join(f"{v:.6f}" for v in out_row_dir + out_col_dir)

    _donor_frames, png_table_locs, n_donor_frames = build_frame_and_png_tables(
        underlay_geom, donor_geom, len(pngs)
    )
    if len(png_table_locs) != n_donor_frames:
        log(
            f"Warning: {len(png_table_locs)} PNGs vs {n_donor_frames} donor frames — "
            f"using nearest-neighbour matching"
        )
    # png_table entries: (original_index, location) sorted by location. Pair
    # with the actual file path by original index.
    png_table = [(idx, loc, pngs[idx]) for idx, loc in png_table_locs]

    geom = compute_output_geometry(underlay_geom)
    sample_img_w, sample_img_h = _png_size(pngs[len(pngs) // 2])
    crop_box = compute_crop_box(
        sample_img_w,
        sample_img_h,
        geom["j_fov"],
        geom["k_fov"],
        geom["global_max_fov_mm"],
    )
    log(
        f"Geometric crop: PNG={sample_img_w}×{sample_img_h}  FOV={geom['j_fov']:.1f}×{geom['k_fov']:.1f}mm  "
        f"scale={crop_box.scale:.3f}px/mm  content={crop_box.content_w}×{crop_box.content_h}px  "
        f"crop cols {crop_box.crop_left}:{crop_box.crop_right} rows {crop_box.crop_top}:{crop_box.crop_bot}  "
        f"→ resizing to {geom['out_w']}×{geom['out_h']} @ {geom['out_sp_x']:.4f}×{geom['out_sp_y']:.4f}mm"
    )

    mod_date, mod_time = _mod_stamp()
    seriesdesc = args.seriesdescription or "IKTsimple - KUL_NIS"
    seriesnumber = args.seriesnumber or ""
    study_uid = _resolve_study_uid_from_donor(donor, mod_date, mod_time, log)
    series_uid = make_series_instance_uid(mod_date, mod_time, seriesdesc)

    series_tags = copy_donor_tags(donor.get_tag) + build_common_series_tags(
        sop_class_for(False),
        mod_date,
        mod_time,
        study_uid,
        series_uid,
        seriesdesc,
        seriesnumber,
    )

    clean_output_dir(args.dicomdir)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    for out_idx, (png_idx, png_loc, png_path) in enumerate(png_table):
        img_resized = crop_and_resize_png_sag(
            png_path, crop_box, geom["out_w"], geom["out_h"]
        )
        position, _, _, normal, _ = compute_slice_geometry(
            underlay_geom, "SAG", png_idx
        )

        slice_2d = sitk.GetImageFromArray(img_resized, isVector=True)
        slice_2d.SetSpacing([geom["out_sp_x"], geom["out_sp_y"]])

        ipp_str = "\\".join(f"{v:.6f}" for v in position)
        slice_loc = sum(position[x] * normal[x] for x in range(3))

        for tag, val in series_tags:
            slice_2d.SetMetaData(tag, val)
        slice_2d.SetMetaData("0020|0037", out_iop_str)
        slice_2d.SetMetaData("0020|0032", ipp_str)
        slice_2d.SetMetaData("0020|1041", f"{slice_loc:.4f}")
        slice_2d.SetMetaData("0018|0050", f"{geom['out_slice_sp']:.4f}")
        slice_2d.SetMetaData("0018|0088", f"{geom['out_slice_sp']:.4f}")
        slice_2d.SetMetaData(
            "0028|0030", f"{geom['out_sp_y']:.6f}\\{geom['out_sp_x']:.6f}"
        )
        slice_2d.SetMetaData("0020|0013", str(out_idx))
        slice_2d.SetMetaData("0008|0012", mod_date)
        slice_2d.SetMetaData("0008|0013", mod_time)
        slice_2d.SetMetaData(
            "0008|0018", make_sop_instance_uid(mod_date, mod_time, seriesdesc, out_idx)
        )

        out_path = os.path.join(args.dicomdir, str(out_idx).rjust(6, "0") + ".dcm")
        writer.SetFileName(out_path)
        writer.Execute(slice_2d)

    log(f"Wrote {len(png_table)} matched SAG DICOMs to {args.dicomdir}")


def _png_size(path):
    from PIL import Image

    with Image.open(path) as im:
        return im.size


# --------------------------------------------------------------------------
# Mode: TRA/COR PNG
# --------------------------------------------------------------------------


def run_tra_cor_mode(args, donor, underlay_geom, log=print):
    pngs = list_pngs_tc(args.nifti)
    geom = plane_output_geometry(underlay_geom, args.plane)
    sample_w, sample_h = _png_size(pngs[len(pngs) // 2])
    crop_box = compute_crop_box(
        sample_w, sample_h, geom["h_fov"], geom["v_fov"], geom["global_max_fov_mm"]
    )
    log(
        f"Geometric crop ({args.plane}): PNG={sample_w}×{sample_h}  FOV={geom['h_fov']:.1f}×{geom['v_fov']:.1f}mm  "
        f"scale={crop_box.scale:.3f}px/mm  content={crop_box.content_w}×{crop_box.content_h}px  "
        f"crop cols {crop_box.crop_left}:{crop_box.crop_right} rows {crop_box.crop_top}:{crop_box.crop_bot}  "
        f"→ resizing to {geom['out_w']}×{geom['out_h']}"
    )

    mod_date, mod_time = _mod_stamp()
    seriesdesc = args.seriesdescription or "IKTsimple - KUL_NIS"
    seriesnumber = args.seriesnumber or ""
    study_uid = _resolve_study_uid_from_donor(donor, mod_date, mod_time, log)
    series_uid = make_series_instance_uid(mod_date, mod_time, seriesdesc)

    series_tags = copy_donor_tags(donor.get_tag) + build_common_series_tags(
        sop_class_for(False),
        mod_date,
        mod_time,
        study_uid,
        series_uid,
        seriesdesc,
        seriesnumber,
    )

    clean_output_dir(args.dicomdir)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    for i, png_path in enumerate(pngs):
        position, row_dir, col_dir, normal, thick = compute_slice_geometry(
            underlay_geom, args.plane, i
        )
        iop_str = "\\".join(f"{v:.6f}" for v in row_dir + col_dir)
        ipp_str = "\\".join(f"{v:.6f}" for v in position)
        slice_loc = sum(position[x] * normal[x] for x in range(3))

        img_resized = crop_and_resize_png_tc(
            png_path, crop_box, geom["out_w"], geom["out_h"]
        )
        slice_img = sitk.GetImageFromArray(img_resized, isVector=True)
        slice_img.SetSpacing([geom["out_sp_x"], geom["out_sp_y"]])

        for tag, val in series_tags:
            slice_img.SetMetaData(tag, val)
        slice_img.SetMetaData("0020|0037", iop_str)
        slice_img.SetMetaData("0020|0032", ipp_str)
        slice_img.SetMetaData("0020|1041", f"{slice_loc:.4f}")
        slice_img.SetMetaData("0018|0050", f"{thick:.4f}")
        slice_img.SetMetaData("0018|0088", f"{thick:.4f}")
        slice_img.SetMetaData(
            "0028|0030", f"{geom['out_sp_y']:.6f}\\{geom['out_sp_x']:.6f}"
        )
        slice_img.SetMetaData("0020|0013", str(i))
        slice_img.SetMetaData("0008|0012", mod_date)
        slice_img.SetMetaData("0008|0013", mod_time)
        slice_img.SetMetaData(
            "0008|0018", make_sop_instance_uid(mod_date, mod_time, seriesdesc, i)
        )

        out_path = os.path.join(args.dicomdir, str(i).rjust(6, "0") + ".dcm")
        writer.SetFileName(out_path)
        writer.Execute(slice_img)

    log(f"Wrote {len(pngs)} {args.plane} DICOMs to {args.dicomdir}")


# --------------------------------------------------------------------------
# Mode: standard (NIfTI / TIFF / plain PNG dir)
# --------------------------------------------------------------------------


def run_standard_mode(args, donor, input_type, underlay_geom, log=print):
    is_label = args.label
    is_quant = args.quantitative or args.label

    if input_type == "png_dir":
        new_img, n = load_png_dir_as_volume(args.nifti)
        log(f"Reading {n} PNG slices from {args.nifti}")
        slope = intercept = None
    elif input_type == "tiff":
        new_img = sitk.ReadImage(args.nifti)
        slope = intercept = None
    else:  # nifti
        nii_img = sitk.ReadImage(args.nifti)
        new_img, slope, intercept, is_label = convert_nifti_to_int16(
            nii_img,
            label=args.label,
            quantitative=args.quantitative,
            window_headroom=args.window_headroom,
            log=log,
        )

    new_img = apply_pixel_spacing_override(new_img, args.pixelspacing, log=log)

    mod_date, mod_time = _mod_stamp()
    seriesdesc = args.seriesdescription or "IKTsimple - KUL_NIS"
    seriesnumber = args.seriesnumber or ""

    if underlay_geom is not None:
        _, row_dir, col_dir, _, _ = compute_slice_geometry(underlay_geom, args.plane, 0)
        orientation_str = "\\".join(f"{v:.6f}" for v in row_dir + col_dir)
    else:
        d = new_img.GetDirection()
        orientation_str = "\\".join(
            f"{v:.6f}" for v in (d[0], d[3], d[6], d[1], d[4], d[7])
        )

    series_uid = make_series_instance_uid(mod_date, mod_time, seriesdesc)
    study_uid = _resolve_study_uid_from_donor(donor, mod_date, mod_time, log)

    is_quant_and_nifti = is_quant and input_type == "nifti"
    sop_class = sop_class_for(is_quant_and_nifti)

    series_tags = copy_donor_tags(donor.get_tag) + build_common_series_tags(
        sop_class, mod_date, mod_time, study_uid, series_uid, seriesdesc, seriesnumber
    )
    series_tags.append(("0020|0037", orientation_str))
    if not is_quant_and_nifti:
        series_tags.append(
            ("0008|0064", "WSD")
        )  # Conversion Type, required for Secondary Capture

    log("Incorporating the following dicom tags:")
    log(series_tags)

    clean_output_dir(args.dicomdir)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    # For a quantitative series the geometry must come from the volume itself,
    # never from compute_slice_geometry() -- that reproduces mrview's
    # screenshot display convention, correct for rendered PNGs and wrong for
    # voxel data.
    geom_for_write = None if is_quant_and_nifti else underlay_geom
    plane_for_write = None if is_quant_and_nifti else args.plane
    if is_quant_and_nifti and underlay_geom is not None:
        log(
            "Note: -u/-o ignored in quantitative mode; geometry comes from the input volume"
        )

    for i in range(new_img.GetDepth()):
        write_slice(
            series_tags,
            new_img,
            args.dicomdir,
            i,
            writer,
            underlay_geom=geom_for_write,
            plane=plane_for_write,
            series_uid=series_uid,
        )

    if is_quant_and_nifti:
        wc, ww = compute_window(
            sitk.GetArrayFromImage(new_img),
            slope,
            intercept,
            is_label=is_label,
            headroom=args.window_headroom,
        )
        units = args.units if args.units else "US"
        attach_modality_lut(args.dicomdir, slope, intercept, units, wc, ww)
        log(
            f"Modality LUT attached: slope={slope:.6g} intercept={intercept:.6g} type={units}"
        )

    log(f"Wrote {new_img.GetDepth()} DICOMs to {args.dicomdir}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not os.path.exists(args.donor):
        print(args.donor + " does not exist")
        return 1
    if not os.path.exists(args.nifti):
        print(args.nifti + " does not exist")
        return 1

    input_type = classify_input(args.nifti)
    if input_type == "png_dir":
        print("Assuming input is a directory of PNG slices")
    elif input_type == "tiff":
        print("Assuming input is a 3d-tiff")
    else:
        print("Assuming input is nifti")

    if args.label:
        args.quantitative = True

    donor = DonorHeader.read(args.donor)
    if args.verbose:
        for k, v in donor.dump():
            print(f'({k}) = "{v}"')

    underlay_geom = None
    if args.underlay is not None and args.plane is not None:
        if not os.path.exists(args.underlay):
            print(f"Warning: underlay {args.underlay} not found.")
            return 1
        ref = sitk.ReadImage(args.underlay)
        underlay_geom = (
            ref.GetOrigin(),
            ref.GetSpacing(),
            ref.GetDirection(),
            ref.GetSize(),
        )
        print(f"Spatial reference: {args.underlay} (plane={args.plane})")

    try:
        if args.match_donor and input_type == "png_dir":
            run_donor_match_mode(args, donor, underlay_geom)
        elif (
            input_type == "png_dir"
            and underlay_geom is not None
            and args.plane in ("TRA", "COR")
        ):
            run_tra_cor_mode(args, donor, underlay_geom)
        else:
            run_standard_mode(args, donor, input_type, underlay_geom)
    except (
        RescaleError,
        UnsupportedDimensionError,
        DonorMatchError,
        FileNotFoundError,
        ValueError,
    ) as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
