import glob
import os

import numpy as np
import pydicom
from convert2dicom.cli import main


def _read_series(dcm_dir):
    files = sorted(glob.glob(os.path.join(dcm_dir, "*.dcm")))
    return files, [pydicom.dcmread(f) for f in files]


def test_legacy_mode_end_to_end(donor_dicom_path, small_nifti_path, tmp_path):
    out_dir = str(tmp_path / "out_legacy")
    rc = main([small_nifti_path, donor_dicom_path, out_dir])
    assert rc == 0

    files, datasets = _read_series(out_dir)
    assert len(files) == 8  # matches the z-depth of small_nifti_path
    ds0 = datasets[0]
    assert ds0.PatientID == "DONOR001"
    assert ds0.SOPClassUID == "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture
    # Legacy mode never calls attach_modality_lut. GDCM's own writer inserts
    # its own default RescaleSlope=1/RescaleIntercept=0/RescaleType='US' for
    # int16 pixel data regardless of what we do (harmless, standard values) --
    # so those aren't a useful signal here. What IS specific to
    # attach_modality_lut is the default display window, which only
    # quantitative mode computes and attaches.
    assert not hasattr(ds0, "WindowCenter")
    assert not hasattr(ds0, "WindowWidth")


def test_quantitative_mode_writes_modality_lut(
    donor_dicom_path, small_nifti_path, tmp_path
):
    out_dir = str(tmp_path / "out_quant")
    rc = main(["-q", "--units", "ml/100g", small_nifti_path, donor_dicom_path, out_dir])
    assert rc == 0

    files, datasets = _read_series(out_dir)
    assert len(files) == 8
    ds0 = datasets[0]
    assert ds0.SOPClassUID == "1.2.840.10008.5.1.4.1.1.4"  # MR Image Storage
    assert float(ds0.RescaleSlope) != 0.0
    assert ds0.RescaleType == "ml/100g"
    assert hasattr(ds0, "WindowCenter")
    assert hasattr(ds0, "WindowWidth")
    # Every slice should carry the same series-level identity.
    study_uids = {d.StudyInstanceUID for d in datasets}
    series_uids = {d.SeriesInstanceUID for d in datasets}
    assert len(study_uids) == 1
    assert len(series_uids) == 1
    # SOP Instance UIDs must all be distinct (regression guard: duplicate
    # SOP UIDs make PACS silently drop slices).
    sop_uids = [d.SOPInstanceUID for d in datasets]
    assert len(sop_uids) == len(set(sop_uids))


def test_label_mode_preserves_integer_values(donor_dicom_path, tmp_path):
    import SimpleITK as sitk

    nifti_path = tmp_path / "labels.nii.gz"
    arr = np.zeros((4, 4, 4), dtype=np.float32)
    arr[1, 1, 1] = 1
    arr[2, 2, 2] = 3
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(img, str(nifti_path))

    out_dir = str(tmp_path / "out_label")
    rc = main(["--label", str(nifti_path), donor_dicom_path, out_dir])
    assert rc == 0

    files, datasets = _read_series(out_dir)
    assert len(files) == 4
    assert float(datasets[0].RescaleSlope) == 1.0
    assert float(datasets[0].RescaleIntercept) == 0.0


def test_4d_nifti_input_is_rejected_cleanly(donor_dicom_path, tmp_path, capsys):
    import SimpleITK as sitk

    nifti_path = tmp_path / "vol4d.nii.gz"
    # isVector=False is required here: with distinct-enough axis sizes
    # SimpleITK's default isVector=None auto-detects a short last axis as
    # color/vector components rather than a true 4th spatial dimension.
    arr = np.zeros((3, 4, 5, 6), dtype=np.float32)
    img = sitk.GetImageFromArray(arr, isVector=False)
    sitk.WriteImage(img, str(nifti_path))

    out_dir = str(tmp_path / "out_4d")
    rc = main([str(nifti_path), donor_dicom_path, out_dir])
    assert rc == 1
    captured = capsys.readouterr()
    assert "only 3-D volumes are supported" in captured.out


def test_all_zero_nifti_legacy_mode_rejected(donor_dicom_path, tmp_path, capsys):
    import SimpleITK as sitk

    nifti_path = tmp_path / "zeros.nii.gz"
    img = sitk.GetImageFromArray(np.zeros((3, 3, 3), dtype=np.float32))
    sitk.WriteImage(img, str(nifti_path))

    out_dir = str(tmp_path / "out_zeros")
    rc = main([str(nifti_path), donor_dicom_path, out_dir])
    assert rc == 1
    captured = capsys.readouterr()
    assert "all zeros" in captured.out


def test_missing_donor_file_reported(tmp_path, small_nifti_path):
    rc = main([small_nifti_path, str(tmp_path / "nope.dcm"), str(tmp_path / "out")])
    assert rc == 1


def test_missing_input_file_reported(donor_dicom_path, tmp_path):
    rc = main([str(tmp_path / "nope.nii.gz"), donor_dicom_path, str(tmp_path / "out")])
    assert rc == 1


def test_png_dir_plain_mode_end_to_end(donor_dicom_path, png_slice_dir, tmp_path):
    # No underlay/plane given: goes through the standard path, geometry taken
    # from the (identity) direction of the freshly-built vector image.
    out_dir = str(tmp_path / "out_png")
    rc = main([png_slice_dir, donor_dicom_path, out_dir])
    assert rc == 0
    files, _datasets = _read_series(out_dir)
    assert len(files) == 4


def test_tra_png_mode_with_underlay(donor_dicom_path, tmp_path):
    import SimpleITK as sitk
    from PIL import Image

    underlay_path = tmp_path / "underlay.nii.gz"
    vol = np.random.rand(8, 6, 6).astype(np.float32)
    uimg = sitk.GetImageFromArray(vol)
    uimg.SetSpacing((2.0, 2.0, 2.0))
    sitk.WriteImage(uimg, str(underlay_path))

    png_dir = tmp_path / "pngs_tra"
    png_dir.mkdir()
    for i in range(8):  # matches underlay z-depth (N_j for TRA)
        arr = np.full((40, 40, 3), fill_value=(i * 10) % 255, dtype=np.uint8)
        Image.fromarray(arr).save(png_dir / f"s{i:03d}.png")

    out_dir = str(tmp_path / "out_tra")
    rc = main(
        ["-u", str(underlay_path), "-o", "TRA", str(png_dir), donor_dicom_path, out_dir]
    )
    assert rc == 0

    files, datasets = _read_series(out_dir)
    assert len(files) == 8
    ds0 = datasets[0]
    assert ds0.SOPClassUID == "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture
    assert ds0.Rows == 6 and ds0.Columns == 6  # underlay's in-plane size, not the PNG's
    # Every slice should have a distinct SOP Instance UID and Image Position.
    sop_uids = [d.SOPInstanceUID for d in datasets]
    assert len(sop_uids) == len(set(sop_uids))
    positions = {tuple(float(v) for v in d.ImagePositionPatient) for d in datasets}
    assert len(positions) == 8


def test_donor_match_sag_mode_runs_end_to_end(donor_dicom_path, tmp_path):
    import SimpleITK as sitk
    from PIL import Image

    underlay_path = tmp_path / "underlay.nii.gz"
    vol = np.random.rand(8, 6, 6).astype(np.float32)
    uimg = sitk.GetImageFromArray(vol)
    uimg.SetSpacing((2.0, 2.0, 2.0))
    sitk.WriteImage(uimg, str(underlay_path))

    png_dir = tmp_path / "pngs_sag"
    png_dir.mkdir()
    for i in range(8):
        arr = np.full((40, 40, 3), fill_value=(i * 15) % 255, dtype=np.uint8)
        Image.fromarray(arr).save(png_dir / f"s{i:03d}.png")

    out_dir = str(tmp_path / "out_sag")
    rc = main(
        [
            "-u",
            str(underlay_path),
            "-o",
            "SAG",
            "-M",
            str(png_dir),
            donor_dicom_path,
            out_dir,
        ]
    )
    assert rc == 0

    files, datasets = _read_series(out_dir)
    assert len(files) == 8
    sop_uids = [d.SOPInstanceUID for d in datasets]
    assert len(sop_uids) == len(set(sop_uids))


def test_match_donor_without_underlay_is_rejected(
    donor_dicom_path, png_slice_dir, tmp_path
):
    # -M requires -u/-o for correct SAG orientation; omitting it must be a
    # clean, reported error rather than a crash.
    out_dir = str(tmp_path / "out_bad")
    rc = main(["-M", png_slice_dir, donor_dicom_path, out_dir])
    assert rc == 1
