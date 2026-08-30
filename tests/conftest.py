import numpy as np
import pytest
import SimpleITK as sitk
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


@pytest.fixture
def donor_dicom_path(tmp_path):
    """A minimal-but-valid single-instance MR donor DICOM with the tags this
    tool reads (patient/study identity, dates/times, Frame of Reference)."""
    path = tmp_path / "donor.dcm"

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"  # MR Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.PatientName = "TEST^PATIENT"
    ds.PatientID = "DONOR001"
    ds.PatientBirthDate = "19800101"
    ds.PatientSex = "M"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyID = "1"
    ds.FrameOfReferenceUID = generate_uid()
    ds.StudyDate = "20260101"
    ds.StudyTime = "120000"
    ds.AccessionNumber = "ACC001"
    ds.Modality = "MR"
    ds.InstitutionName = "Test Hospital"
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = 2
    ds.Columns = 2
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = np.zeros((2, 2), dtype=np.uint16).tobytes()

    ds.save_as(str(path), enforce_file_format=True)
    return str(path)


@pytest.fixture
def small_nifti_path(tmp_path):
    """A small synthetic 3-D NIfTI volume with a simple ramp of values,
    including a masked (zero) background, roughly like a real brain map."""
    path = tmp_path / "volume.nii.gz"
    arr = np.zeros((8, 6, 6), dtype=np.float32)  # (z, y, x)
    arr[2:6, 1:5, 1:5] = np.linspace(0.1, 100.0, num=4 * 4 * 4).reshape(4, 4, 4)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((2.0, 2.0, 2.0))
    sitk.WriteImage(img, str(path))
    return str(path)


@pytest.fixture
def png_slice_dir(tmp_path):
    """A directory of small RGB PNG slices."""
    from PIL import Image

    d = tmp_path / "pngs"
    d.mkdir()
    for i in range(4):
        arr = np.full((10, 10, 3), fill_value=i * 20, dtype=np.uint8)
        Image.fromarray(arr).save(d / f"slice_{i:03d}.png")
    return str(d)
