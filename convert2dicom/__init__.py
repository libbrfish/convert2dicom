"""nifti2dicom: convert NIfTI / 3D-TIFF / PNG-slice-directories to DICOM,
using a donor DICOM for patient/study metadata and (optionally) a NIfTI
underlay for spatially-correct geometry when the input is a directory of
rendered PNG screenshots (e.g. mrview captures).

This package is a refactor of a single monolithic script into small,
independently testable units. See README.md for the module map.
"""
