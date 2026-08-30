"""Series-level DICOM tag assembly shared by all three write paths.

Two SOP Classes are used, both deliberately NOT copied from the donor:
  * Secondary Capture Image Storage -- what rendered-PNG-screenshot output
    actually is, regardless of the donor's own (real-acquisition) SOP Class.
  * MR Image Storage -- used only for -q/--label quantitative series, because
    Secondary Capture has no Modality LUT module and PACS ROI tools would
    ignore RescaleSlope/RescaleIntercept on it.

Copying the donor's own SOP Class (e.g. MR Image Storage) onto RGB pixel data
makes GDCM's DICOM writer silently re-derive/reset Image Orientation Patient
and Pixel Spacing from the image's own (default identity) geometry, discarding
whatever was explicitly set via SetMetaData -- hence the exclusion.
"""

SC_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture Image Storage
MR_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.4"  # MR Image Storage (quantitative only)

# Patient/study identity + acquisition context copied verbatim from the donor.
# Study/Series/SOP UIDs are NOT in this list -- they're set explicitly per
# output series/instance, never copied from the donor.
TAGS_TO_COPY = [
    "0010|0010",  # Patient Name
    "0010|0020",  # Patient ID
    "0010|0030",  # Patient Birth Date
    "0010|0040",  # Patient Sex
    "0020|0010",  # Study ID
    "0020|0052",  # Frame of Reference UID -- links series for cursor alignment in PACS
    "0008|0020",  # Study Date
    "0008|0022",  # Acquisition Date
    "0008|0023",  # Content Date
    "0008|0030",  # Study Time
    "0008|0032",  # Acquisition Time
    "0008|0033",  # Content Time
    "0008|0050",  # Accession Number
    "0008|0060",  # Modality
    "0008|0080",  # Institution Name
]


def copy_donor_tags(get_tag, tags=TAGS_TO_COPY):
    """Build the list of (tag, value) pairs to copy from the donor.

    `get_tag(key)` returns the decoded string value for `key`, or None if the
    donor doesn't have it -- absent tags are simply omitted, matching the
    original script's `if reader.HasMetaDataKey(k)` filter.
    """
    out = []
    for k in tags:
        v = get_tag(k)
        if v is not None:
            out.append((k, v))
    return out


def build_common_series_tags(
    sop_class, mod_date, mod_time, study_uid, series_uid, seriesdesc, seriesnumber
):
    """The series-level tags common to all three write paths."""
    return [
        ("0002|0002", sop_class),
        ("0008|0016", sop_class),
        ("0008|0031", mod_time),
        ("0008|0021", mod_date),
        ("0008|0008", "DERIVED\\SECONDARY"),
        ("0020|000d", study_uid),
        ("0020|000e", series_uid),
        ("0008|103e", seriesdesc),
        ("0020|0011", seriesnumber),
    ]


def sop_class_for(is_quantitative):
    """MR Image Storage for a quantitative series, Secondary Capture otherwise.

    A measurable series is an MR image, not a screenshot: Secondary Capture
    has no Modality LUT module, so PACS would ignore RescaleSlope/Intercept
    and report raw stored values in an ROI.
    """
    return MR_SOP_CLASS if is_quantitative else SC_SOP_CLASS
