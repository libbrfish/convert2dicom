from nifti2dicom.series_tags import (
    MR_SOP_CLASS,
    SC_SOP_CLASS,
    build_common_series_tags,
    copy_donor_tags,
    sop_class_for,
)


def test_copy_donor_tags_omits_absent_tags():
    donor = {"0010|0010": "DOE^JANE", "0010|0020": "12345"}
    result = copy_donor_tags(lambda k: donor.get(k))
    assert ("0010|0010", "DOE^JANE") in result
    assert ("0010|0020", "12345") in result
    assert not any(k == "0008|0080" for k, v in result)  # Institution Name absent


def test_copy_donor_tags_preserves_order():
    from nifti2dicom.series_tags import TAGS_TO_COPY

    donor = {k: f"val-{i}" for i, k in enumerate(TAGS_TO_COPY)}
    result = copy_donor_tags(lambda k: donor.get(k))
    assert [k for k, v in result] == TAGS_TO_COPY


def test_sop_class_for_quantitative_vs_secondary_capture():
    assert sop_class_for(True) == MR_SOP_CLASS
    assert sop_class_for(False) == SC_SOP_CLASS


def test_build_common_series_tags_contents():
    tags = dict(
        build_common_series_tags(
            SC_SOP_CLASS,
            "20260829",
            "120000",
            "study-uid",
            "series-uid",
            "My Series",
            "5",
        )
    )
    assert tags["0002|0002"] == SC_SOP_CLASS
    assert tags["0008|0016"] == SC_SOP_CLASS
    assert tags["0020|000d"] == "study-uid"
    assert tags["0020|000e"] == "series-uid"
    assert tags["0008|103e"] == "My Series"
    assert tags["0020|0011"] == "5"
    assert tags["0008|0008"] == "DERIVED\\SECONDARY"
