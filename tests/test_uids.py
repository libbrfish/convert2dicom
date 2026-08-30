from nifti2dicom.uids import (
    make_series_instance_uid,
    make_sop_instance_uid,
    resolve_study_uid,
    series_desc_hash,
)


def test_series_desc_hash_is_deterministic():
    assert series_desc_hash("rCBV map") == series_desc_hash("rCBV map")


def test_series_desc_hash_differs_for_different_descriptions():
    assert series_desc_hash("rCBV map") != series_desc_hash("MTT map")


def test_make_series_instance_uid_format():
    uid = make_series_instance_uid("20260829", "120000", "rCBV map")
    assert uid.startswith("1.2.826.0.1.3680043.2.1125.20260829.1120000.")
    assert all(c.isdigit() or c == "." for c in uid)


def test_make_sop_instance_uid_includes_index():
    uid0 = make_sop_instance_uid("20260829", "120000", "rCBV map", 0)
    uid1 = make_sop_instance_uid("20260829", "120000", "rCBV map", 1)
    assert uid0 != uid1
    assert uid0.endswith(".0")
    assert uid1.endswith(".1")


def test_resolve_study_uid_prefers_donor_lowercase_key():
    def get_tag(key):
        return {"0020|000d": "1.2.3.4.5"}.get(key)

    assert resolve_study_uid(get_tag, "20260829", "120000") == "1.2.3.4.5"


def test_resolve_study_uid_falls_back_to_uppercase_key():
    def get_tag(key):
        return {"0020|000D": "1.2.3.4.5"}.get(key)

    assert resolve_study_uid(get_tag, "20260829", "120000") == "1.2.3.4.5"


def test_resolve_study_uid_generates_and_warns_when_absent():
    warnings = []
    uid = resolve_study_uid(
        lambda key: None, "20260829", "120000", on_warning=warnings.append
    )
    assert uid.startswith("1.2.826.0.1.3680043.2.1125.20260829.2120000")
    assert len(warnings) == 1
    assert "StudyInstanceUID" in warnings[0]


def test_resolve_study_uid_treats_blank_value_as_absent():
    warnings = []
    uid = resolve_study_uid(
        lambda key: "   ", "20260829", "120000", on_warning=warnings.append
    )
    assert len(warnings) == 1
    assert uid != "   "
