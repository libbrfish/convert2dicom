"""UID generation and Study Instance UID resolution, shared by the
donor-match SAG path, the TRA/COR PNG path, and the standard NIfTI/TIFF path.

The org root "1.2.826.0.1.3680043.2.1125" is KU Leuven's DICOM UID root,
kept unchanged from the original script.
"""

import hashlib

_ORG_ROOT = "1.2.826.0.1.3680043.2.1125"


def series_desc_hash(seriesdesc):
    """A stable integer derived from the series description, folded into
    generated UIDs so re-running with the same description is reproducible
    rather than colliding by pure chance with a different one."""
    return int(hashlib.md5(seriesdesc.encode()).hexdigest()[:8], 16)


def make_series_instance_uid(mod_date, mod_time, seriesdesc):
    return f"{_ORG_ROOT}.{mod_date}.1{mod_time}.{series_desc_hash(seriesdesc)}"


def make_sop_instance_uid(mod_date, mod_time, seriesdesc, index):
    """SOP Instance UID for slice `index`, used by the two PNG-screenshot
    write paths (donor-match SAG, TRA/COR)."""
    return f"{_ORG_ROOT}.{mod_date}{mod_time}.{series_desc_hash(seriesdesc)}.{index}"


def resolve_study_uid(get_tag, mod_date, mod_time, on_warning=None):
    """Return the donor's Study Instance UID (0020,000D), trying both key
    spellings, or generate a fallback if the donor has none.

    `get_tag(key)` should return the decoded tag string for a given tag key
    (e.g. "0020|000d") or None if absent -- this indirection is what makes
    the function testable without a real SimpleITK reader.

    `on_warning`, if given, is called with a message string when a UID has to
    be generated (the donor lacked one). Consolidated here to always warn:
    the original script warned only on the plain NIfTI/TIFF path and stayed
    silent about the exact same situation on the two PNG paths.
    """
    for key in ("0020|000d", "0020|000D"):
        val = get_tag(key)
        if val:
            val = val.strip()
            if val:
                return val
    generated = f"{_ORG_ROOT}.{mod_date}.2{mod_time}"
    if on_warning:
        on_warning("donor has no StudyInstanceUID; generating one")
    return generated
