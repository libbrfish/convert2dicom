"""Decoding donor DICOM text tags under an arbitrary Specific Character Set.

Split out of the original script unchanged in behavior, just isolated so it
can be unit tested without touching SimpleITK/pydicom I/O.
"""

# DICOM Specific Character Set (0008,0005) defined term -> Python codec.
# Covers the single-byte ISO 8859 families plus UTF-8, which is everything a
# clinical donor realistically uses. Multibyte code-extension sets that need a
# stateful ISO 2022 decoder (Japanese ISO 2022 IR 87/159, Korean IR 149, and
# GB18030 escape sequences) are deliberately NOT claimed here — Arabic does not
# need them (ISO 8859-6 is single-byte; modern Arabic is UTF-8).
DICOM_CHARSET = {
    "": "ascii",
    "ISO_IR 6": "ascii",
    "ISO_IR 100": "latin-1",  # Latin-1  Western European (DICOM default extension)
    "ISO_IR 101": "iso8859-2",  # Latin-2  Central European
    "ISO_IR 109": "iso8859-3",  # Latin-3
    "ISO_IR 110": "iso8859-4",  # Latin-4
    "ISO_IR 144": "iso8859-5",  # Cyrillic
    "ISO_IR 127": "iso8859-6",  # Arabic
    "ISO_IR 126": "iso8859-7",  # Greek
    "ISO_IR 138": "iso8859-8",  # Hebrew
    "ISO_IR 148": "iso8859-9",  # Latin-5  Turkish
    "ISO_IR 192": "utf-8",  # Unicode UTF-8
    "GB18030": "gb18030",
    "GBK": "gbk",
}


def resolve_codec_from_term(term):
    """Map a raw (0008,0005) defined-term string to a Python codec name.

    Pure function: takes the already-extracted term (first component of a
    possibly multi-valued value, already stripped), returns a codec name or
    None if the term maps to a multibyte set we don't handle / is unknown.
    """
    if term is None:
        return None
    term = str(term).split("\\")[0].strip().replace("ISO 2022 IR", "ISO_IR").strip()
    return DICOM_CHARSET.get(term)


def resolve_codec(reader, keys=("0008|0005", "0008|0005 ")):
    """Pick a decode codec from Specific Character Set (0008,0005) on a reader.

    `reader` is anything exposing HasMetaDataKey/GetMetaData (e.g. a
    SimpleITK.ImageFileReader, or a small fake in tests). Falls back to
    Latin-1 (the DICOM default) when the tag is absent or maps to a
    multibyte set we don't handle, so decoding always has a sane single-byte
    codec rather than raising.
    """
    for key in keys:
        if reader.HasMetaDataKey(key):
            term = reader.GetMetaData(key)
            if isinstance(term, bytes):
                term = term.decode("ascii", "ignore")
            codec = resolve_codec_from_term(term)
            if codec:
                return codec
    return "latin-1"


def dcm_str(v, codec="latin-1"):
    """Coerce a donor tag value to a str SimpleITK.SetMetaData will accept.

    A non-UTF-8 tag value (accented Latin name, Cyrillic/Greek/Arabic, ...) surfaces
    differently across SimpleITK builds: older ones return raw ``bytes``; newer ones
    (2.5.x) return a surrogate-escaped ``str`` (e.g. 'H\\udcf4pital'). SetMetaData
    rejects BOTH with 'argument 3 of type std::string const &', so a version bump
    does not fix it — the trigger is the tag content, not the version.

    Three cases:
      * bytes                      -> decode with the donor `codec`.
      * str with surrogate escapes -> SimpleITK could not decode it; recover the
                                      original bytes and decode with `codec`.
      * clean str (no surrogates)  -> SimpleITK already decoded valid UTF-8
                                      correctly (e.g. an ISO_IR 192 Arabic name);
                                      trust it as-is. This passthrough is what keeps
                                      UTF-8 content from being re-decoded — and
                                      corrupted — by a single-byte codec.
    Errors fall back to replacement so it can never raise. NUL pad is stripped.
    """
    if v is None:
        return ""
    if isinstance(v, bytes):
        return v.decode(codec, "replace").rstrip("\x00")
    s = str(v)
    if any(0xD800 <= ord(c) <= 0xDFFF for c in s):
        return (
            s.encode("utf-8", "surrogateescape").decode(codec, "replace").rstrip("\x00")
        )
    return s.rstrip("\x00")
