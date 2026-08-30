from nifti2dicom.charset import dcm_str, resolve_codec, resolve_codec_from_term


class FakeReader:
    """Minimal stand-in for SimpleITK.ImageFileReader's metadata interface."""

    def __init__(self, tags):
        self._tags = tags

    def HasMetaDataKey(self, key):
        return key in self._tags

    def GetMetaData(self, key):
        return self._tags[key]


def test_resolve_codec_from_term_known_sets():
    assert resolve_codec_from_term("ISO_IR 100") == "latin-1"
    assert resolve_codec_from_term("ISO_IR 192") == "utf-8"
    assert resolve_codec_from_term("ISO_IR 144") == "iso8859-5"
    assert resolve_codec_from_term("") == "ascii"


def test_resolve_codec_from_term_code_extension_spelling():
    # "ISO 2022 IR nnn" (code-extension spelling) normalises to "ISO_IR nnn"
    assert resolve_codec_from_term("ISO 2022 IR 100") == "latin-1"


def test_resolve_codec_from_term_multivalued_takes_first():
    assert resolve_codec_from_term("ISO_IR 100\\ISO_IR 192") == "latin-1"


def test_resolve_codec_from_term_unknown_returns_none():
    assert resolve_codec_from_term("ISO_IR 87") is None  # Japanese, unsupported
    assert resolve_codec_from_term("nonsense") is None


def test_resolve_codec_defaults_to_latin1_when_tag_absent():
    reader = FakeReader({})
    assert resolve_codec(reader) == "latin-1"


def test_resolve_codec_defaults_to_latin1_for_unsupported_multibyte():
    reader = FakeReader({"0008|0005": "ISO_IR 87"})
    assert resolve_codec(reader) == "latin-1"


def test_resolve_codec_reads_declared_charset():
    reader = FakeReader({"0008|0005": "ISO_IR 192"})
    assert resolve_codec(reader) == "utf-8"


def test_resolve_codec_handles_bytes_value():
    reader = FakeReader({"0008|0005": b"ISO_IR 100"})
    assert resolve_codec(reader) == "latin-1"


def test_dcm_str_none_returns_empty():
    assert dcm_str(None) == ""


def test_dcm_str_bytes_decoded_with_codec():
    name = "Hôpital".encode("latin-1")
    assert dcm_str(name, "latin-1") == "Hôpital"


def test_dcm_str_strips_nul_padding():
    assert dcm_str(b"SMITH^JOHN\x00", "ascii") == "SMITH^JOHN"


def test_dcm_str_clean_str_passthrough_utf8():
    # Already-clean str (e.g. correctly-decoded UTF-8 Arabic) must not be
    # re-decoded with a single-byte codec -- that would corrupt it.
    s = "محمد"
    assert dcm_str(s, "latin-1") == s


def test_dcm_str_recovers_surrogate_escaped_bytes():
    # Simulate what newer SimpleITK returns for undecodable bytes: a str with
    # surrogate escapes standing in for the raw byte values.
    original_bytes = "Hôpital".encode("latin-1")
    surrogate_str = original_bytes.decode("utf-8", "surrogateescape")
    assert dcm_str(surrogate_str, "latin-1") == "Hôpital"


def test_dcm_str_never_raises_on_garbage():
    # Undecodable bytes for the given codec fall back to replacement chars
    # rather than raising.
    garbage = b"\xff\xfe\x00"
    result = dcm_str(garbage, "ascii")
    assert isinstance(result, str)
