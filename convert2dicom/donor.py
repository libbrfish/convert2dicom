"""Reads the donor DICOM's header and exposes its tags as plain decoded
strings, insulating the rest of the pipeline from SimpleITK's reader API and
from the bytes-vs-surrogate-escaped-str inconsistency handled in charset.py.
"""

import SimpleITK as sitk

from .charset import dcm_str, resolve_codec


class DonorHeader:
    """Donor DICOM header: tag lookup + the resolved text codec."""

    def __init__(self, reader, charset):
        self._reader = reader
        self.charset = charset

    @classmethod
    def read(cls, donor_path, load_private_tags=True):
        reader = sitk.ImageFileReader()
        reader.SetFileName(donor_path)
        if load_private_tags:
            reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        charset = resolve_codec(reader)
        return cls(reader, charset)

    def get_tag(self, key):
        """Decoded string value for `key`, or None if the donor lacks it."""
        if self._reader.HasMetaDataKey(key):
            return dcm_str(self._reader.GetMetaData(key), self.charset)
        return None

    def keys(self):
        return self._reader.GetMetaDataKeys()

    def dump(self):
        """(key, value) pairs for all present tags, for --verbose logging.
        Mirrors the original script's best-effort try/except per key."""
        out = []
        for k in self.keys():
            if self._reader.HasMetaDataKey(k):
                out.append((k, self._reader.GetMetaData(k)))
        return out
