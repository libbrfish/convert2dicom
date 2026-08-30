"""Thin executable entry point -- keeps `python convert2dicom_convert.py ...`
working exactly like the original single-file script, while all real logic
now lives in the importable `convert2dicom` package (see README.md).
"""

import sys

from convert2dicom.cli import main

if __name__ == "__main__":
    sys.exit(main())
