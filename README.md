# convert2dicom

Converts a NIfTI, 3D-TIFF, or a directory of PNG slices to DICOM, using a
donor DICOM for patient/study metadata and (optionally) a NIfTI underlay for
spatially-correct geometry when the input is a directory of rendered PNG
screenshots (e.g. mrview captures).

## Running it

```
python3 convert2dicom.py [options] <nifti|tiff|png_dir> <donor.dcm> <out_dicom_dir>
```

Options (`-s/--seriesdescription`, `-n/--seriesnumber`,
`-p/--pixelspacing`, `-u/--underlay`, `-o/--plane`, `-M/--match-donor`,
`-q/--quantitative`, `--label`, `--units`, `--window-headroom`, `-v/--verbose`).
`python3 convert2dicom_convert.py -h` prints the full help.

## Running the tests

```
pip install -r requirements.txt
pytest
```

Or with [uv](https://docs.astral.sh/uv/) (see `pyproject.toml`):

```
uv sync --group dev
uv run pytest
```

85 tests, no network access or real patient data needed -- everything is
synthetic (tiny in-memory NIfTI volumes, PNGs, and a minimal-but-valid
pydicom-built donor file).

## Building / installing with uv

```
uv build          # produces dist/convert2dicom-<version>-py3-none-any.whl + sdist
uv sync           # installs into .venv, including the `convert2dicom` console script
uv run convert2dicom -h
```

`requirements.txt` is kept too, for anyone installing with plain `pip` instead.

## Module map

Pure math / logic (no SimpleITK, pydicom, or filesystem I/O -- these have the
most unit tests and are safe to reuse elsewhere):

| Module | What it does |
|---|---|
| `charset.py` | DICOM Specific Character Set (0008,0005) resolution + safe donor-tag string coercion |
| `formatting.py` | DS (Decimal String) value formatting within DICOM's 16-byte limit |
| `geometry.py` | Per-slice IPP/IOP/SliceLocation/Thickness, reproducing mrview's fixed display convention |
| `crop.py` | Geometric crop/scale math for isolating rendered anatomy inside a PNG screenshot |
| `rescale.py` | Voxel value -> int16 rescaling (label / quantitative / legacy modes) + default window computation |
| `donor_match_geom.py` | Slice-location tables for matching PNGs to donor frames (SAG donor-match mode) |
| `uids.py` | Series/SOP Instance UID generation, Study UID resolution |
| `series_tags.py` | Shared DICOM series-tag list assembly (SOP Class selection, common tags) |

I/O-adjacent (thin wrappers around SimpleITK/pydicom, orchestration):

| Module | What it does |
|---|---|
| `donor.py` | Reads the donor DICOM header, exposes tags as plain decoded strings |
| `writer.py` | Writes one 2-D slice of a SimpleITK volume as a DICOM file |
| `modality_lut.py` | Attaches RescaleSlope/Intercept/Type + default window post-hoc via pydicom (can't go through the SimpleITK writer -- see the module docstring) |
| `pipeline_standard.py` | NIfTI/TIFF/plain-PNG-dir loading + int16 rescale |
| `pipeline_donor_match.py` | SAG donor-match mode (`-M`) |
| `pipeline_tra_cor.py` | TRA/COR PNG mode (geometry from the underlay) |
| `cli.py` | Argument parsing + orchestration -- picks a mode and calls into the above |

`convert2dicom.py` at the repo root is a thin executable wrapper so the
tool can still be run the same way as the original single-file script.
