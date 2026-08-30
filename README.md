# convert2dicom

Converts a NIfTI, 3D-TIFF, or a directory of PNG slices to DICOM, using a
donor DICOM for patient/study metadata and (optionally) a NIfTI underlay for
spatially-correct geometry when the input is a directory of rendered PNG
screenshots (e.g. mrview captures).

This is a refactor of a single ~750-line script into a small, independently
tested package. Behavior is preserved except for the bug fixes listed below.

## Running it

```
python3 convert2dicom_convert.py [options] <nifti|tiff|png_dir> <donor.dcm> <out_dicom_dir>
```

Same options as before (`-s/--seriesdescription`, `-n/--seriesnumber`,
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

`convert2dicom_convert.py` at the repo root is a thin executable wrapper so the
tool can still be run the same way as the original single-file script.

## Bugs found and fixed during the refactor

1. **Syntax error.** The original had a stray trailing `:` after the final
   `_attach_modality_lut(...)` call in `main()`, which is a `SyntaxError` --
   the script as given could not run at all in quantitative/label mode (or
   really, at all, since it's a module-level syntax error).

2. **Label mode silently skipped its own core step.** `img_int16`,
   `rescale_slope`, and `rescale_intercept` were assigned only *inside* the
   "values aren't integers, rounding" warning branch:
   ```python
   if not np.allclose(img_data, np.round(img_data)):
       print("Warning: --label given but values are not integers; rounding")
       img_int16 = np.round(img_data).astype(np.int16)
       rescale_slope, rescale_intercept = 1.0, 0.0
       print(f"Label mode: ...")
   ```
   For an already-integer label map -- the normal case -- none of those
   variables were ever set, so the script would crash with a `NameError`
   later. Fixed in `pipeline_standard.convert_nifti_to_int16` /
   `rescale.compute_label_rescale`: the int16 conversion and slope/intercept
   assignment always run; only the warning is conditional.

3. **Quantitative mode's log message never printed for normal data.** The
   `'Quantitative mode: [...] -> int16, slope=... intercept=...'` print was
   nested under the `_peak == 0` (all-zero) branch only, so it silently never
   fired for ordinary non-zero data -- the opposite of what the message and
   its indentation level suggested. Fixed the same way: the print happens
   once, unconditionally, in the caller.

4. **Inconsistent "no StudyInstanceUID" handling.** The plain NIfTI/TIFF path
   printed `Warning: donor has no StudyInstanceUID; generating one` when the
   donor lacked one; the two PNG-screenshot paths (donor-match SAG, TRA/COR)
   silently generated a UID in the exact same situation with no warning.
   Unified in `uids.resolve_study_uid`, used by all three write paths, which
   always calls the `on_warning` callback.

None of these change output for inputs that previously "worked" end-to-end
(bugs 2 and 3 only affected code paths that would have crashed or under-logged;
bug 4 only affects a printed warning). Bug 1 means the original script could
not run as pasted, so there's no prior "working" behavior to preserve there.

## Notes / things I did not change

- UID formats/roots (`1.2.826.0.1.3680043.2.1125...`, KU Leuven's DICOM root)
  are unchanged.
- The three write paths (standard / donor-match SAG / TRA-COR) still build
  their per-instance metadata somewhat differently (e.g. exactly which tags
  get a per-slice vs. per-series value) -- I consolidated the genuinely
  shared pieces (tag copying, common series tags, UID generation) but did not
  force full byte-for-byte identical code paths where the original didn't
  have them, to avoid changing output for cases that currently work.
- I did not add DICOM validity checking beyond what the original had (e.g.
  no VR-length validation on copied donor tags beyond what pydicom/GDCM
  already enforce on write).
