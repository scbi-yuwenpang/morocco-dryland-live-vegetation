# Field-calibrated satellite indicators of live vegetation in Moroccan dryland steppes

Reference code supporting the manuscript *Field-calibrated satellite indicators of live vegetation dynamics in sparsely vegetated Moroccan dryland steppes* (submitted to *Ecological Indicators*, manuscript ECOLIND-62301).

**Scope:** This repository shares a Google Earth Engine (GEE) processing example and three Python reference scripts for working with field-calibrated satellite indicators. These scripts document the study's analytical approach; they are **not an independently verified, end-to-end reproduction** of every published result. Original field observations and restricted geospatial data are not distributed here.

## Repository contents

| File | Purpose |
| --- | --- |
| `gee/01_gee_preprocess_extract.js` | Reference GEE workflow for Landsat, Sentinel-2, and MODIS filtering, quality masking, nine spectral indices, and extraction from 50-m buffers around survey stations. |
| `python/01_field_rs_utils.py` | Load **already matched** field–satellite observations; inspect temporal matching, sample coverage, and habitat/season summaries (supplementary Tables S3–S4). It does **not** match raw GEE exports to field visits. |
| `python/02_grouped_rf_validation.py` | Illustrative 5-fold plot-grouped Random Forest validation with NDVI-only linear and ridge regression benchmarks. |
| `python/03_trend_tests.py` | Reference monthly and annual aggregation and trend testing: Sen's slope, Mann–Kendall, Benjamini–Hochberg FDR, and monthly Hamed–Rao sensitivity analysis. |
| `requirements.txt` | Suggested Python dependencies (unpinned). |

## Study periods

- **Field surveys and RF calibration/validation:** 2016–2023.
- **GEE image extraction:** Landsat/MODIS 2016–2025 and Sentinel-2 surface reflectance 2017–2025, subject to image availability and quality filters.
- **Reported cross-sensor interannual analysis:** 2017–2025, with annual results restricted by the specified minimum of nine valid months. The field and reconstruction periods serve different purposes.

## Input data

The Python examples refer to **sensor-specific, field–satellite matched CSVs**. These files combine field-observed fractional live vegetation cover (`F_alive`) with spectral indices extracted using GEE and matched to field dates within ±8 days (2016–2023).

For readability, this repository uses the common habitat column name **`Habitat`** in all reference scripts. Users adapting their own datasets should harmonize their column names before running the examples.

| Script | Expected fields |
| --- | --- |
| `01_field_rs_utils.py` | `Station`, `Habitat`, `Season`, field date (`Date` or `Date_Picture`), `RS_date`, `Date_diff_days`; a separate field observation table is also needed to calculate match rates against all available surveys. |
| `02_grouped_rf_validation.py` | `Station`, `F_alive`, and nine spectral indices: `NDVI`, `EVI`, `SAVI`, `MSAVI`, `NDMI`, `NBR`, `NBR2`, `NDWI`, `MNDWI`. |
| `03_trend_tests.py` | Satellite-derived **station-level predictions** with `Sensor`, `Habitat`, `Station`, `date`, and `F_alive_Pred`. These are a different dataset from the field-matched calibration CSVs. |

The suggested filenames for matched inputs are `Landsat_field_satellite_matched.csv`, `Sentinel2_field_satellite_matched.csv`, and `MODIS_field_satellite_matched.csv`. They are illustrative aliases, not files provided by this repository.

## Getting started

Python 3.10+ is recommended. Install the suggested packages with:

```bash
python -m pip install -r requirements.txt
```

The utility module can be imported to inspect authorized matched CSVs:

```python
# Filenames beginning with digits can be loaded via runpy.
from pathlib import Path
import runpy

utils = runpy.run_path("python/01_field_rs_utils.py")
matched = utils["load_matched_sensor_files"](Path("data/private"))
qc = utils["validate_matched_sensor_files"](matched)
print(qc)
```

The other scripts provide example command-line entry points:

```bash
# Demonstrate plot-grouped validation on one authorized matched sensor dataset.
python python/02_grouped_rf_validation.py \
  --input data/private/Landsat_field_satellite_matched.csv \
  --output outputs/landsat_oof_example.csv

# Demonstrate trend tests using a separate station-level prediction dataset.
python python/03_trend_tests.py \
  --input data/private/rf_station_predictions.csv \
  --output-dir outputs/trends_example
```

**The second command does not produce the input required by the third command.** The RF validation example outputs out-of-fold field-match predictions; an independently prepared, date-indexed station-level satellite prediction series is required for trend analysis. Data preparation, complete model fitting, and final figure production are not bundled into an executable pipeline.

## GEE setup and data access

A study-related Code Editor snapshot was referenced in manuscript materials: https://code.earthengine.google.com/15de2bd6239b4f62fed488fb3913767e . Its access permissions, version consistency, and execution have not been independently verified. The repository's reference script should not be assumed to be identical to that snapshot.

Supply an authorized study-area `FeatureCollection` and station-point `FeatureCollection`, then check that the station geometries are points before applying 50-m buffers. The example includes quality masking, scaled surface reflectance, nine spectral indices, and station-buffer extraction. Institutional Earth Engine assets and plot locations are not publicly distributed by this repository.


## Interpretation and reproducibility limits

These are **methodological examples**, not verified source-to-result reproduction scripts. 

Original field records, precise station coordinates, project-specific GEE assets, and record-level field–satellite datasets require data-provider authorization and are not included. Public satellite products remain accessible through their providers and GEE, subject to applicable access terms. 

## Citation and licensing

Please cite the associated manuscript once published. A versioned release may be added after institutional review. Code licensing should be decided by the relevant rights holders before public distribution.
