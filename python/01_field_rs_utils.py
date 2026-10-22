"""Utilities for checking field–satellite matches in Moroccan dryland steppes.

Reference code accompanying the manuscript on field-calibrated satellite
indicators of live vegetation dynamics in Moroccan dryland steppes.

Input tables contain field-observed fractional live cover (F_alive) matched
within ±8 days to spectral indices extracted using the GEE workflow.
Field-matching records cover 2016–2023; the separate trend reconstruction
uses a common 2017–2025 satellite analysis period.

This module prepares supplementary field–satellite matching diagnostics.
It does not construct matched observations from raw GEE CSV exports.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional, Union

import numpy as np
import pandas as pd

# Rename the input files to these names, or supply sensor_files=... when loading.
DEFAULT_SENSOR_FILES = {
    "Landsat": "Landsat_field_satellite_matched.csv",
    "Sentinel-2": "Sentinel2_field_satellite_matched.csv",
    "MODIS": "MODIS_field_satellite_matched.csv",
}

DEFAULT_HABITAT_ORDER = [
    "Reg",
    "Chamaephyte steppe",
    "Anabasis steppe",
    "Alfah-grass steppe",
    "Salty steppe",
    "Spreading area",
    "Wadi and gullies",
]
DEFAULT_SEASON_ORDER = ["Spring", "Summer", "Fall", "Winter"]

DataSource = Union[str, os.PathLike, pd.DataFrame]


def _read_table(data: DataSource) -> pd.DataFrame:
    if isinstance(data, (str, os.PathLike)):
        return pd.read_csv(data, dtype={"Station": "string"})
    if isinstance(data, pd.DataFrame):
        return data.copy()
    raise TypeError("Input must be a CSV path or pandas DataFrame.")


def _require_columns(df: pd.DataFrame, cols, name="dataframe") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _first_existing(df: pd.DataFrame, candidates, *, required=True):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"None of the expected columns were found: {list(candidates)}")
    return None


def _clean_station_season(df: pd.DataFrame, station_col="Station", season_col="Season") -> pd.DataFrame:
    """Normalize keys while retaining all original data columns."""
    _require_columns(df, [station_col, season_col])
    out = df.copy()
    out[station_col] = out[station_col].astype("string").str.strip()
    out[season_col] = out[season_col].astype("string").str.strip()
    out[season_col] = out[season_col].replace({"Autumn": "Fall"})
    for col in [station_col, season_col]:
        if out[col].isna().any() or out[col].eq("").any():
            raise ValueError(f"Missing or blank {col} values in input data.")
    return out


def _set_canonical_habitat(df: pd.DataFrame, column="Habitat") -> pd.DataFrame:
    """Standardize the user-facing habitat label to `Habitat`."""
    _require_columns(df, [column], "input table")
    out = df.copy()
    out["Habitat"] = out[column].astype("string").str.strip()
    if out["Habitat"].isna().any() or out["Habitat"].eq("").any():
        raise ValueError(f"Missing or blank habitat values in {column!r}.")
    return out

def _normalize_field_table(
    field_data: DataSource,
    station_metadata: Optional[DataSource] = None,
    *,
    station_col="Station",
    habitat_col="Habitat",
    season_col="Season",
) -> pd.DataFrame:
    """Prepare field records for matching-coverage summaries.

    Accepts a `Habitat` column in field_data, or joins authorized
    station_metadata containing `Station` and `Habitat`.
    """
    df = _clean_station_season(_read_table(field_data), station_col, season_col)
    date_col = _first_existing(df, ["Date", "Date_Picture", "date"])
    df["_field_date"] = pd.to_datetime(df[date_col], errors="coerce")

    if habitat_col not in df.columns:
        if station_metadata is None:
            raise ValueError(
                "field_data requires Habitat, or station_metadata with Station and Habitat."
            )
        meta = _read_table(station_metadata)
        _require_columns(meta, [station_col, habitat_col], "station_metadata")
        meta = meta[[station_col, habitat_col]].copy()
        meta[station_col] = meta[station_col].astype("string").str.strip()
        meta[habitat_col] = meta[habitat_col].astype("string").str.strip()
        meta = meta.drop_duplicates()
        if meta[station_col].duplicated().any():
            dup = meta.loc[meta[station_col].duplicated(), station_col].tolist()
            raise ValueError(f"Conflicting habitat metadata for stations: {dup[:10]}")
        df = df.merge(meta, on=station_col, how="left", validate="many_to_one")

    return _set_canonical_habitat(df, habitat_col)


def _normalize_matched_table(df: pd.DataFrame, name="matched sensor") -> pd.DataFrame:
    """Standardize habitat and calculate *absolute* date offsets for QC."""
    _require_columns(
        df, ["Station", "Season", "RS_date", "Date_diff_days"], name
    )
    out = _clean_station_season(df)
    out = _set_canonical_habitat(out)
    field_date_col = _first_existing(out, ["Date", "Date_Picture", "date"])
    out["_field_date"] = pd.to_datetime(out[field_date_col], errors="coerce")
    out["_rs_date"] = pd.to_datetime(out["RS_date"], errors="coerce")
    out["Date_diff_days"] = pd.to_numeric(out["Date_diff_days"], errors="coerce")
    out["_abs_day_difference"] = out["Date_diff_days"].abs()
    return out


def _prepare_matched(matched: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if not matched:
        raise ValueError("matched must contain at least one sensor table.")
    return {name: _normalize_matched_table(df, name) for name, df in matched.items()}


def _check_categories(df: pd.DataFrame, name: str, habitats, seasons) -> None:
    bad_habitats = sorted(set(df["Habitat"].dropna()) - set(habitats))
    bad_seasons = sorted(set(df["Season"].dropna()) - set(seasons))
    if bad_habitats or bad_seasons:
        raise ValueError(
            f"{name}: category labels are not in the requested order. "
            f"Unknown habitats: {bad_habitats}; unknown seasons: {bad_seasons}. "
            "Check spelling or specify habitat_order/season_order."
        )


def load_matched_sensor_files(
    matched_dir: Union[str, os.PathLike],
    sensor_files: Optional[Mapping[str, str]] = None,
) -> dict[str, pd.DataFrame]:
    """Load previously matched field–satellite CSVs; normalize labels and dates.

    Default filenames are publication-friendly aliases, not the original
    local Merged_Veg_*_26.csv filenames. Pass sensor_files to use other names.
    """
    folder = Path(matched_dir)
    files = dict(DEFAULT_SENSOR_FILES if sensor_files is None else sensor_files)
    if not files:
        raise ValueError("sensor_files cannot be empty.")
    out = {}
    for sensor, filename in files.items():
        path = folder / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. Rename your CSV or provide sensor_files=..."
            )
        out[sensor] = _normalize_matched_table(_read_table(path), filename)
    return out


def validate_matched_sensor_files(
    matched: Mapping[str, pd.DataFrame],
    *,
    expected_window_days=8,
    expected_n_field_records=3532,
) -> pd.DataFrame:
    """Audit coverage and temporal matching; do not silently remove bad rows."""
    if expected_n_field_records <= 0:
        raise ValueError("expected_n_field_records must be positive.")
    rows = []
    for sensor, df in _prepare_matched(matched).items():
        d = df["_abs_day_difference"]
        rows.append({
            "Sensor": sensor,
            "Matched_records": len(df),
            "Expected_field_records": expected_n_field_records,
            "Matched_fraction": len(df) / expected_n_field_records,
            "Unique_plots": df["Station"].nunique(),
            "Missing_field_date": int(df["_field_date"].isna().sum()),
            "Missing_RS_date": int(df["_rs_date"].isna().sum()),
            "Missing_Date_diff_days": int(d.isna().sum()),
            f"Records_over_{expected_window_days}d": int(d.gt(expected_window_days).sum()),
            "Negative_Date_diff_days": int(df["Date_diff_days"].lt(0).sum()),
        })
    return pd.DataFrame(rows)


def make_table_s3(
    field_data: DataSource,
    matched: Mapping[str, pd.DataFrame],
    *,
    station_metadata: Optional[DataSource] = None,
    habitat_col="Habitat",
    season_col="Season",
    station_col="Station",
    habitat_order=None,
    season_order=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Table S3: field availability and matched records by habitat and season.

    Returns long (habitat × season × sensor) and wide tables.
    """
    habitats = list(DEFAULT_HABITAT_ORDER if habitat_order is None else habitat_order)
    seasons = list(DEFAULT_SEASON_ORDER if season_order is None else season_order)
    field = _normalize_field_table(
        field_data,
        station_metadata=station_metadata,
        station_col=station_col,
        habitat_col=habitat_col,
        season_col=season_col,
    )
    prepared = _prepare_matched(matched)
    _check_categories(field, "field_data", habitats, seasons)
    for sensor, df in prepared.items():
        _check_categories(df, sensor, habitats, seasons)

    field_counts = (
        field.groupby(["Habitat", season_col], observed=True)
        .agg(
            Field_records=(station_col, "size"),
            Field_unique_plots=(station_col, "nunique"),
        )
        .reset_index()
        .rename(columns={season_col: "Season"})
    )
    sensor_rows = []
    for sensor, df in prepared.items():
        g = (
            df.groupby(["Habitat", "Season"], observed=True)
            .agg(
                Matched_records=("Station", "size"),
                Matched_unique_plots=("Station", "nunique"),
                Unique_field_dates=("_field_date", "nunique"),
                Unique_RS_dates=("_rs_date", "nunique"),
            )
            .reset_index()
        )
        g["Sensor"] = sensor
        sensor_rows.append(g)
    matched_counts = pd.concat(sensor_rows, ignore_index=True)

    grid = pd.MultiIndex.from_product(
        [habitats, seasons, list(prepared)], names=["Habitat", "Season", "Sensor"]
    ).to_frame(index=False)
    table_long = (
        grid.merge(field_counts, on=["Habitat", "Season"], how="left", validate="many_to_one")
        .merge(matched_counts, on=["Habitat", "Season", "Sensor"], how="left", validate="one_to_one")
    )
    count_cols = [
        "Field_records", "Field_unique_plots", "Matched_records",
        "Matched_unique_plots", "Unique_field_dates", "Unique_RS_dates",
    ]
    table_long[count_cols] = table_long[count_cols].fillna(0).astype(int)
    bad = table_long["Matched_records"] > table_long["Field_records"]
    if bad.any():
        example = table_long.loc[bad, ["Habitat", "Season", "Sensor"]].head(5)
        raise ValueError(
            "Matched row count exceeds field row count in one or more strata. "
            "Check duplicate GEE matches or field-row identifiers: "
            f"{example.to_dict(orient='records')}"
        )
    table_long["Match_rate_pct"] = np.where(
        table_long["Field_records"] > 0,
        100 * table_long["Matched_records"] / table_long["Field_records"],
        np.nan,
    )
    table_long["Habitat"] = pd.Categorical(table_long["Habitat"], categories=habitats, ordered=True)
    table_long["Season"] = pd.Categorical(table_long["Season"], categories=seasons, ordered=True)
    table_long = table_long.sort_values(["Habitat", "Season", "Sensor"]).reset_index(drop=True)

    # Include all requested habitat–season strata, including zero-field rows.
    base = table_long[["Habitat", "Season", "Field_records", "Field_unique_plots"]].drop_duplicates()
    table_wide = base.copy()
    for sensor in prepared:
        x = table_long.loc[table_long["Sensor"].eq(sensor)].copy()
        x = x.rename(columns={
            "Matched_records": f"{sensor}_matched_n",
            "Match_rate_pct": f"{sensor}_match_pct",
            "Matched_unique_plots": f"{sensor}_unique_plots",
        })
        x = x[[
            "Habitat", "Season", f"{sensor}_matched_n",
            f"{sensor}_match_pct", f"{sensor}_unique_plots",
        ]]
        table_wide = table_wide.merge(x, on=["Habitat", "Season"], how="left", validate="one_to_one")
    table_wide = table_wide.sort_values(["Habitat", "Season"]).reset_index(drop=True)
    return table_long, table_wide


def make_table_s4(
    field_data: DataSource,
    matched: Mapping[str, pd.DataFrame],
    *,
    station_metadata: Optional[DataSource] = None,
    expected_window_days=8,
) -> pd.DataFrame:
    """Table S4: sample coverage and absolute temporal-offset diagnostics."""
    field = _normalize_field_table(field_data, station_metadata=station_metadata)
    n_field = len(field)
    rows = []
    for sensor, df in _prepare_matched(matched).items():
        d = df["_abs_day_difference"].dropna()
        q1, q3 = (d.quantile(0.25), d.quantile(0.75)) if len(d) else (np.nan, np.nan)
        rows.append({
            "Sensor": sensor,
            "Field_records_available": n_field,
            "Matched_records": len(df),
            "Match_rate_pct": 100 * len(df) / n_field if n_field else np.nan,
            "Unique_plots_matched": df["Station"].nunique(),
            "Unique_field_dates_matched": df["_field_date"].nunique(),
            "Unique_RS_dates_used": df["_rs_date"].nunique(),
            "Median_abs_day_difference": d.median() if len(d) else np.nan,
            "Q1_abs_day_difference": q1,
            "Q3_abs_day_difference": q3,
            "IQR_abs_day_difference": q3 - q1 if len(d) else np.nan,
            "Max_abs_day_difference": d.max() if len(d) else np.nan,
            "Exact_date_matches_n": int(d.eq(0).sum()),
            f"Within_{expected_window_days}d_n": int(d.le(expected_window_days).sum()),
            f"Beyond_{expected_window_days}d_n": int(d.gt(expected_window_days).sum()),
        })
    return pd.DataFrame(rows)


def make_matched_rs_date_list(matched: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """List RS acquisition dates selected by the field-matching procedure."""
    rows = []
    for sensor, df in _prepare_matched(matched).items():
        x = (
            df.dropna(subset=["_rs_date"])
            .groupby("_rs_date")
            .agg(
                Matched_field_records=("Station", "size"),
                Unique_plots=("Station", "nunique"),
                Unique_field_dates=("_field_date", "nunique"),
                Median_abs_day_difference=("_abs_day_difference", "median"),
                Max_abs_day_difference=("_abs_day_difference", "max"),
            )
            .reset_index()
            .rename(columns={"_rs_date": "RS_date"})
        )
        x.insert(0, "Sensor", sensor)
        rows.append(x)
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["Sensor", "RS_date"]).reset_index(drop=True)


def make_day_difference_distribution(
    matched: Mapping[str, pd.DataFrame], *, window_days=8
) -> pd.DataFrame:
    """QC: distribution of *absolute* date offsets (0 through window_days).

    Percentages use all nonmissing offsets as denominator; if an offset is
    out of range, displayed percentages may sum to less than 100%.
    """
    rows = []
    for sensor, df in _prepare_matched(matched).items():
        d = df["_abs_day_difference"].dropna()
        total = len(d)
        for day in range(window_days + 1):
            n = int(d.eq(day).sum())
            rows.append({
                "Sensor": sensor,
                "Absolute_day_difference": day,
                "Matched_records": n,
                "Matched_pct": 100 * n / total if total else np.nan,
            })
    return pd.DataFrame(rows)


def audit_raw_station_date_duplicates(
    rs_files: Mapping[str, DataSource],
    *,
    station_col="Station",
    date_col="date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit raw GEE extractions for repeated Station × acquisition-date rows."""
    summary_rows, duplicate_rows = [], []
    for sensor, source in rs_files.items():
        df = _read_table(source)
        _require_columns(df, [station_col, date_col], f"{sensor} raw RS table")
        df[station_col] = df[station_col].astype("string").str.strip()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[station_col, date_col]).copy()
        counts = df.groupby([station_col, date_col]).size().reset_index(name="n_rows")
        dup = counts[counts["n_rows"] > 1].copy()
        dup.insert(0, "Sensor", sensor)
        duplicate_rows.append(dup)
        summary_rows.append({
            "Sensor": sensor,
            "Rows": len(df),
            "Unique_plots": df[station_col].nunique(),
            "Unique_dates": df[date_col].nunique(),
            "Unique_station_date_pairs": len(counts),
            "Duplicate_station_date_pairs": len(dup),
            "Max_rows_per_station_date": int(counts["n_rows"].max()) if len(counts) else 0,
        })
    summary = pd.DataFrame(summary_rows)
    duplicates = pd.concat(duplicate_rows, ignore_index=True) if duplicate_rows else pd.DataFrame()
    return summary, duplicates
