"""
Dataset selection shared by all the forecasting scripts.

Set the env var DATASET to switch the input series, the day/night site and
the output directory in one place:

    DATASET=Palaiseau   (default)  -> data/Palaiseau/PV_AC_..._Palaiseau.csv, results/Palaiseau
    DATASET=oxelar                 -> data/cs/oxelar.csv,                results/oxelar
    DATASET=signes                 -> data/cs/signes.csv,                results/signes
    DATASET=solete                 -> data/Solete/DimRed_PAC_Solete.nc,  results/Solete
    DATASET=alice                  -> data/Alice/DimRed_PAC_Alice.nc,    results/Alice
    DATASET=ghi_palaiseau          -> data/GHI_Pal/...GHI...csv,         results/GHI_Palaiseau
    DATASET=ghi_alice              -> data/GHI_Alice/...GHI...csv,        results/GHI_Alice

Each entry carries the raw source (CSV or NetCDF), the 30-min cache, the
results dir, the forecast target name + unit (QUANTITY/UNIT, cosmetic: print
labels and a `quantity` CSV column), the site coordinates, the series start
date (00:00 UTC) and a distinct day-mask cache so the sites never overwrite
each other's mask.
CSV sources (Palaiseau, oxelar) have nc=None; NetCDF sources (solete, alice)
have csv=None and are built by preprocessing_nc.py.

Adding a new CSV dataset (no new script needed). The generic CSV preprocessing
(preprocessing_csv.py, reached via the dispatcher preprocessing.py) is driven by
a small block of optional parsing keys read with .get() defaults, so the
existing entries keep working unchanged:

    "parser"        : "csv" | "nc" | "meteo" (else inferred: nc if NC_FILE,
                      meteo if DATASET startswith "meteo_", else csv),
    "sep"           : CSV separator                         (default ",")
    "datetime_col"  : timestamp column name                 (default "datetime")
    "target_col"    : value column name                     (default "PAC")
    "date_format"   : strptime format for the timestamps    (default below)
    "split_plus"    : strip a "+HH:MM" tz suffix before parse (default True)
    "decimal_comma" : value column uses comma decimals       (default False)
    "raw_freq"      : native step of the source (grid/reindex), e.g. "5min"
    "cache_freq"    : step the .npy is stored at (default "30min"; set it to
                      raw_freq to predict at the dataset's native step).

STEPS_PER_DAY is derived from cache_freq when not given explicitly, so a native
-step dataset just declares raw_freq/cache_freq and the whole pipeline (lookback,
horizons in hours, cyclic period) follows automatically.

This module is import-cycle-free (it only depends on utils), so elm_common,
blend_optimisation, blend_correlation and timegpt can all import from it.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils import compute_is_day_mask

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET = os.environ.get("DATASET", "Palaiseau")
_DATASETS = {
    "Palaiseau": {
        "csv": PROJECT_ROOT / "data" / "Palaiseau" / "PV_AC_20200801_20250706_Palaiseau.csv",
        "nc": None,
        "cache": PROJECT_ROOT / "data" / "Palaiseau" / "data_30min.npy",
        "results": PROJECT_ROOT / "results" / "Palaiseau",
        "quantity": "PAC",
        "unit": "W",
        "lat": 48.7128,
        "lon": 2.2188,
        "start_utc": "2020-08-01 00:00:00",
        "mask_cache": PROJECT_ROOT / "data" / "Palaiseau" / "is_day_mask.npy",
        # Full-mode window: 2 balanced years (2 seasonal cycles).
        "ndata_full": round(2 * 365.25 * 48),
        # CSV parsing (15-min source, ',' sep, 'datetime'/'PAC', '+00:00' suffix).
        "datetime_col": "datetime", "target_col": "PAC", "raw_freq": "15min",
    },
    "oxelar": {
        "csv": PROJECT_ROOT / "data" / "cs" / "oxelar.csv",
        "nc": None,
        "cache": PROJECT_ROOT / "data" / "cs" / "oxelar_30min.npy",
        "results": PROJECT_ROOT / "results" / "oxelar",
        "quantity": "PAC",
        "unit": "W",
        "lat": 50.773375,
        "lon": 2.474552,
        "start_utc": "2023-11-07 00:00:00",
        "mask_cache": PROJECT_ROOT / "data" / "cs" / "is_day_mask_oxelar.npy",
        # Full-mode window: 2 balanced years (2 seasonal cycles), like Signes /
        # Palaiseau, so the 50/50 split gives 1 yr train / 1 yr test (comparable
        # windows across sites). Series spans ~2.56 yr; the extra ~0.56 yr is
        # dropped for seasonal balance.
        "ndata_full": round(2 * 365.25 * 48),
        # CSV parsing (5-min source, 'measure_date'/'prod', '+00:00' suffix).
        "datetime_col": "measure_date", "target_col": "prod", "raw_freq": "5min",
    },
    "signes": {
        # Signes (France, 730 kWp rooftop). CSV: measure_date / prod (5-min).
        # Same structure/conventions as oxelar (end-of-interval, not-midnight
        # start; empty prod = nighttime -> NaN -> 0 via load_30min).
        "csv": PROJECT_ROOT / "data" / "cs" / "signes.csv",
        "nc": None,
        "cache": PROJECT_ROOT / "data" / "cs" / "signes_30min.npy",
        "results": PROJECT_ROOT / "results" / "signes",
        "quantity": "PAC",
        "unit": "W",
        "lat": 43.25551,
        "lon": 5.8,
        "start_utc": "2023-10-31 00:00:00",  # first 00:00 UTC <= first row (08:00)
        "mask_cache": PROJECT_ROOT / "data" / "cs" / "is_day_mask_signes.npy",
        # Full-mode window: 2 balanced years (2 seasonal cycles), like
        # Palaiseau/Alice. Series spans ~2.58 yr, so the extra ~0.58 yr is
        # dropped for seasonal balance -> 50/50 split = 1 yr train / 1 yr test.
        "ndata_full": round(2 * 365.25 * 48),
        # CSV parsing (5-min source, 'measure_date'/'prod', '+00:00' suffix).
        "datetime_col": "measure_date", "target_col": "prod", "raw_freq": "5min",
    },
    "solete": {
        # SOLETE (DTU Risø, Denmark, 10 kWac). NetCDF-4: time + PAC (W).
        "csv": None,
        "nc": PROJECT_ROOT / "data" / "Solete" / "DimRed_PAC_Solete.nc",
        "cache": PROJECT_ROOT / "data" / "Solete" / "solete_30min.npy",
        "results": PROJECT_ROOT / "results" / "Solete",
        "quantity": "PAC",
        "unit": "W",
        "lat": 55.6867,
        "lon": 12.0985,
        "start_utc": "2018-06-01 00:00:00",
        "mask_cache": PROJECT_ROOT / "data" / "Solete" / "is_day_mask_solete.npy",
        # Whole series (~15 months): a 50/50 split still leaves ~7.5 months of
        # test, below 1 year, so we keep everything and read it from per-FH.
        "ndata_full": None,
    },
    "alice": {
        # Alice Springs (DKASC, Australia, Southern Hemisphere). NetCDF-4:
        # time + PAC (W) + QC flags (ignored). ~12 yr available.
        "csv": None,
        "nc": PROJECT_ROOT / "data" / "Alice" / "DimRed_PAC_Alice.nc",
        "cache": PROJECT_ROOT / "data" / "Alice" / "alice_30min.npy",
        "results": PROJECT_ROOT / "results" / "Alice",
        "quantity": "PAC",
        "unit": "W",
        "lat": -23.7603,
        "lon": 133.8784,
        "start_utc": "2013-08-14 00:00:00",
        "mask_cache": PROJECT_ROOT / "data" / "Alice" / "is_day_mask_alice.npy",
        # 2 balanced years (like Palaiseau): 2 seasonal cycles, bounded compute.
        "ndata_full": round(2 * 365.25 * 48),
    },
    "ghi_palaiseau": {
        # SIRTA Palaiseau GHI (Global Horizontal Irradiance, W/m2). CSV:
        # ';'-separated, 'DD/MM/YYYY HH:MM' timestamps (UTC), 30-min native,
        # column GHI (dot decimals). Same site as the Palaiseau PAC set.
        "csv": PROJECT_ROOT / "data" / "GHI_Pal" / "Données_Palaiseau_SIRTA_GHI_2022-2025.csv",
        "nc": None,
        "cache": PROJECT_ROOT / "data" / "GHI_Pal" / "ghi_palaiseau_30min.npy",
        "results": PROJECT_ROOT / "results" / "GHI_Palaiseau",
        "quantity": "GHI",
        "unit": "W/m2",
        "lat": 48.7128,
        "lon": 2.2188,
        "start_utc": "2022-01-01 00:00:00",
        "mask_cache": PROJECT_ROOT / "data" / "GHI_Pal" / "is_day_mask_ghi_palaiseau.npy",
        # 2 balanced years (like Palaiseau); series spans ~3.5 yr.
        "ndata_full": round(2 * 365.25 * 48),
        # CSV parsing (';' sep, 'timestamp'/'GHI', 'DD/MM/YYYY HH:MM', no tz suffix;
        # decimal_comma normalises dots+commas in one path).
        "sep": ";", "datetime_col": "timestamp", "target_col": "GHI",
        "date_format": "%d/%m/%Y %H:%M", "split_plus": False,
        "decimal_comma": True, "raw_freq": "30min",
    },
    "ghi_alice": {
        # Alice Springs GHI (Southern Hemisphere, W/m2). CSV: ';'-separated,
        # 'DD/MM/YYYY HH:MM' timestamps (UTC), 30-min native, column GHI
        # (comma decimals + trailing empty column). Same site as Alice PAC set.
        "csv": PROJECT_ROOT / "data" / "GHI_Alice" / "N°24_213-Site AliceSprings_QCell_6100W_Poly-Si_Fixed_2022-2025.csv",
        "nc": None,
        "cache": PROJECT_ROOT / "data" / "GHI_Alice" / "ghi_alice_30min.npy",
        "results": PROJECT_ROOT / "results" / "GHI_Alice",
        "quantity": "GHI",
        "unit": "W/m2",
        "lat": -23.7603,
        "lon": 133.8784,
        "start_utc": "2022-01-01 00:00:00",
        "mask_cache": PROJECT_ROOT / "data" / "GHI_Alice" / "is_day_mask_ghi_alice.npy",
        # 2 balanced years (like Alice PAC); series spans ~4 yr.
        "ndata_full": round(2 * 365.25 * 48),
        # CSV parsing (';' sep, comma decimals + trailing empty column,
        # 'DD/MM/YYYY HH:MM', no tz suffix).
        "sep": ";", "datetime_col": "timestamp", "target_col": "GHI",
        "date_format": "%d/%m/%Y %H:%M", "split_plus": False,
        "decimal_comma": True, "raw_freq": "30min",
    },
    **{
        # Barani MeteoHelix weather stations (Corsica). One source CSV holding 6
        # stations; we keep 2 (Vignola, Ajaccio) x 3 targets (temperature,
        # pressure, humidity) = 6 series, one .npy each (built by
        # preprocessing_meteo.py). These differ from all the PV/GHI sets:
        #   * native 10-min step kept  -> steps_per_day=144 (not 48),
        #   * non-solar targets        -> clip_nonneg=False (temperature < 0),
        #                                 solar=False (day mask neutralised, _day==_all).
        f"meteo_{site}_{target}": {
            "csv": PROJECT_ROOT / "data" / "Xavier" / "eml_barani_helix_2024_2025.csv",
            "nc": None,
            "cache": PROJECT_ROOT / "data" / "Xavier" / f"meteo_{site}_{target}_10min.npy",
            "results": PROJECT_ROOT / "results" / f"meteo_{site}_{target}",
            "quantity": target,
            "unit": unit,
            "lat": lat,
            "lon": lon,
            "start_utc": "2024-01-01 00:00:00",
            "mask_cache": PROJECT_ROOT / "data" / "Xavier" / f"is_day_mask_meteo_{site}_{target}.npy",
            # 2 balanced years at 144 steps/day (series spans 2024->2025, ~2 yr).
            "ndata_full": round(2 * 365.25 * 144),
            "steps_per_day": 144,
            "clip_nonneg": False,
            "solar": False,
            # Preprocessing-only fields (consumed by preprocessing_meteo.py).
            "station_name": station_name,
            "serial": serial,
            "target_col": target,
        }
        for site, station_name, serial, lat, lon in (
            ("vignola", "SAPHIR MeteoHelix & RainSensor Vignola", "2110SH043", 41.912453, 8.653093),
            ("ajaccio", "SAPHIR MeteoHelix, INSPE Garden, Ajacccio", "2008SH045", 41.913641, 8.728020),
        )
        for target, unit in (("temperature", "degC"), ("pressure", "hPa"), ("humidity", "%"))
    },
}

if DATASET not in _DATASETS:
    raise ValueError(f"Unknown DATASET={DATASET!r}; choose from {list(_DATASETS)}")
_DS = _DATASETS[DATASET]

CSV_FILE = _DS["csv"]
NC_FILE = _DS["nc"]
CACHE_NPY = _DS["cache"]
RESULTS_DIR = _DS["results"]
LAT = _DS["lat"]
LON = _DS["lon"]
START_UTC = _DS["start_utc"]
MASK_CACHE = _DS["mask_cache"]
NDATA_FULL = _DS["ndata_full"]
QUANTITY = _DS["quantity"]  # forecast target name (e.g. "PAC", "GHI"), cosmetic
UNIT = _DS["unit"]          # physical unit of the target (e.g. "W", "kW", "W/m2")

# --- Generic CSV-preprocessing parsing knobs (read with .get() defaults so the
# existing entries keep working unchanged). Consumed by preprocessing_csv.py. ---
PARSER = _DS.get(
    "parser",
    "nc" if NC_FILE is not None
    else "meteo" if DATASET.startswith("meteo_")
    else "csv",
)
SEP = _DS.get("sep", ",")
DATETIME_COL = _DS.get("datetime_col", "datetime")
TARGET_COL = _DS.get("target_col", "PAC")
DATE_FORMAT = _DS.get("date_format", "%Y-%m-%d %H:%M:%S")
SPLIT_PLUS = _DS.get("split_plus", True)        # strip a "+HH:MM" tz suffix
DECIMAL_COMMA = _DS.get("decimal_comma", False)
RAW_FREQ = _DS.get("raw_freq", "30min")         # native source step (grid)
# Step the .npy is stored at: default 30-min (historical); set raw_freq to keep
# the dataset's native step. STEPS_PER_DAY is derived from it when not explicit.
CACHE_FREQ = _DS.get("cache_freq", "30min")


def _steps_per_day_from_freq(freq: str) -> int:
    """Number of CACHE_FREQ slots in a day (e.g. '30min' -> 48, '5min' -> 288)."""
    minutes = pd.Timedelta(freq) / pd.Timedelta(minutes=1)
    return round(24 * 60 / minutes)


# Steps per day (48 = 30-min step, the historical default; 144 = 10-min meteo).
# Derived from CACHE_FREQ unless given explicitly.
STEPS_PER_DAY = _DS.get("steps_per_day", _steps_per_day_from_freq(CACHE_FREQ))
# Clip the series to >= 0 in load_30min (True for PV/GHI production/irradiance,
# False for meteo targets like temperature that can be negative).
CLIP_NONNEG = _DS.get("clip_nonneg", True)
# Solar dataset: when False (meteo targets) the day/night mask is meaningless,
# so day_mask returns all-True -> the _day table equals the _all table.
SOLAR = _DS.get("solar", True)


def day_mask(n_steps: int) -> np.ndarray:
    """compute_is_day_mask wired to the selected dataset's site + start date.

    For non-solar datasets (SOLAR=False) the day mask has no meaning, so we
    return an all-True mask: the _day metrics then coincide with _all and no
    downstream code path needs to change.
    """
    if not SOLAR:
        return np.ones(n_steps, dtype=bool)
    return compute_is_day_mask(
        n_steps, latitude=LAT, longitude=LON,
        start_utc=START_UTC, cache_path=MASK_CACHE,
    )
