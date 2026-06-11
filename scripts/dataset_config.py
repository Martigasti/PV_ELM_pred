"""
Dataset selection shared by all the forecasting scripts.

Set the env var DATASET to switch the input series, the day/night site and
the output directory in one place:

    DATASET=Palaiseau   (default)  -> data/Palaiseau/PV_AC_..._Palaiseau.csv, results/Palaiseau
    DATASET=oxelar                 -> data/cs/oxelar.csv,                results/oxelar
    DATASET=signes                 -> data/cs/signes.csv,                results/signes
    DATASET=solete                 -> data/Solete/DimRed_PAC_Solete.nc,  results/Solete
    DATASET=alice                  -> data/Alice/DimRed_PAC_Alice.nc,    results/Alice

Each entry carries the raw source (CSV or NetCDF), the 30-min cache, the
results dir, the forecast target name + unit (QUANTITY/UNIT, cosmetic: print
labels and a `quantity` CSV column), the site coordinates, the series start
date (00:00 UTC) and a distinct day-mask cache so the sites never overwrite
each other's mask.
CSV sources (Palaiseau, oxelar) have nc=None; NetCDF sources (solete, alice)
have csv=None and are built by preprocessing_nc.py.

This module is import-cycle-free (it only depends on utils), so elm_common,
blend_optimisation, blend_correlation and timegpt can all import from it.
"""
import os
from pathlib import Path

import numpy as np

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
    },
    "oxelar": {
        "csv": PROJECT_ROOT / "data" / "cs" / "oxelar.csv",
        "nc": None,
        "cache": PROJECT_ROOT / "data" / "cs" / "oxelar_30min.npy",
        "results": PROJECT_ROOT / "results" / "oxelar",
        "quantity": "PAC",
        "unit": "kW",
        "lat": 50.773375,
        "lon": 2.474552,
        "start_utc": "2023-11-07 00:00:00",
        "mask_cache": PROJECT_ROOT / "data" / "cs" / "is_day_mask_oxelar.npy",
        # Full-mode window: whole series provided by the company (2.56 yr),
        # so the benchmark is evaluated on the same span they use for theirs.
        # None => load_30min reads every available point.
        "ndata_full": None,
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
        "unit": "kW",
        "lat": 43.25551,
        "lon": 5.8,
        "start_utc": "2023-10-31 00:00:00",  # first 00:00 UTC <= first row (08:00)
        "mask_cache": PROJECT_ROOT / "data" / "cs" / "is_day_mask_signes.npy",
        # Full-mode window: 2 balanced years (2 seasonal cycles), like
        # Palaiseau/Alice. Series spans ~2.58 yr, so the extra ~0.58 yr is
        # dropped for seasonal balance -> 50/50 split = 1 yr train / 1 yr test.
        "ndata_full": round(2 * 365.25 * 48),
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


def day_mask(n_steps: int) -> np.ndarray:
    """compute_is_day_mask wired to the selected dataset's site + start date."""
    return compute_is_day_mask(
        n_steps, latitude=LAT, longitude=LON,
        start_utc=START_UTC, cache_path=MASK_CACHE,
    )
