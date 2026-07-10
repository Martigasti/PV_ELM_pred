"""
Generic preprocessing for the Barani MeteoHelix weather dataset (Corsica),
driven by the env var DATASET (one of the `meteo_<site>_<target>` entries of
dataset_config.py, e.g. DATASET=meteo_vignola_temperature).

The single source CSV `data/Xavier/eml_barani_helix_2024_2025.csv` holds 6
stations (`station_name` / `serial_number`) at a 10-min native step. For the
selected dataset we:

  * filter to the one station (by serial_number, robust to the commas inside
    `station_name`),
  * select the one target column (temperature / pressure / humidity),
  * reindex onto a complete 10-min grid from the first 00:00 UTC (the series
    already starts at midnight, so this is a no-op for alignment but fills the
    internal gaps -> NaN, and guarantees arr[0] = 00:00 as the whole pipeline
    assumes),
  * interpolate the NaN holes linearly (the ~1000 missing rows are short gaps;
    filling them with 0 -- what load_30min does -- would be absurd for a
    temperature), then resample to the same 10-min step (no-op, robustness),
  * save the 1D array to `data/Xavier/<dataset>_10min.npy`.

Unlike the PV/GHI sets the targets are NOT solar: temperature can be negative
(clip disabled in load_30min via CLIP_NONNEG) and the day/night mask is
neutralised (SOLAR=False). This script is not a model -> NOT in run_full.py.
"""

import time

import numpy as np
import pandas as pd

from dataset_config import CACHE_NPY, CSV_FILE, DATASET, NDATA_FULL, _DS

RAW_FREQ = "10min"


def main():
    station_name = _DS["station_name"]
    serial = _DS["serial"]
    target_col = _DS["target_col"]
    print(f"DATASET       : {DATASET}")
    print(f"Station       : {station_name} ({serial})  target={target_col}")

    df = pd.read_csv(CSV_FILE)

    # Filter to the one station by serial (robust to commas in station_name).
    df = df[df["serial_number"] == serial].copy()
    if df.empty:
        raise ValueError(f"No rows for serial_number={serial!r} in {CSV_FILE}")

    df["datetime"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
    df = df.sort_values("datetime").set_index("datetime")
    df = df[~df.index.duplicated(keep="first")]
    print(f"Raw rows      : {len(df):,}")
    print(f"Range         : {df.index.min()} -> {df.index.max()}")

    target = pd.to_numeric(df[target_col], errors="coerce")

    # Reindex onto a complete 10-min grid starting at the first 00:00 UTC so
    # arr[0] is the midnight slot; internal gaps become NaN.
    start = df.index.min().normalize()  # 00:00 of the first day
    end = df.index.max()
    full_index = pd.date_range(start, end, freq=RAW_FREQ)
    target = target.reindex(full_index)
    print(f"Aligned start : {target.index.min()} (should be 00:00)")
    n_nan = int(target.isna().sum())
    print(f"NaN holes     : {n_nan:,} ({n_nan/len(target)*100:.2f}%)  -> interpolated")

    # Interpolate the NaN holes (short gaps), then resample (no-op on the
    # regular 10-min grid, kept for robustness). Any residual NaN -> 0.
    target = target.interpolate(method="linear", limit_direction="both")
    arr = target.resample(RAW_FREQ).mean().to_numpy()
    arr = np.where(np.isnan(arr), 0.0, arr)
    assert not np.isnan(arr).any(), "NaN left after interpolation"

    print(f"10-min rows   : {len(arr):,}  ({len(arr)/144/365.25:.2f} years)")
    print(f"arr[0]={arr[0]:.3f} (first slot = 00:00 UTC, real value not 0)")
    print(f"min={arr.min():.2f}  max={arr.max():.2f}  mean={arr.mean():.2f} {_DS['unit']}")

    CACHE_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_NPY, arr)
    print(f"Saved {CACHE_NPY}  ({arr.nbytes/1e6:.1f} MB, {len(arr):,} samples)")

    print(f"Required for full run : {NDATA_FULL:,}")
    print(f"Available             : {len(arr):,}")
    assert len(arr) >= NDATA_FULL, "Not enough samples for the full-mode run."


if __name__ == "__main__":
    t0 = time.time()
    main()
    dt = time.time() - t0
    print(f"\n[chrono] preprocessing_meteo.py : {dt:.2f} s  ({dt/60:.2f} min)")
