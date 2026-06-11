"""
Generic preprocessing for the NetCDF PV datasets (Solete, Alice).

Reads the dataset's NetCDF-4 file (`time` + `PAC` in Watts, plus scalar lat/lon and, for Alice,
QC `flag_*` variables that we ignore), resamples the production to 30-minute
averages and saves the 1D array to the dataset's `.npy` cache. All scripts
then consume this cache via `load_30min` (cache-first).

Usage:
    DATASET=solete python scripts/preprocessing_nc.py
    DATASET=alice python scripts/preprocessing_nc.py

Same alignment logic as preprocessing_oxelar.py:

  * arr[0] must be the 00:00 UTC slot (the whole pipeline assumes it: cyclic
    time features via idx % 48, day mask start date, P° at 48 steps). We
    reindex onto a complete grid starting at the first 00:00 UTC <= first
    timestamp; leading/internal gaps become NaN -> 0 downstream (load_30min).
    Both .nc files already start at midnight, so the padding is a no-op there,
    but it stays for robustness.
  * Timestamps are end-of-interval (NetCDF attr timestamp_ref), like oxelar.
    We do NOT shift: a uniform shift is invisible to the AR/persistence/NICE
    machinery and the residual offset is below the 30-min resolution.

The unit (W) matches Palaiseau. QC flags (Alice) are not applied: PAC is taken
raw, so the protocol is identical across all datasets (load_30min only does
NaN -> 0 and clip >= 0).
"""

import time

import numpy as np
import pandas as pd
import xarray as xr

from dataset_config import CACHE_NPY, DATASET, NC_FILE, UNIT


def main():
    if NC_FILE is None:
        raise SystemExit(
            f"DATASET={DATASET!r} has no NetCDF source. "
            f"preprocessing_nc.py is for solete/alice; use the CSV "
            f"preprocessing for Palaiseau/oxelar."
        )

    # Read time + PAC (xarray decodes the epoch-seconds axis to datetime64).
    ds = xr.open_dataset(NC_FILE)
    pac = ds["PAC"].to_series()
    ds.close()
    pac.index = pd.DatetimeIndex(pac.index)  # tz-naive UTC, as the other sets
    pac = pac.sort_index()
    print(f"Raw rows      : {len(pac):,}")
    print(f"Range         : {pac.index.min()} -> {pac.index.max()}")

    # Infer the raw step from the median spacing of the time axis.
    raw_freq = pac.index.to_series().diff().median()
    print(f"Raw step      : {raw_freq}")

    # Reindex onto a complete grid from the first 00:00 UTC <= first timestamp,
    # so arr[0] is the midnight slot. Missing slots become NaN -> 0 downstream.
    start = pac.index.min().normalize()
    full_index = pd.date_range(start, pac.index.max(), freq=raw_freq)
    pac = pac.reindex(full_index)
    print(f"Aligned start : {pac.index.min()} (should be 00:00)")

    # Resample to 30 min (mean of the raw measurements in each slot).
    arr = pac.resample("30min").mean().to_numpy()
    print(f"30-min rows   : {len(arr):,}  ({len(arr)/48/365.25:.2f} years)")
    print(f"arr[0]={arr[0]} (first slot = 00:00, expected ~0 / NaN)")
    print(f"Mean={np.nanmean(arr):.1f} {UNIT},  max={np.nanmax(arr):.1f} {UNIT}")

    CACHE_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_NPY, arr)
    print(f"Saved {CACHE_NPY}  ({arr.nbytes/1e6:.1f} MB, {len(arr):,} samples)")


if __name__ == "__main__":
    t0 = time.time()
    main()
    dt = time.time() - t0
    print(f"\n[chrono] preprocessing_nc.py ({DATASET}) : {dt:.2f} s  ({dt/60:.2f} min)")
