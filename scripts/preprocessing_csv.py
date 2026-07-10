"""
Generic CSV preprocessing, driven by the env var DATASET (the parsing knobs
live in dataset_config.py: SEP, DATETIME_COL, TARGET_COL, DATE_FORMAT,
SPLIT_PLUS, DECIMAL_COMMA, RAW_FREQ, CACHE_FREQ). It replaces the per-dataset
hardcoded scripts (Palaiseau, oxelar, signes, GHI): the only differences between
those were parameters, so one code path covers them all.

Usage (normally reached via the dispatcher preprocessing.py):
    DATASET=oxelar python scripts/preprocessing_csv.py

What it does (same skeleton as the old per-dataset scripts):

  * read the CSV (SEP), parse DATETIME_COL with DATE_FORMAT (stripping a
    "+HH:MM" tz suffix first when SPLIT_PLUS), read TARGET_COL (normalising
    comma decimals + empties -> NaN when DECIMAL_COMMA),
  * reindex onto a complete RAW_FREQ grid from the first 00:00 UTC <= first
    timestamp, so arr[0] is the midnight slot (the whole pipeline assumes it:
    cyclic time features via idx % STEPS_PER_DAY, day-mask start date, P° at one
    day). Leading/internal gaps become NaN -> 0 downstream (load_30min),
  * resample to CACHE_FREQ (default "30min"; set raw_freq to keep the dataset's
    native step and predict at that step) and save the 1D array to CACHE_NPY.

The final clean-up (NaN -> 0, clip >= 0, n_rows truncation) is done downstream by
load_30min (utils.py).
"""

import time

import numpy as np
import pandas as pd

from dataset_config import (
    CACHE_FREQ,
    CACHE_NPY,
    CSV_FILE,
    DATASET,
    DATE_FORMAT,
    DATETIME_COL,
    DECIMAL_COMMA,
    QUANTITY,
    RAW_FREQ,
    SEP,
    SPLIT_PLUS,
    STEPS_PER_DAY,
    TARGET_COL,
    UNIT,
)


def main():
    if CSV_FILE is None:
        raise SystemExit(
            f"DATASET={DATASET!r} has no CSV source. preprocessing_csv.py is for "
            f"the CSV datasets; use preprocessing_nc.py / preprocessing_meteo.py "
            f"for the others (the dispatcher preprocessing.py routes automatically)."
        )

    df = pd.read_csv(CSV_FILE, sep=SEP)

    ts = df[DATETIME_COL].astype(str)
    if SPLIT_PLUS:
        ts = ts.str.split("+").str[0]  # strip a "+HH:MM" tz suffix before parse
    df["datetime"] = pd.to_datetime(ts, format=DATE_FORMAT)

    if DECIMAL_COMMA:
        # Normalize dots + commas and empties (-> NaN) in one path.
        col = pd.to_numeric(
            df[TARGET_COL].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
    else:
        col = df[TARGET_COL]
    df = df.assign(_target=col)

    df = df.sort_values("datetime").set_index("datetime")
    print(f"DATASET       : {DATASET}")
    print(f"Raw rows      : {len(df):,}")
    print(f"Range         : {df.index.min()} -> {df.index.max()}")

    # Reindex onto a complete RAW_FREQ grid from the first 00:00 UTC <= first
    # timestamp, so arr[0] is the midnight slot. Missing slots become NaN -> 0
    # downstream (load_30min). No-op when the series already starts at midnight.
    start = df.index.min().normalize()
    full_index = pd.date_range(start, df.index.max(), freq=RAW_FREQ)
    target = df["_target"].reindex(full_index)
    print(f"Aligned start : {target.index.min()} (should be 00:00)")

    # Resample to the cache step (default 30 min; native step when raw_freq set).
    arr = target.resample(CACHE_FREQ).mean().to_numpy()
    yr = len(arr) / STEPS_PER_DAY / 365.25
    print(f"{CACHE_FREQ} rows : {len(arr):,}  ({yr:.2f} years, {STEPS_PER_DAY}/day)")
    print(f"arr[0]={arr[0]} (first slot = 00:00, expected ~0 / NaN)")
    print(
        f"{QUANTITY} mean={np.nanmean(arr):.1f} {UNIT},  "
        f"max={np.nanmax(arr):.1f} {UNIT}"
    )

    CACHE_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_NPY, arr)
    print(f"Saved {CACHE_NPY}  ({arr.nbytes/1e6:.1f} MB, {len(arr):,} samples)")


if __name__ == "__main__":
    t0 = time.time()
    main()
    dt = time.time() - t0
    print(f"\n[chrono] preprocessing_csv.py ({DATASET}) : {dt:.2f} s  ({dt/60:.2f} min)")
