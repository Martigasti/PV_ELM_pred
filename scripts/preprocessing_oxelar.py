"""
Preprocessing of the Oxelar PV dataset.

Reads `data/cs/oxelar.csv` (columns `measure_date`, `prod`; 5-min step;
production in kW), resamples the production into 30-minute averages and
saves the resulting 1D array to `data/cs/oxelar_30min.npy`.

Two differences with the Palaiseau series matter here:

  * The raw series does NOT start at midnight (first timestamp is
    2023-11-07 14:30 UTC). The whole pipeline assumes arr[0] is the
    00:00 UTC slot (time features = idx % 48, day mask start date,
    cyclic persistence at 48 steps). We therefore reindex onto a complete
    5-min grid starting at the first 00:00 UTC <= first timestamp, so the
    leading missing slots become NaN -> 0 (handled downstream by
    load_30min) and arr[0] lands on midnight.
  * Timestamps are end-of-interval (12:35 = mean over 12:30->12:35) vs
    start-of-interval for Palaiseau. We do NOT shift: a uniform shift is
    invisible to the AR/persistence/NICE machinery and the residual offset
    is below the 30-min resolution, so it does not bias the metrics.

The unit (kW) is kept as-is; metrics will be in kW for Oxelar (vs W for
Palaiseau), but NICE / nRMSE / R2 are dimensionless and remain comparable.
"""

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
CSV_FILE = os.path.join(PROJECT_ROOT, 'data', 'cs', 'oxelar.csv')
CACHE_NPY = os.path.join(PROJECT_ROOT, 'data', 'cs', 'oxelar_30min.npy')

RAW_FREQ = '5min'


def main():
    # Load the raw data (5-min time step, columns measure_date / prod)
    df = pd.read_csv(CSV_FILE)
    df = df.rename(columns={'measure_date': 'datetime', 'prod': 'PAC'})
    df['datetime'] = pd.to_datetime(
        df['datetime'].astype(str).str.split('+').str[0],
        format='%Y-%m-%d %H:%M:%S',
    )
    df = df.sort_values('datetime').set_index('datetime')
    print(f'Raw rows      : {len(df):,}')
    print(f'Range         : {df.index.min()} -> {df.index.max()}')

    # Reindex onto a complete 5-min grid starting at the first 00:00 UTC
    # <= first timestamp, so arr[0] is the midnight slot. Missing leading
    # slots become NaN -> 0 downstream (load_30min).
    start = df.index.min().normalize()  # 00:00 of the first day
    end = df.index.max()
    full_index = pd.date_range(start, end, freq=RAW_FREQ)
    df = df.reindex(full_index)
    print(f'Aligned start : {df.index.min()} (should be 00:00)')

    # Resample to 30 min (average of the six 5-min measurements)
    arr = df['PAC'].resample('30min').mean().to_numpy()
    print(f'30-min rows   : {len(arr):,}  ({len(arr)/48/365.25:.2f} years)')
    print(f'arr[0]={arr[0]} (first slot = 00:00, expected ~0 / NaN)')
    print(f'Mean={np.nanmean(arr):.1f} kW,  max={np.nanmax(arr):.1f} kW')

    # Save the cache
    os.makedirs(os.path.dirname(CACHE_NPY), exist_ok=True)
    np.save(CACHE_NPY, arr)
    print(f'Saved {CACHE_NPY}  ({arr.nbytes/1e6:.1f} MB, {len(arr):,} samples)')

    # Check the size is sufficient for full mode (2 years)
    N_FULL = round(2 * 365.25 * 48)
    print(f'Required for full run : {N_FULL:,}')
    print(f'Available : {len(arr):,}')
    assert len(arr) >= N_FULL, 'Not enough samples for the full-mode run.'


if __name__ == '__main__':
    t0 = time.time()
    main()
    dt = time.time() - t0
    print(f'\n[chrono] preprocessing_oxelar.py : {dt:.2f} s  ({dt/60:.2f} min)')
