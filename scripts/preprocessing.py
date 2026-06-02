"""
Preprocessing of the PV_AC Palaiseau dataset.

Reads `PV_AC_20200801_20250706_Palaiseau.csv`, resamples the PAC column
into 30-minute averages, and saves the resulting 1D array to
`data_30min.npy`. The scripts in scripts/ consume this cache via their
`load_30min` helper.
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CSV_FILE = os.path.join(DATA_DIR, 'PV_AC_20200801_20250706_Palaiseau.csv')
CACHE_NPY = os.path.join(DATA_DIR, 'data_30min.npy')


def main():
    # Load the raw data (15-min time step)
    df = pd.read_csv(CSV_FILE)
    df['datetime'] = pd.to_datetime(
        df['datetime'].astype(str).str.split('+').str[0],
        format='%Y-%m-%d %H:%M:%S',
    )
    df = df.sort_values('datetime').set_index('datetime')
    print(f'Raw rows      : {len(df):,}')
    print(f'Range         : {df.index.min()} -> {df.index.max()}')

    # Resample to 30 min (average of the two 15-min measurements)
    arr = df['PAC'].resample('30min').mean().to_numpy()
    print(f'30-min rows   : {len(arr):,}  ({len(arr)/48/365.25:.2f} years)')
    print(f'Mean={arr.mean():.1f} W,  max={arr.max():.1f} W')

    # Save the cache
    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(CACHE_NPY, arr)
    print(f'Saved {CACHE_NPY}  ({arr.nbytes/1e6:.1f} MB, {len(arr):,} samples)')

    # Check the size is sufficient for full mode (2 years)
    N_FULL = round(2 * 365.25 * 48)
    print(f'Required for full run : {N_FULL:,}')
    print(f'Available : {len(arr):,}')
    assert len(arr) >= N_FULL, 'Not enough samples for the full-mode run.'


if __name__ == '__main__':
    main()
