"""
Prétraitement du jeu de données PV_AC Palaiseau.

Lit `PV_AC_20200801_20250706_Palaiseau.csv`, rééchantillonne la colonne PAC
en moyennes sur 30 minutes, et sauvegarde le tableau 1D résultant dans
`data_30min.npy`. Les scripts dans scripts/ consomment ce cache via leur
helper `load_30min`.
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
    # Chargement des données brutes (pas de temps de 15 min)
    df = pd.read_csv(CSV_FILE)
    df['datetime'] = pd.to_datetime(
        df['datetime'].astype(str).str.split('+').str[0],
        format='%Y-%m-%d %H:%M:%S',
    )
    df = df.sort_values('datetime').set_index('datetime')
    print(f'Lignes brutes : {len(df):,}')
    print(f'Plage         : {df.index.min()} -> {df.index.max()}')

    # Rééchantillonnage à 30 min (moyenne des deux mesures de 15 min)
    arr = df['PAC'].resample('30min').mean().to_numpy()
    print(f'Lignes 30-min : {len(arr):,}  ({len(arr)/48/365.25:.2f} annees)')
    print(f'Moy={arr.mean():.1f} W,  max={arr.max():.1f} W')

    # Sauvegarde du cache
    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(CACHE_NPY, arr)
    print(f'Sauve {CACHE_NPY}  ({arr.nbytes/1e6:.1f} MB, {len(arr):,} echantillons)')

    # Vérification que la taille est suffisante pour le mode complet (2 ans)
    N_FULL = round(2 * 365.25 * 48)
    print(f'Requis pour run complet : {N_FULL:,}')
    print(f'Disponible : {len(arr):,}')
    assert len(arr) >= N_FULL, 'Pas assez d\'echantillons pour le run en mode complet.'


if __name__ == '__main__':
    main()
