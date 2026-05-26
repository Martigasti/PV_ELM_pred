"""
Lance les scripts blend et les baselines ELM/AR avec les vraies donnees pretraitees.

Usage :
    python run_full.py                    # lance tous les scripts d'affilee
    python run_full.py blend_opt          # uniquement BLEND optimise (lambda OLS)
    python run_full.py blend_corr         # uniquement BLEND correlation (formule papier)
    python run_full.py ols                # uniquement la variante ELM OLS (pseudo-inverse)
    python run_full.py ridge              # uniquement la variante ELM ridge
    python run_full.py ar                 # uniquement la baseline lineaire AR-OLS

Prerequis :
    Lancer `python scripts/preprocessing.py` une fois. Cela ecrit
    `data/data_30min.npy`, que tous les scripts recuperent via leur
    `load_30min(...)`.
"""

import os
import subprocess
import sys
import time

HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
CACHE_NPY    = os.path.join(PROJECT_ROOT, 'data', 'data_30min.npy')

SCRIPTS = {
    'blend_opt' : 'blend_optimisation.py',
    'blend_corr': 'blend_correlation.py',
    'ridge'     : 'dr_elm_ridge.py',
    'ols'       : 'dr_elm_ols.py',
    'ar'        : 'dr_ar_ols.py',
}


def run_one(name: str) -> None:
    script = SCRIPTS[name]
    print(f'\n{"="*78}\n  Execution de {script}  (SMOKE_TEST=0)\n{"="*78}')
    env = {**os.environ, 'SMOKE_TEST': '0'}
    t0 = time.time()
    proc = subprocess.run([sys.executable, script], cwd=HERE, env=env)
    dt = time.time() - t0
    if proc.returncode != 0:
        sys.exit(f'{script} a echoue (code {proc.returncode}) apres {dt:.1f}s')
    print(f'\n--- {script} termine en {dt/60:.1f} min ---')


def main() -> None:
    if not os.path.exists(CACHE_NPY):
        print(f'ATTENTION : {CACHE_NPY} introuvable.')
        print('            Lancer d\'abord python scripts/preprocessing.py pour construire le cache.')
        print('            Repli sur le reechantillonnage CSV en interne (plus lent).')

    targets = sys.argv[1:] or list(SCRIPTS.keys())
    bad = [t for t in targets if t not in SCRIPTS]
    if bad:
        sys.exit(f'Cible(s) inconnue(s) : {bad}. Choisir parmi {list(SCRIPTS)}.')

    for name in targets:
        run_one(name)

    print('\nTermine.')


if __name__ == '__main__':
    main()
