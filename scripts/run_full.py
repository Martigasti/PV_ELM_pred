"""
Run the blend scripts and the ELM/AR baselines with the real preprocessed data.

Usage:
    python run_full.py                    # run all scripts in sequence
    python run_full.py blend_opt          # only the optimized BLEND (lambda OLS)
    python run_full.py blend_corr         # only the correlation BLEND (paper formula)
    python run_full.py ols                # only the ELM OLS variant (pseudo-inverse)
    python run_full.py ridge              # only the ELM ridge variant
    python run_full.py ar                 # only the linear AR-OLS baseline
    python run_full.py robust_risk        # only the ELM Robust Risk variant
    python run_full.py tikhonov           # only the anisotropic Tikhonov ELM variant
    python run_full.py box_cox            # only the ELM Box-Cox variant (Ridge on transformed target)
    python run_full.py mae                # only the ELM-MAE variant (IRLS, L1 cost)
    python run_full.py log_mse            # only the ELM-Log-MSE variant (IRLS, log of the target)
    python run_full.py huber              # only the ELM-Huber variant (IRLS, delta adaptive via MAD)
    python run_full.py l3                 # only the ELM-L3 variant (cubic norm, 2-pass)
    python run_full.py corr               # only the Corr-ELM variant (temporal correlation of residuals, closed-form)
    python run_full.py elastic_net        # only the ELM Elastic Net variant (L1+L2, 2-pass linearized)
    python run_full.py m_estimator        # only the ELM M-Estimator variant (Welsch piecewise, strict 2-pass)
    python run_full.py lp                 # only the ELM L_p variant (general norm, strict 2-pass)
    python run_full.py glm                # only the ELM GLM variant (Gamma+log link, 2-pass Fisher linearized)
    python run_full.py timegpt            # only TimeGPT (Nixtla foundation model, zero-shot, subsampled)
    python run_full.py ols_nocyclic      # ablation: ELM OLS without the 4 cyclic features, FH=1
    python run_full.py ridge_nocyclic    # ablation: ELM Ridge without the 4 cyclic features, FH=1
    python run_full.py rr_nocyclic     # ablation: ELM Robust Risk without the 4 cyclic features, FH=1

TimeGPT is an external/API model: it is NOT run by default (the no-argument run
skips it), but stays launchable explicitly with `run_full.py timegpt`.
NeuralProphet is scored separately and is not part of this runner.

Preprocessing is automatic: if the dataset's cache (.npy) is missing, this runner
builds it first by calling the dispatcher `preprocessing.py` (which routes to the
CSV/NetCDF/meteo parser based on the dataset). So a single command does
everything for any declared dataset:
    DATASET=<x> python scripts/run_full.py
"""

import os
import subprocess
import sys
import time

HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

# Cache path follows the selected dataset (env var DATASET) so the
# "cache missing" pre-check below is correct for any series.
sys.path.insert(0, HERE)
from dataset_config import CACHE_NPY  # noqa: E402

SCRIPTS = {
    'blend_opt'  : 'blend_optimisation.py',
    'blend_corr' : 'blend_correlation.py',
    'ridge'      : 'dr_elm_ridge.py',
    'ols'        : 'dr_elm_ols.py',
    'ar'         : 'dr_ar_ols.py',
    'rr': 'dr_elm_robust_risk.py',
    'tikhonov'   : 'dr_elm_tikhonov.py',
    'box_cox'    : 'dr_elm_box_cox.py',
    'mae'        : 'dr_elm_mae.py',
    'log_mse'    : 'dr_elm_log_mse.py',
    'huber'      : 'dr_elm_huber.py',
    'l3'         : 'dr_elm_l3.py',
    'corr'       : 'dr_elm_corr.py',
    'elastic_net': 'dr_elm_elastic_net.py',
    'm_estimator': 'dr_elm_m_estimator.py',
    'lp'         : 'dr_elm_lp.py',
    'glm'        : 'dr_elm_glm.py',
    'timegpt'    : 'timegpt.py',
    'ols_nocyclic'  : 'dr_elm_znocyclic_ols.py',
    'ridge_nocyclic': 'dr_elm_znocyclic_ridge.py',
    'rr_nocyclic'   : 'dr_elm_znocyclic_rr.py',
}

# Targets run by the no-argument default: everything except the external/API
# model TimeGPT (still launchable explicitly via `run_full.py timegpt`).
DEFAULT_TARGETS = [k for k in SCRIPTS if k != 'timegpt']


def run_one(name: str) -> None:
    script = SCRIPTS[name]
    print(f'\n{"="*78}\n  Running {script}  (SMOKE_TEST=0)\n{"="*78}')
    env = {**os.environ, 'SMOKE_TEST': '0'}
    t0 = time.time()
    proc = subprocess.run([sys.executable, script], cwd=HERE, env=env)
    dt = time.time() - t0
    if proc.returncode != 0:
        sys.exit(f'{script} failed (code {proc.returncode}) after {dt:.1f}s')
    print(f'\n--- {script} finished in {dt/60:.1f} min ---')


def main() -> None:
    if not os.path.exists(CACHE_NPY):
        from dataset_config import DATASET
        print(f'{CACHE_NPY} not found (DATASET={DATASET}); building it via '
              'the preprocessing dispatcher...')
        env = {**os.environ, 'SMOKE_TEST': '0'}
        proc = subprocess.run([sys.executable, 'preprocessing.py'], cwd=HERE, env=env)
        if proc.returncode != 0:
            sys.exit(f'preprocessing failed (code {proc.returncode}) for {DATASET}')
        if not os.path.exists(CACHE_NPY):
            sys.exit(f'preprocessing ran but {CACHE_NPY} is still missing.')

    targets = sys.argv[1:] or DEFAULT_TARGETS
    bad = [t for t in targets if t not in SCRIPTS]
    if bad:
        sys.exit(f'Unknown target(s): {bad}. Choose from {list(SCRIPTS)}.')

    for name in targets:
        run_one(name)

    print('\nDone.')


if __name__ == '__main__':
    main()
