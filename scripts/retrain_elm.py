"""
Fichier pour re-entrainer les ELM avec ses hyperparamètres optimales
et observer ses temps respectives d'éxécution.

Mesure, pour chaque horizon (FH) et chaque variante (OLS / Ridge) :
    - t_fit : temps d'un fit unique avec les meilleurs hyperparamètres déjà
              identifiés (n_hidden, n_candidates, lambda).
    - t_predict : temps d'inférence sur l'ensemble de test.
    - t_per_pred : temps d'inférence par échantillon.
"""
import os
import time
from math import floor
from pathlib import Path

import numpy as np

from utils import (
    compute_is_day_mask,
    load_30min,
    sertomat,
    time_features_for_targets,
)


# ============================================================================
# CONFIG
# ============================================================================
SMOKE_TEST = os.environ.get("SMOKE_TEST", "1") == "1"

if SMOKE_TEST:
    print("*** MODE TEST DE FUMEE ***")
    Ndata = 1000
    FH_list = [1, 12]
else:
    print("*** MODE COMPLET ***")
    Ndata = round(2 * 365.25 * 48)
    FH_list = [1, 2, 6, 12, 20]

LB = 48
ratio = 0.50
SEED = 42
N_REPEATS = 3  # moyenne sur N runs pour stabiliser la mesure

# Meilleurs hyperparamètres identifiés par le grid search (mode complet).
BEST_HPARAMS = {
    #     FH (pas 30 min) : hyperparamètres retenus par le grid search (mode complet)
    #     OLS et Ridge convergent vers les mêmes n_hidden/n_candidates
    1:  {"n_hidden": 500, "n_candidates": 500, "lambda_ridge": 10.0},
    2:  {"n_hidden": 500, "n_candidates": 500, "lambda_ridge": 10.0},
    6:  {"n_hidden": 500, "n_candidates": 100, "lambda_ridge": 10.0},
    12: {"n_hidden": 200, "n_candidates": 100, "lambda_ridge": 10.0},
    20: {"n_hidden": 200, "n_candidates": 100, "lambda_ridge": 10.0},
}


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_FILE = PROJECT_ROOT / "data" / "PV_AC_20200801_20250706_Palaiseau.csv"
CACHE_NPY = PROJECT_ROOT / "data" / "data_30min.npy"


# ============================================================================
# Entraînement ELM 
# ============================================================================
def elm_sigmoid(X: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-X))


def fit_elm_ols(
    X: np.ndarray, y: np.ndarray, n_hidden: int, n_candidates: int,
    rng: np.random.Generator, val_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Entraîne ELM OLS avec sélection sur n_candidates tirages aléatoires."""
    in_size = X.shape[1]
    n_fit = max(1, int((1.0 - val_ratio) * len(X)))
    X_fit, X_val = X[:n_fit], X[n_fit:]
    y_fit, y_val = y[:n_fit], y[n_fit:]
    best_rmse, best_IW, best_bias, best_beta = np.inf, None, None, None
    for _ in range(n_candidates):
        IW = rng.uniform(-1.0, 1.0, size=(n_hidden, in_size))
        bias = rng.uniform(0.0, 1.0, size=n_hidden)
        H_fit = elm_sigmoid(X_fit @ IW.T + bias)
        beta, *_ = np.linalg.lstsq(H_fit, y_fit, rcond=None)
        y_val_pred = np.clip(elm_sigmoid(X_val @ IW.T + bias) @ beta, 0.0, None)
        rmse = float(np.mean((y_val - y_val_pred) ** 2))
        if rmse < best_rmse:
            best_rmse, best_IW, best_bias, best_beta = rmse, IW, bias, beta
    # Refit sur X complet avec le meilleur (IW, bias)
    H_full = elm_sigmoid(X @ best_IW.T + best_bias)
    best_beta, *_ = np.linalg.lstsq(H_full, y, rcond=None)
    return best_beta, best_IW, best_bias


def fit_elm_ridge(
    X: np.ndarray, y: np.ndarray, n_hidden: int, n_candidates: int, lam: float,
    rng: np.random.Generator, val_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Entraîne ELM Ridge avec sélection sur n_candidates tirages aléatoires."""
    in_size = X.shape[1]
    n_fit = max(1, int((1.0 - val_ratio) * len(X)))
    X_fit, X_val = X[:n_fit], X[n_fit:]
    y_fit, y_val = y[:n_fit], y[n_fit:]
    best_rmse, best_IW, best_bias = np.inf, None, None
    for _ in range(n_candidates):
        IW = rng.uniform(-1.0, 1.0, size=(n_hidden, in_size))
        bias = rng.uniform(0.0, 1.0, size=n_hidden)
        H_fit = elm_sigmoid(X_fit @ IW.T + bias)
        A = H_fit.T @ H_fit + lam * np.eye(n_hidden)
        beta = np.linalg.solve(A, H_fit.T @ y_fit)
        y_val_pred = np.clip(elm_sigmoid(X_val @ IW.T + bias) @ beta, 0.0, None)
        rmse = float(np.mean((y_val - y_val_pred) ** 2))
        if rmse < best_rmse:
            best_rmse, best_IW, best_bias = rmse, IW, bias
    # Refit sur X complet avec le meilleur (IW, bias)
    H_full = elm_sigmoid(X @ best_IW.T + best_bias)
    A = H_full.T @ H_full + lam * np.eye(n_hidden)
    best_beta = np.linalg.solve(A, H_full.T @ y)
    return best_beta, best_IW, best_bias


def predict_elm(
    X: np.ndarray, beta: np.ndarray, IW: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    return np.clip(elm_sigmoid(X @ IW.T + bias) @ beta, a_min=0.0, a_max=None)


# ============================================================================
# BENCH UNE CONFIG
# ============================================================================
def bench_one(
    data: np.ndarray, FH: int, variant: str, n_repeats: int
) -> dict:
    hp = BEST_HPARAMS[FH]
    n_hidden = hp["n_hidden"]
    n_candidates = hp["n_candidates"]
    lam = hp["lambda_ridge"]

    PVin, PVout = sertomat(data, LB, FH)
    mu_in = PVin.mean(axis=0)
    sd_in = PVin.std(axis=0, ddof=1)
    sd_in = np.where(sd_in == 0, 1.0, sd_in)
    PVin_norm = (PVin - mu_in) / sd_in
    tfeat = time_features_for_targets(PVin_norm.shape[0], LB, FH)
    PVin_norm = np.concatenate([PVin_norm, tfeat], axis=1)

    idx_split = floor(ratio * PVin_norm.shape[0])
    X_train, X_test = PVin_norm[:idx_split], PVin_norm[idx_split:]
    y_train = PVout[:idx_split]

    fit_times = []
    pred_times = []
    for k in range(n_repeats):
        rng = np.random.default_rng(SEED + k)
        if variant == "OLS":
            t0 = time.perf_counter()
            beta, IW, bias = fit_elm_ols(X_train, y_train, n_hidden, n_candidates, rng)
            t_fit = time.perf_counter() - t0
        elif variant == "Ridge":
            t0 = time.perf_counter()
            beta, IW, bias = fit_elm_ridge(X_train, y_train, n_hidden, n_candidates, lam, rng)
            t_fit = time.perf_counter() - t0
        else:
            raise ValueError(variant)

        t0 = time.perf_counter()
        _ = predict_elm(X_test, beta, IW, bias)
        t_pred = time.perf_counter() - t0

        fit_times.append(t_fit)
        pred_times.append(t_pred)

    n_test = X_test.shape[0]
    return {
        "FH": FH,
        "FH_h": FH * 0.5,
        "variant": variant,
        "n_hidden": n_hidden,
        "n_candidates": n_candidates,
        "lambda": lam if variant == "Ridge" else None,
        "n_train": X_train.shape[0],
        "n_test": n_test,
        "t_fit_mean_s": float(np.mean(fit_times)),
        "t_fit_std_s": float(np.std(fit_times, ddof=1)) if n_repeats > 1 else 0.0,
        "t_predict_mean_s": float(np.mean(pred_times)),
        "t_per_pred_us": float(np.mean(pred_times)) / n_test * 1e6,
    }


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    data = load_30min(CSV_FILE, CACHE_NPY, n_rows=Ndata)
    print(f"Donnees : {len(data)} points ({len(data)/48/365.25:.2f} ans)")
    _ = compute_is_day_mask(len(data))  # préchauffage du cache
    print(f"Repeats par config : {N_REPEATS}\n")

    results = []
    for FH in FH_list:
        if FH not in BEST_HPARAMS:
            print(f"[skip] FH={FH} : pas d'hyperparam dans BEST_HPARAMS")
            continue
        for variant in ("OLS", "Ridge"):
            r = bench_one(data, FH, variant, N_REPEATS)
            results.append(r)
            lam_str = f"  lam={r['lambda']}" if r["lambda"] is not None else ""
            print(
                f"FH={r['FH_h']:>4.1f} h  {variant:5s}  "
                f"n_hidden={r['n_hidden']}  n_cand={r['n_candidates']}{lam_str}  "
                f"t_fit={r['t_fit_mean_s']:.3f}s "
                f"(±{r['t_fit_std_s']:.3f})  "
                f"t_predict={r['t_predict_mean_s']*1000:.1f} ms  "
                f"t_per_pred={r['t_per_pred_us']:.1f} us"
            )

    # Recap
    print("\n===== RECAP =====")
    print(f"{'FH (h)':>6} {'variant':>7} {'n_h':>4} {'n_cand':>6} {'t_fit (s)':>10} {'t_pred (ms)':>12} {'µs/pred':>8}")
    print("-" * 65)
    for r in results:
        print(
            f"{r['FH_h']:>6.1f} {r['variant']:>7} {r['n_hidden']:>4d} {r['n_candidates']:>6d} "
            f"{r['t_fit_mean_s']:>10.3f} {r['t_predict_mean_s']*1000:>12.1f} "
            f"{r['t_per_pred_us']:>8.1f}"
        )

    total_fit = sum(r["t_fit_mean_s"] for r in results) / 2  # 5 FH par variante, on veut le total par variante
    print(
        f"\nTotal fit (somme des 5 FH) — OLS  : "
        f"{sum(r['t_fit_mean_s'] for r in results if r['variant']=='OLS'):.2f} s"
    )
    print(
        f"Total fit (somme des 5 FH) — Ridge: "
        f"{sum(r['t_fit_mean_s'] for r in results if r['variant']=='Ridge'):.2f} s"
    )
    print(
        "\nReference grid search (mode complet, 5 FH) : "
        "ELM OLS ~228 min  /  ELM Ridge ~118 min" # Pas automatisé, noté d'après le grid search dans run_full.py
    )


if __name__ == "__main__":
    main()
