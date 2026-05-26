"""
Prévision ELM sur les données PV_AC Palaiseau, avec la pseudo-inverse (OLS).

Poids de sortie obtenus avec la pseudo-inverse de Moore-Penrose :

    beta = pinv(H) @ y         minimise ||H beta - y||^2

Modèles reportés par (LB, FH) :
    - Persistence_P : y_pred = PAC(t)
    - Persistence_Pcyclic : y_pred = PAC(t + FH - 24h)
    - BLEND_opti : combinaison convexe par moindres carrés des deux persistances
    - ELM : ELM par pseudo-inverse sur [LB retards + 4 features temporelles]
"""
import os
import time
from math import floor, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from blend_optimisation import fit_blend_lambda_per_phase
from utils import (
    build_metric_row,
    compute_is_day_mask,
    load_30min,
    predict_blend,
    predict_cyclic_persistence,
    sertomat,
    split_and_save,
    time_features_for_targets,
)


# ============================================================================
# CONFIG
# ============================================================================
SMOKE_TEST = os.environ.get("SMOKE_TEST", "1") == "1"

if SMOKE_TEST:
    print("*** MODE TEST DE FUMEE ***")
    Ndata                 = 1000
    LB_list               = [48]
    FH_list               = [1, 12]
    N_ELM_candidates_list = [2, 4]
    N_ELM_hidden_list     = [4, 8]
    ratio                 = 0.50
    T_period              = 48
    VAL_RATIO             = 0.20
else:
    print("*** MODE COMPLET ***")
    Ndata                 = round(2 * 365.25 * 48)
    LB_list               = [48]
    FH_list               = [1, 2, 6, 12, 20]
    N_ELM_candidates_list = [100, 500]
    N_ELM_hidden_list     = [100, 200, 500]
    ratio                 = 0.50
    T_period              = 48
    VAL_RATIO             = 0.20

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_FILE = PROJECT_ROOT / "data" / "PV_AC_20200801_20250706_Palaiseau.csv"
CACHE_NPY = PROJECT_ROOT / "data" / "data_30min.npy"
RESULTS_DIR = PROJECT_ROOT / "results"
OUT_FILE_ALL = RESULTS_DIR / "Results_DR_ELM_NICE_sklearn_simple_all.csv"
OUT_FILE_DAY = RESULTS_DIR / "Results_DR_ELM_NICE_sklearn_simple_day.csv"
PRED_FILE = RESULTS_DIR / "Predictions_DR_ELM_NICE_sklearn_simple.csv"
SEED = 42

RNG = np.random.default_rng(SEED)


# ============================================================================
# ENTRAINEMENT ELM (pseudo-inverse / OLS)
# ============================================================================
def elm_sigmoid(X: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-X))


def train_elm(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden: int,
    n_candidates: int,
    rng: np.random.Generator,
    val_ratio: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # La validation sur une queue mise de côté empêche la sélection (IW, bias)
    # d'être récompensée uniquement par un n_hidden plus grand (ce qui réduit
    # mécaniquement le RMSE d'entraînement).
    in_size = X.shape[1]
    n_train = X.shape[0]
    n_fit = max(1, floor((1.0 - val_ratio) * n_train))
    X_fit, X_val = X[:n_fit], X[n_fit:]
    y_fit, y_val = y[:n_fit], y[n_fit:]

    best_val, best_IW, best_bias = np.inf, None, None
    for _ in range(n_candidates):
        IW = rng.uniform(-1.0, 1.0, size=(n_hidden, in_size))
        bias = rng.uniform(0.0, 1.0, size=n_hidden)
        H_fit = elm_sigmoid(X_fit @ IW.T + bias)
        H_val = elm_sigmoid(X_val @ IW.T + bias)
        beta, *_ = np.linalg.lstsq(H_fit, y_fit, rcond=None)
        y_val_pred = np.clip(H_val @ beta, a_min=0.0, a_max=None)
        val_rmse = sqrt(np.mean((y_val_pred - y_val) ** 2))
        if val_rmse < best_val:
            best_val, best_IW, best_bias = val_rmse, IW, bias

    H_full = elm_sigmoid(X @ best_IW.T + best_bias)
    best_beta, *_ = np.linalg.lstsq(H_full, y, rcond=None)
    return best_beta, best_IW, best_bias, best_val


def train_elm_grid(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden_list: list[int],
    n_candidates_list: list[int],
    rng: np.random.Generator,
    val_ratio: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    best_val = np.inf
    best_beta = best_IW = best_bias = None
    best_h = best_c = None
    for n_hidden in n_hidden_list:
        for n_candidates in n_candidates_list:
            beta, IW, bias, val_rmse = train_elm(
                X, y, n_hidden, n_candidates, rng, val_ratio=val_ratio
            )
            print(
                f"    n_hidden={n_hidden:4d}  n_cand={n_candidates:4d}  "
                f"val_RMSE={val_rmse:.4g}"
            )
            if val_rmse < best_val:
                best_val, best_beta, best_IW, best_bias = val_rmse, beta, IW, bias
                best_h, best_c = n_hidden, n_candidates
    print(
        f"    -> retenu: n_hidden={best_h}  n_cand={best_c}  val_RMSE={best_val:.4g}"
    )
    return best_beta, best_IW, best_bias, best_h, best_c


def elm_predict(
    X: np.ndarray, beta: np.ndarray, IW: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    # PAC est une puissance physique : on borne les valeurs négatives à 0.
    return np.clip(elm_sigmoid(X @ IW.T + bias) @ beta, a_min=0.0, a_max=None)


# ============================================================================
# EXECUTION D'UNE CONFIG (LB, FH)
# ============================================================================
def run_one(
    data: np.ndarray,
    is_day_full: np.ndarray,
    LB: int,
    FH: int,
) -> tuple[list[dict], list[pd.DataFrame]]:
    print(f"\n=== LB={LB} ({LB/48:g}j) | FH={FH} ({FH*0.5:.1f}h) ===")

    PVin, PVout = sertomat(data, LB, FH)
    mu_in = PVin.mean(axis=0)
    sd_in = PVin.std(axis=0, ddof=1)
    sd_in = np.where(sd_in == 0, 1.0, sd_in)
    PVin_norm = (PVin - mu_in) / sd_in

    tfeat = time_features_for_targets(PVin_norm.shape[0], LB, FH)
    PVin_norm = np.concatenate([PVin_norm, tfeat], axis=1)

    idx_split = floor(ratio * PVin_norm.shape[0])
    X_train, X_test = PVin_norm[:idx_split], PVin_norm[idx_split:]
    y_train, y_test = PVout[:idx_split], PVout[idx_split:]
    n_test = len(y_test)

    Persis_simple_test = PVin[idx_split:, -1]
    offset_base = idx_split + LB + FH - 1
    mask_day_test = is_day_full[offset_base : offset_base + n_test]

    rows: list[dict] = []
    pred_rows: list[pd.DataFrame] = []

    def log_predictions(method: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        pred_rows.append(
            pd.DataFrame(
                {
                    "Method": method,
                    "LB_days": LB / 48,
                    "FH_hours": FH * 0.5,
                    "t_index": np.arange(len(y_true)),
                    "y_true": np.asarray(y_true).ravel(),
                    "y_pred": np.asarray(y_pred).ravel(),
                }
            )
        )

    # ---- Persistance P
    y_pred_P = Persis_simple_test
    rows.append(
        build_metric_row(
            "Persistence_P", LB, FH, Persis_simple_test, y_test, y_pred_P,
            mask_day_test, extra_fields={"N_params": 0},
        )
    )
    log_predictions("Persistence_P", y_test, y_pred_P)

    # ---- Persistance cyclique
    y_pred_Pc = predict_cyclic_persistence(
        data, offset_base, n_test, T_period, fallback=y_pred_P
    )
    rows.append(
        build_metric_row(
            "Persistence_Pcyclic", LB, FH, Persis_simple_test, y_test, y_pred_Pc,
            mask_day_test, extra_fields={"N_params": 0},
        )
    )
    log_predictions("Persistence_Pcyclic", y_test, y_pred_Pc)

    # ---- BLEND. La tranche de train se termine à idx_split + LB + FH - 1 (inclus).
    data_tr_raw = data[: idx_split + LB + FH]
    lam_phase = fit_blend_lambda_per_phase(data_tr_raw, FH, T_period)
    y_pred_BL = predict_blend(y_pred_P, y_pred_Pc, lam_phase, offset_base, T_period)
    print(
        f"    [BLEND] lam_phase: min={lam_phase.min():.3f} "
        f"max={lam_phase.max():.3f} mean={lam_phase.mean():.3f}"
    )
    rows.append(
        build_metric_row(
            "BLEND_opti", LB, FH, Persis_simple_test, y_test, y_pred_BL,
            mask_day_test, extra_fields={"N_params": 0},
        )
    )
    log_predictions("BLEND_opti", y_test, y_pred_BL)

    # ---- ELM (grille sur n_hidden, n_candidates)
    # RNG dérivé de (SEED, FH) pour que chaque horizon soit reproductible
    # indépendamment des autres (sinon l'état du RNG global propage les
    # modifications en amont).
    rng_fh = np.random.default_rng(SEED + FH)
    print("  [ELM_full]")
    beta, IW, bias, sel_h, sel_c = train_elm_grid(
        X_train, y_train,
        N_ELM_hidden_list, N_ELM_candidates_list,
        rng_fh, val_ratio=VAL_RATIO,
    )
    nParams_full = sel_h * (LB + 4) + sel_h + sel_h
    y_pred_f = elm_predict(X_test, beta, IW, bias)
    rows.append(
        build_metric_row(
            "ELM", LB, FH, Persis_simple_test, y_test, y_pred_f,
            mask_day_test,
            extra_fields={
                "N_params": nParams_full,
                "n_hidden": sel_h,
                "n_candidates": sel_c,
            },
        )
    )
    log_predictions("ELM", y_test, y_pred_f)

    return rows, pred_rows


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    data = load_30min(CSV_FILE, CACHE_NPY, n_rows=Ndata)
    print(f"Donnees : {len(data)} points ({len(data)/48/365.25:.2f} ans)")

    is_day_full = compute_is_day_mask(len(data))
    print(
        f"Day mask  : {is_day_full.sum()}/{len(is_day_full)} pas "
        f"({is_day_full.mean()*100:.1f}% jour)"
    )

    all_rows: list[dict] = []
    all_pred_rows: list[pd.DataFrame] = []
    for LB in LB_list:
        for FH in FH_list:
            rows, pred_rows = run_one(data, is_day_full, LB, FH)
            all_rows.extend(rows)
            all_pred_rows.extend(pred_rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_all, df_day = split_and_save(
        all_rows,
        extra_cols=["N_params", "n_hidden", "n_candidates"],
        out_path_all=str(OUT_FILE_ALL),
        out_path_day=str(OUT_FILE_DAY),
    )
    print("\n===== RESULTATS (tous les echantillons) =====")
    print(df_all.to_string())
    print("\n===== RESULTATS (jour uniquement) =====")
    print(df_day.to_string())
    print(f"\nResultats _all : {OUT_FILE_ALL}")
    print(f"Resultats _day : {OUT_FILE_DAY}")

    predictions = pd.concat(all_pred_rows, ignore_index=True)
    predictions.to_csv(PRED_FILE, index=False)
    print(f"Predictions sauvegardees : {PRED_FILE}")


if __name__ == "__main__":
    _t0 = time.time()
    main()
    _dt = time.time() - _t0
    print(f"\n[chrono] dr_elm_ols.py : {_dt:.2f} s  ({_dt/60:.2f} min)")
