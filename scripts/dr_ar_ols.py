"""
Prévision AR-OLS sur les données PV_AC Palaiseau — baseline linéaire pour ELM.

Mêmes entrées que dr_elm_ols.py (fenêtre de LB retards + 4 features
temporelles cycliques), mais sans couche cachée (non-linéarité).

Comparer AR-OLS à ELM quantifie ce que les fonctions de transfert
sigmoïdes apportent par rapport à un modèle purement linéaire sur les mêmes
features.

Modèles reportés par (LB, FH) :
    - Persistence_P : y_pred = PAC(t)
    - Persistence_Pcyclic  : y_pred = PAC(t + FH - 24h)
    - BLEND_opti : combinaison convexe par moindres carrés des deux persistances
    - AR_OLS : OLS linéaire sur [LB retards + 4 features temporelles]
"""
import os
import time
from math import floor
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
    Ndata    = 1000
    LB_list  = [48]
    FH_list  = [1, 12]
    ratio    = 0.50
    T_period = 48
else:
    print("*** MODE COMPLET ***")
    Ndata    = round(2 * 365.25 * 48)
    LB_list  = [48]
    FH_list  = [1, 2, 6, 12, 20]
    ratio    = 0.50
    T_period = 48

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_FILE = PROJECT_ROOT / "data" / "PV_AC_20200801_20250706_Palaiseau.csv"
CACHE_NPY = PROJECT_ROOT / "data" / "data_30min.npy"
RESULTS_DIR = PROJECT_ROOT / "results"
OUT_FILE_ALL = RESULTS_DIR / "Results_AR_OLS_all.csv"
OUT_FILE_DAY = RESULTS_DIR / "Results_AR_OLS_day.csv"
PRED_FILE = RESULTS_DIR / "Predictions_AR_OLS.csv"


# ============================================================================
# AR-OLS
# ============================================================================
def train_ar_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X_aug = np.column_stack([np.ones(X.shape[0]), X])
    beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    return beta


def ar_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    X_aug = np.column_stack([np.ones(X.shape[0]), X])
    # On borne les valeurs négatives à 0 car PAC est une puissance physique
    return np.clip(X_aug @ beta, a_min=0.0, a_max=None)


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

    # ---- BLEND. La tranche de train se termine à idx_split + LB + FH - 1 (inclus)
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

    # ---- AR-OLS
    print("  [AR_OLS]")
    beta = train_ar_ols(X_train, y_train)
    y_pred_ar = ar_predict(X_test, beta)
    rows.append(
        build_metric_row(
            "AR_OLS", LB, FH, Persis_simple_test, y_test, y_pred_ar,
            mask_day_test, extra_fields={"N_params": beta.size},
        )
    )
    log_predictions("AR_OLS", y_test, y_pred_ar)

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
        extra_cols=["N_params"],
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
    print(f"\n[chrono] dr_ar_ols.py : {_dt:.2f} s  ({_dt/60:.2f} min)")
