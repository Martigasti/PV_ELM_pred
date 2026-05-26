"""
Baselines de persistance (P, Pcyclic, BLEND) sur l'ensemble des données.

Poids du BLEND appris par phase en minimisant l'erreur quadratique sur le
jeu d'entraînement (moindres carrés avec solution analytique) :

    lam* = sum((Pc - y_true)(Pc - P)) / sum((Pc - P)^2)   borné dans [0, 1]

où P(t) = y(t) et Pc(t) = y(t + FH - T_period). Lambda est estimé
directement à partir des données d'entraînement en minimisant la MSE,
plutôt que dérivé de la structure de corrélation comme dans la formule de
l'article (voir blend_correlation.py).
"""
import os
import time
from math import floor
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
    build_metric_row,
    compute_is_day_mask,
    load_30min,
    predict_blend,
    predict_cyclic_persistence,
    sertomat,
    split_and_save,
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
OUT_FILE_ALL = RESULTS_DIR / "Results_BLEND_optimisation_all.csv"
OUT_FILE_DAY = RESULTS_DIR / "Results_BLEND_optimisation_day.csv"
PRED_FILE = RESULTS_DIR / "Predictions_BLEND_optimisation.csv"


# ============================================================================
# BLEND : lam appris par phase par moindres carres sur le train
# ============================================================================
def fit_blend_lambda_per_phase(
    data_train: np.ndarray, FH: int, T_period: int
) -> np.ndarray:
    """Solution analytique lam* = sum((Pc - y)(Pc - P)) / sum((Pc - P)^2), bornée, par phase."""
    N = len(data_train)
    lam_phase = np.zeros(T_period)
    for k_tgt in range(1, T_period + 1):
        start = (k_tgt - 1 - FH) % T_period
        idx_t = np.arange(start, N - FH, T_period)
        # On a besoin que Pc(t) = data[t + FH - T_period] soit dans les bornes
        # => t >= T_period - FH
        idx_t = idx_t[idx_t >= T_period - FH]
        # On a besoin d'au moins 3 points, sinon on replie sur λ=0.5 (poids égal)
        if len(idx_t) < 3:
            lam_phase[k_tgt - 1] = 0.5
            continue
        y_true = data_train[idx_t + FH]
        P_pred = data_train[idx_t]
        Pc_pred = data_train[idx_t + FH - T_period]
        diff = Pc_pred - P_pred
        denom = np.sum(diff * diff)
        # P et P° identiques sur cette phase : λ indéfini, on prend 0.5 par convention
        if denom < 1e-12:
            lam_phase[k_tgt - 1] = 0.5
            continue
        lam = np.sum((Pc_pred - y_true) * diff) / denom
        lam_phase[k_tgt - 1] = max(0.0, min(1.0, lam))
    return lam_phase


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
    idx_split = floor(ratio * PVin.shape[0])
    y_test = PVout[idx_split:]
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

    # ---- BLEND
    # La tranche de train se termine à idx_split + LB + FH - 1 (inclus) pour
    # que la dernière cible de train reste dans l'échantillon
    # => tranche [:idx_split + LB + FH]
    data_tr_raw = data[: idx_split + LB + FH]
    lam_phase = fit_blend_lambda_per_phase(data_tr_raw, FH, T_period)

    y_pred_BL = predict_blend(
        y_pred_P, y_pred_Pc, lam_phase, offset_base, T_period
    )
    print(
        f"[BLEND] lam_phase: min={lam_phase.min():.3f} "
        f"max={lam_phase.max():.3f} mean={lam_phase.mean():.3f}"
    )
    rows.append(
        build_metric_row(
            "BLEND_opti", LB, FH, Persis_simple_test, y_test, y_pred_BL,
            mask_day_test, extra_fields={"N_params": 0},
        )
    )
    log_predictions("BLEND_opti", y_test, y_pred_BL)

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
    print(f"\n[chrono] blend_optimisation.py : {_dt:.2f} s  ({_dt/60:.2f} min)")
