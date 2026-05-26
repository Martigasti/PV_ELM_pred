"""
Baselines de persistance (P, Pcyclic, BLEND) sur l'ensemble des données.

Poids du BLEND estimé via la formule de l'article (eq. 17) :
    lambda_{phi,h} = 0.5 * (1 + rho_{phi,h})
où rho est la corrélation empirique conditionnée par phase entre P et la
cible, calculée sur la période de calibration (reproduction fidèle de Matlab).
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
OUT_FILE_ALL = RESULTS_DIR / "Results_BLEND_correlation_all.csv"
OUT_FILE_DAY = RESULTS_DIR / "Results_BLEND_correlation_day.csv"
PRED_FILE = RESULTS_DIR / "Predictions_BLEND_correlation.csv"


# ============================================================================
# BLEND : rho par phase cible (reproduction Matlab)
# ============================================================================
def compute_cyclic_correlation(
    data_train: np.ndarray, FH: int, T_period: int
) -> np.ndarray:
    """rho_phase[phi] = Pearson(P(t), y(t+FH)) sur les échantillons dont la cible est en phase phi."""
    N = len(data_train)
    rho_phase = np.zeros(T_period)

    for phi in range(T_period):
        # On veut (t + FH) mod T == phi  =>  t mod T == (phi - FH) mod T
        start = (phi - FH) % T_period
        idx_t = np.arange(start, N - FH, T_period)
        if len(idx_t) < 3:
            continue

        P_pred = data_train[idx_t]
        y_true = data_train[idx_t + FH]

        std_P = P_pred.std()
        std_y = y_true.std()
        if std_P < 1e-12 or std_y < 1e-12:
            continue

        rho = np.mean((P_pred - P_pred.mean()) * (y_true - y_true.mean())) / (
            std_P * std_y
        )
        rho_phase[phi] = rho

    return rho_phase


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

    # ---- BLEND.
    # La tranche de train se termine à idx_split + LB + FH - 1 (inclus) pour
    # que la dernière cible de train reste dans l'échantillon
    # => tranche [:idx_split + LB + FH].
    data_tr_raw = data[: idx_split + LB + FH]
    rho_phase = compute_cyclic_correlation(data_tr_raw, FH, T_period)
    # Formule de l'article : lam = clip(0.5 * (1 + rho), 0, 1) par phase
    lam_phase = np.clip(0.5 * (1.0 + rho_phase), 0.0, 1.0)

    y_pred_BL = predict_blend(
        y_pred_P, y_pred_Pc, lam_phase, offset_base, T_period
    )

    rows.append(
        build_metric_row(
            "BLEND_corre", LB, FH, Persis_simple_test, y_test, y_pred_BL,
            mask_day_test, extra_fields={"N_params": 0},
        )
    )
    log_predictions("BLEND_corre", y_test, y_pred_BL)

    return rows, pred_rows


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    data = load_30min(CSV_FILE, CACHE_NPY, n_rows=Ndata)
    print(f"Donnees : {len(data)} points ({len(data)/48/365.25:.2f} ans)")

    is_day_full = compute_is_day_mask(len(data))
    print(
        f"Day mask : {is_day_full.sum()}/{len(is_day_full)} pas "
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
    print(f"\n[chrono] blend_correlation.py : {_dt:.2f} s  ({_dt/60:.2f} min)")