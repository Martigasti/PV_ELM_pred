"""
Persistence baselines (P, Pcyclic, BLEND) over the whole dataset.

BLEND weights learned per phase by minimizing the squared error on the
training set (least squares with analytical solution):

    lam* = sum((Pc - y_true)(Pc - P)) / sum((Pc - P)^2)   clipped to [0, 1]

where P(t) = y(t) and Pc(t) = y(t + FH - T_period). Lambda is estimated
directly from the training data by minimizing the MSE,
rather than derived from the correlation structure (see blend_correlation.py).
"""
import os
import time
from math import floor

import numpy as np
import pandas as pd

from dataset_config import CACHE_NPY, CSV_FILE, NDATA_FULL, RESULTS_DIR, day_mask
from utils import (
    build_metric_row,
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
    print("*** SMOKE TEST MODE ***")
    Ndata    = 1000
    LB_list  = [48]
    FH_list  = [1, 12]
    ratio    = 0.50
    T_period = 48
else:
    print("*** FULL MODE ***")
    Ndata    = NDATA_FULL
    LB_list  = [48]
    FH_list  = [1, 2, 6, 12, 20]
    ratio    = 0.50
    T_period = 48

OUT_FILE_ALL = RESULTS_DIR / "Results_BLEND_optimisation_all.csv"
OUT_FILE_DAY = RESULTS_DIR / "Results_BLEND_optimisation_day.csv"
PRED_FILE = RESULTS_DIR / "Predictions_BLEND_optimisation.csv"


# ============================================================================
# BLEND: lam learned per phase by least squares on the train set
# ============================================================================
def fit_blend_lambda_per_phase(
    data_train: np.ndarray, FH: int, T_period: int
) -> np.ndarray:
    """Analytical solution lam* = sum((Pc - y)(Pc - P)) / sum((Pc - P)^2), clipped, per phase."""
    N = len(data_train)
    lam_phase = np.zeros(T_period)
    for k_tgt in range(1, T_period + 1):
        start = (k_tgt - 1 - FH) % T_period
        idx_t = np.arange(start, N - FH, T_period)
        # We need Pc(t) = data[t + FH - T_period] to be within bounds => t >= T_period - FH
        idx_t = idx_t[idx_t >= T_period - FH]
        # We need at least 3 points, otherwise fall back to λ=0.5 (equal weight)
        if len(idx_t) < 3:
            lam_phase[k_tgt - 1] = 0.5
            continue
        y_true = data_train[idx_t + FH]
        P_pred = data_train[idx_t]
        Pc_pred = data_train[idx_t + FH - T_period]
        diff = Pc_pred - P_pred
        denom = np.sum(diff * diff)
        # P and P° identical on this phase: λ undefined, take 0.5 by convention
        if denom < 1e-12:
            lam_phase[k_tgt - 1] = 0.5
            continue
        lam = np.sum((Pc_pred - y_true) * diff) / denom
        lam_phase[k_tgt - 1] = max(0.0, min(1.0, lam))
    return lam_phase


# ============================================================================
# RUN ONE CONFIG (LB, FH)
# ============================================================================
def run_one(
    data: np.ndarray,
    is_day_full: np.ndarray,
    LB: int,
    FH: int,
) -> tuple[list[dict], list[pd.DataFrame]]:
    print(f"\n=== LB={LB} ({LB/48:g}d) | FH={FH} ({FH*0.5:.1f}h) ===")

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

    # ---- Persistence P
    y_pred_P = Persis_simple_test
    rows.append(
        build_metric_row(
            "Persistence_P", LB, FH, Persis_simple_test, y_test, y_pred_P,
            mask_day_test, extra_fields={"N_params": 0},
        )
    )
    log_predictions("Persistence_P", y_test, y_pred_P)

    # ---- Cyclic persistence
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
    # The train slice ends at idx_split + LB + FH - 1 (inclusive) so that
    # the last train target stays within the sample
    # => slice [:idx_split + LB + FH]
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
    print(f"Data: {len(data)} points ({len(data)/48/365.25:.2f} years)")

    is_day_full = day_mask(len(data))
    print(
        f"Day mask  : {is_day_full.sum()}/{len(is_day_full)} steps "
        f"({is_day_full.mean()*100:.1f}% day)"
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
    print("\n===== RESULTS (all samples) =====")
    print(df_all.to_string())
    print("\n===== RESULTS (day only) =====")
    print(df_day.to_string())
    print(f"\nResults _all : {OUT_FILE_ALL}")
    print(f"Results _day : {OUT_FILE_DAY}")

    predictions = pd.concat(all_pred_rows, ignore_index=True)
    predictions.to_csv(PRED_FILE, index=False)
    print(f"Predictions saved: {PRED_FILE}")


if __name__ == "__main__":
    _t0 = time.time()
    main()
    _dt = time.time() - _t0
    print(f"\n[chrono] blend_optimisation.py : {_dt:.2f} s  ({_dt/60:.2f} min)")
