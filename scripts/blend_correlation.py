"""
Persistence baselines (P, Pcyclic, BLEND) over the whole dataset.

BLEND (paper eq. 16): x_hat_{t+h} = lambda * x_{t+h-T} + (1 - lambda) * x_t,
i.e. lambda weights the cyclic-persistence term P°.

BLEND weight estimated via the paper's formula (eq. 17):
    lambda_{phi,h} = 0.5 * (1 + rho_{phi,h})
where rho_{phi,h} is the empirical correlation between the current trajectory
x_t and the same phase of the previous day x_{t+h-T} (the P° term), conditioned
on the issue-time phase phi(t), computed over the calibration period.
"""
import os
import time
from math import floor

import numpy as np
import pandas as pd

from dataset_config import (
    CACHE_NPY,
    CLIP_NONNEG,
    CSV_FILE,
    NDATA_FULL,
    RESULTS_DIR,
    STEPS_PER_DAY,
    day_mask,
)
from utils import (
    apply_night_mask,
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


def _h_to_steps(hours):
    return [round(h * STEPS_PER_DAY / 24.0) for h in hours]


if SMOKE_TEST:
    print("*** SMOKE TEST MODE ***")
    Ndata    = round(1000 * STEPS_PER_DAY / 48)
    LB_list  = [STEPS_PER_DAY]
    FH_list  = _h_to_steps([0.5, 6])
    ratio    = 0.50
    T_period = STEPS_PER_DAY
else:
    print("*** FULL MODE ***")
    Ndata    = NDATA_FULL
    LB_list  = [STEPS_PER_DAY]
    FH_list  = _h_to_steps([0.5, 1, 3, 6, 10])
    ratio    = 0.50
    T_period = STEPS_PER_DAY

OUT_FILE_ALL = RESULTS_DIR / "Results_BLEND_correlation_all.csv"
OUT_FILE_DAY = RESULTS_DIR / "Results_BLEND_correlation_day.csv"
PRED_FILE = RESULTS_DIR / "Predictions_BLEND_correlation.csv"


# ============================================================================
# BLEND: rho per issue-time phase (paper eq. 17)
# ============================================================================
def compute_cyclic_correlation(
    data_train: np.ndarray, FH: int, T_period: int
) -> np.ndarray:
    """rho_phase[phi] = Pearson(x_t, x_{t+h-T}) over samples whose issue time t is in phase phi.

    Follows the paper (eq. 17): rho is the empirical correlation between the
    current trajectory x_t and the same phase of the previous day x_{t+h-T}
    (the cyclic-persistence term P°), conditioned on the issue-time phase phi(t).
    """
    N = len(data_train)
    rho_phase = np.zeros(T_period)

    for phi in range(T_period):
        # Issue-time phase: t mod T == phi
        idx_t = np.arange(phi, N - FH, T_period)
        # P° at the forecast time t+h is x_{t+h-T} => need idx_t + FH - T >= 0
        idx_t = idx_t[idx_t + FH - T_period >= 0]
        if len(idx_t) < 3:
            continue

        x_t = data_train[idx_t]
        x_cyc = data_train[idx_t + FH - T_period]

        std_t = x_t.std()
        std_cyc = x_cyc.std()
        if std_t < 1e-12 or std_cyc < 1e-12:
            continue

        rho = np.mean((x_t - x_t.mean()) * (x_cyc - x_cyc.mean())) / (
            std_t * std_cyc
        )
        rho_phase[phi] = rho

    return rho_phase


# ============================================================================
# RUN ONE CONFIG (LB, FH)
# ============================================================================
def run_one(
    data: np.ndarray,
    is_day_full: np.ndarray,
    LB: int,
    FH: int,
) -> tuple[list[dict], list[pd.DataFrame]]:
    print(
        f"\n=== LB={LB} ({LB/STEPS_PER_DAY:g}d) | "
        f"FH={FH} ({FH*24.0/STEPS_PER_DAY:.1f}h) ==="
    )

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
                    "LB_days": LB / STEPS_PER_DAY,
                    "FH_hours": FH * 24.0 / STEPS_PER_DAY,
                    "t_index": np.arange(len(y_true)),
                    "y_true": np.asarray(y_true).ravel(),
                    "y_pred": np.asarray(y_pred).ravel(),
                }
            )
        )

    # ---- Persistence P
    y_pred_P = apply_night_mask(Persis_simple_test, mask_day_test)
    rows.append(
        build_metric_row(
            "Persistence_P", LB, FH, Persis_simple_test, y_test, y_pred_P,
            mask_day_test, extra_fields={"N_params": 0}, steps_per_day=T_period,
        )
    )
    log_predictions("Persistence_P", y_test, y_pred_P)

    # ---- Cyclic persistence
    y_pred_Pc = predict_cyclic_persistence(
        data, offset_base, n_test, T_period, fallback=y_pred_P
    )
    y_pred_Pc = apply_night_mask(y_pred_Pc, mask_day_test)
    rows.append(
        build_metric_row(
            "Persistence_Pcyclic", LB, FH, Persis_simple_test, y_test, y_pred_Pc,
            mask_day_test, extra_fields={"N_params": 0}, steps_per_day=T_period,
        )
    )
    log_predictions("Persistence_Pcyclic", y_test, y_pred_Pc)

    # ---- BLEND.
    # The train slice ends at idx_split + LB + FH - 1 (inclusive) so that
    # the last train target stays within the sample
    # => slice [:idx_split + LB + FH].
    data_tr_raw = data[: idx_split + LB + FH]
    # rho indexed by issue-time phase phi(t) (paper eq. 17)
    rho_phase = compute_cyclic_correlation(data_tr_raw, FH, T_period)
    # lam (paper eq. 16/17) multiplies the cyclic term P°: y = lam*Pc + (1-lam)*P
    lam_phase = np.clip(0.5 * (1.0 + rho_phase), 0.0, 1.0)
    # Reindex from issue-time phase to target phase (predict_blend indexes by
    # target phase): phase_target = (phi + FH) mod T.
    lam_target = np.empty_like(lam_phase)
    for phi in range(T_period):
        lam_target[(phi + FH) % T_period] = lam_phase[phi]
    # predict_blend computes lam*P + (1-lam)*Pc; pass (1 - lam_target) so that
    # the effective blend is lam_target*Pc + (1-lam_target)*P (eq. 16).
    y_pred_BL = predict_blend(
        y_pred_P, y_pred_Pc, 1.0 - lam_target, offset_base, T_period
    )
    y_pred_BL = apply_night_mask(y_pred_BL, mask_day_test)

    rows.append(
        build_metric_row(
            "BLEND_corre", LB, FH, Persis_simple_test, y_test, y_pred_BL,
            mask_day_test, extra_fields={"N_params": 0}, steps_per_day=T_period,
        )
    )
    log_predictions("BLEND_corre", y_test, y_pred_BL)

    return rows, pred_rows


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    data = load_30min(CSV_FILE, CACHE_NPY, n_rows=Ndata, clip_nonneg=CLIP_NONNEG)
    print(f"Data: {len(data)} points ({len(data)/STEPS_PER_DAY/365.25:.2f} years)")

    is_day_full = day_mask(len(data))
    print(
        f"Day mask : {is_day_full.sum()}/{len(is_day_full)} steps "
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
    print(f"\n[chrono] blend_correlation.py : {_dt:.2f} s  ({_dt/60:.2f} min)")