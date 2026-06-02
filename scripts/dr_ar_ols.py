"""
AR-OLS forecasting on the PV_AC Palaiseau data — linear baseline for ELM.

Same inputs as dr_elm_ols.py (window of LB lags + 4 cyclic time
features), but without a hidden layer (non-linearity).

Comparing AR-OLS to ELM quantifies what the sigmoid transfer
functions add relative to a purely linear model on the same
features.

Models reported per (LB, FH):
    - Persistence_P : y_pred = PAC(t)
    - Persistence_Pcyclic  : y_pred = PAC(t + FH - 24h)
    - BLEND_opti : convex least-squares combination of the two persistences
    - AR_OLS : linear OLS on [LB lags + 4 time features]
"""
import time
import numpy as np
import pandas as pd

from elm_common import (
    CACHE_NPY,
    CSV_FILE,
    FH_list,
    LB_list,
    Ndata,
    RESULTS_DIR,
    baseline_rows,
    compute_is_day_mask,
    load_30min,
    make_log_predictions,
    prepare_split,
)
from utils import build_metric_row, split_and_save


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
    # Clip negative values to 0 because PAC is a physical power
    return np.clip(X_aug @ beta, a_min=0.0, a_max=None)


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

    split = prepare_split(data, is_day_full, LB, FH, use_time_features=True)
    pred_rows: list[pd.DataFrame] = []
    log_predictions = make_log_predictions(LB, FH, pred_rows)

    # ---- Baselines P / P° / BLEND (shared with the ELM scripts)
    rows = baseline_rows(data, LB, FH, split, log_predictions)

    # ---- AR-OLS
    print("  [AR_OLS]")
    beta = train_ar_ols(split["X_train"], split["y_train"])
    y_pred_ar = ar_predict(split["X_test"], beta)
    rows.append(
        build_metric_row(
            "AR_OLS", LB, FH, split["Persis_simple_test"], split["y_test"], y_pred_ar,
            split["mask_day_test"], extra_fields={"N_params": beta.size},
        )
    )
    log_predictions("AR_OLS", split["y_test"], y_pred_ar)

    return rows, pred_rows


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    data = load_30min(CSV_FILE, CACHE_NPY, n_rows=Ndata)
    print(f"Data: {len(data)} points ({len(data)/48/365.25:.2f} years)")

    is_day_full = compute_is_day_mask(len(data))
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
    print(f"\n[chrono] dr_ar_ols.py : {_dt:.2f} s  ({_dt/60:.2f} min)")
