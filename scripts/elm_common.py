"""
Shared factored core of the ELM scripts (dr_elm_*.py).

All ELM scripts used to share the same code for: the SMOKE_TEST config
block, the elm_sigmoid / ridge_solve functions, the data preparation +
the three baselines (Persistence P, cyclic persistence P°, BLEND) in
run_one, the body of main() and the chrono block. This module factors out all
that common part.
Each dr_elm_<name>.py keeps only its formula (solver), its
train_<name>_grid (readable on its own) and its elm_predict, then calls run_elm().

What stays specific to each variant:
  - the solver (exact formula)
  - the hyperparameter grid(s)
  - train_<name>_grid, returning (beta, IW, bias, sel_dict, predict_fn)
  - elm_predict (identical everywhere except box_cox / glm which transform the target)
"""
import os
import time
from math import floor
from pathlib import Path

import numpy as np
import pandas as pd

from blend_optimisation import fit_blend_lambda_per_phase
from utils import (
    apply_night_mask,
    build_metric_row,
    load_30min,
    predict_blend,
    predict_cyclic_persistence,
    sertomat,
    split_and_save,
    time_features_for_targets,
)


# ============================================================================
# CONFIG (identical in all ELM scripts)
# ============================================================================
SMOKE_TEST = os.environ.get("SMOKE_TEST", "1") == "1"

# Dataset paths, site coords and full-mode window come from dataset_config
# (env var DATASET). NDATA_FULL is the full-mode point cap (None = whole series).
from dataset_config import (
    CACHE_NPY,
    CLIP_NONNEG,
    CSV_FILE,
    DATASET,
    NDATA_FULL,
    RESULTS_DIR,
    STEPS_PER_DAY,
    day_mask,
)

# Horizons expressed in hours (same physical horizons across step sizes), turned
# into step counts via STEPS_PER_DAY: 0.5/1/3/6/10 h -> {1,2,6,12,20} at 48
# steps/day (30 min), {3,6,18,36,60} at 144 steps/day (10 min).
def _h_to_steps(hours):
    return [round(h * STEPS_PER_DAY / 24.0) for h in hours]


if SMOKE_TEST:
    print("*** SMOKE TEST MODE ***")
    # Smoke window scales with the step size so LB=24h still leaves a usable
    # train/test split (1000 pts at 48 steps/day -> proportional at 144).
    Ndata                 = round(1000 * STEPS_PER_DAY / 48)
    LB_list               = [STEPS_PER_DAY]              # 24 h lookback
    FH_list               = _h_to_steps([0.5, 6])       # smoke: 2 horizons
    N_ELM_candidates_list = [2, 4]
    N_ELM_hidden_list     = [4, 8]
    ratio                 = 0.50
    T_period              = STEPS_PER_DAY
    CV_FOLDS              = 5
else:
    print("*** FULL MODE ***")
    Ndata                 = NDATA_FULL
    LB_list               = [STEPS_PER_DAY]             # 24 h lookback
    FH_list               = _h_to_steps([0.5, 1, 3, 6, 10])
    N_ELM_candidates_list = [100]
    N_ELM_hidden_list     = [500]
    ratio                 = 0.50
    T_period              = STEPS_PER_DAY
    CV_FOLDS              = 5

SEED = 42


def build_paths(slug: str) -> tuple[Path, Path, Path]:
    """Output paths following the pattern Results_DR_ELM_NICE_<slug>_*.csv."""
    return (
        RESULTS_DIR / f"Results_DR_ELM_{slug}_all.csv",
        RESULTS_DIR / f"Results_DR_ELM_{slug}_day.csv",
        RESULTS_DIR / f"Predictions_DR_ELM_{slug}.csv",
    )


# ============================================================================
# SHARED ELM BUILDING BLOCKS
# ============================================================================
def elm_sigmoid(X: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-X))


def ridge_solve(H: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    n_hidden = H.shape[1]
    A = H.T @ H + lam * np.eye(n_hidden)
    b = H.T @ y
    return np.linalg.solve(A, b)


# ============================================================================
# TEMPORAL CROSS-VALIDATION (expanding window) — hyperparameter selection
# ============================================================================
def temporal_cv_splits(n: int, k: int = 5) -> list[tuple[slice, slice]]:
    """Expanding-window CV splits over a chronological train set.

    Returns k-1 (fit, val) slice pairs: fold i has fit=[0:b_i], val=[b_i:b_{i+1}]
    with b_j = round(n*j/k). The validation block is always *after* the fit block
    (we never validate on the past) and the fit block stays contiguous (required
    by Corr-ELM's correlation matrix and by the cyclic time features).
    """
    bounds = [round(n * j / k) for j in range(k + 1)]
    return [
        (slice(0, bounds[i]), slice(bounds[i], bounds[i + 1]))
        for i in range(1, k)
        if bounds[i + 1] > bounds[i] and bounds[i] > 0
    ]


def select_by_temporal_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden: int,
    n_candidates: int,
    rng: np.random.Generator,
    hparam_combos: list[tuple],
    fit_score_fn,
    refit_fn,
    k: int = 5,
):
    """Select (IW, bias, hyperparams) by expanding-window temporal CV, then refit.

    For each random hidden layer (IW, bias) and each hyperparameter combo, the
    mean validation RMSE over the k-1 expanding-window folds is computed; the
    triple minimizing it is kept, and the final weights are refit on the *whole*
    train set with that combo.

    Contract of the variant-specific callbacks:
      fit_score_fn(X_fit, y_fit, X_val, y_val, IW, bias, combo) -> val_rmse
          trains beta on the fit slice and returns the RMSE on the val slice.
      refit_fn(X_full, y_full, IW, bias, combo) -> (beta, extra)
          refits beta on the full train set; `extra` carries any data-derived
          quantity selected alongside (e.g. Huber delta, GLM mu offset); use
          None when there is none.

    Returns (beta, IW, bias, combo, extra, best_val_mean).
    """
    in_size = X.shape[1]
    splits = temporal_cv_splits(X.shape[0], k)

    best_val = np.inf
    best_IW = best_bias = best_combo = None

    for _ in range(n_candidates):
        IW = rng.uniform(-1.0, 1.0, size=(n_hidden, in_size))
        bias = rng.uniform(0.0, 1.0, size=n_hidden)
        for combo in hparam_combos:
            scores = [
                fit_score_fn(X[fit], y[fit], X[val], y[val], IW, bias, combo)
                for fit, val in splits
            ]
            val_rmse = float(np.mean(scores))
            if val_rmse < best_val:
                best_val, best_IW, best_bias, best_combo = val_rmse, IW, bias, combo

    beta, extra = refit_fn(X, y, best_IW, best_bias, best_combo)
    return beta, best_IW, best_bias, best_combo, extra, best_val


# ============================================================================
# DATA PREPARATION + BASELINES (shared by ELM and AR-OLS)
# ============================================================================
def make_log_predictions(LB: int, FH: int, pred_rows: list[pd.DataFrame]):
    """Return a function that appends (Method, LB_days, FH_hours, t_index, y_true, y_pred)."""
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

    return log_predictions


def prepare_split(
    data: np.ndarray,
    is_day_full: np.ndarray,
    LB: int,
    FH: int,
    use_time_features: bool,
):
    """Build standardized (X_train, X_test, y_train, y_test) + evaluation metadata.

    Returns a dict with X_train/X_test/y_train/y_test, in_extra (4 if time
    features else 0), Persis_simple_test, offset_base, mask_day_test, idx_split.
    """
    PVin, PVout = sertomat(data, LB, FH)
    mu_in = PVin.mean(axis=0)
    sd_in = PVin.std(axis=0, ddof=1)
    sd_in = np.where(sd_in == 0, 1.0, sd_in)
    PVin_norm = (PVin - mu_in) / sd_in

    in_extra = 0
    if use_time_features:
        tfeat = time_features_for_targets(
            PVin_norm.shape[0], LB, FH, steps_per_day=T_period
        )
        PVin_norm = np.concatenate([PVin_norm, tfeat], axis=1)
        in_extra = 4

    idx_split = floor(ratio * PVin_norm.shape[0])
    n_test = PVin_norm.shape[0] - idx_split
    offset_base = idx_split + LB + FH - 1

    return {
        "X_train": PVin_norm[:idx_split],
        "X_test": PVin_norm[idx_split:],
        "y_train": PVout[:idx_split],
        "y_test": PVout[idx_split:],
        "in_extra": in_extra,
        "idx_split": idx_split,
        "Persis_simple_test": PVin[idx_split:, -1],
        "offset_base": offset_base,
        "mask_day_test": is_day_full[offset_base : offset_base + n_test],
    }


def baseline_rows(
    data: np.ndarray,
    LB: int,
    FH: int,
    split: dict,
    log_predictions,
) -> list[dict]:
    """Compute the 3 baselines (Persistence P, cyclic P°, BLEND) and log their predictions."""
    y_test = split["y_test"]
    Persis_simple_test = split["Persis_simple_test"]
    offset_base = split["offset_base"]
    mask_day_test = split["mask_day_test"]
    idx_split = split["idx_split"]
    n_test = len(y_test)

    rows: list[dict] = []

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

    # ---- BLEND. The train slice ends at idx_split + LB + FH - 1 (inclusive).
    data_tr_raw = data[: idx_split + LB + FH]
    lam_phase = fit_blend_lambda_per_phase(data_tr_raw, FH, T_period)
    y_pred_BL = predict_blend(y_pred_P, y_pred_Pc, lam_phase, offset_base, T_period)
    y_pred_BL = apply_night_mask(y_pred_BL, mask_day_test)
    print(
        f"    [BLEND] lam_phase: min={lam_phase.min():.3f} "
        f"max={lam_phase.max():.3f} mean={lam_phase.mean():.3f}"
    )
    rows.append(
        build_metric_row(
            "BLEND_opti", LB, FH, Persis_simple_test, y_test, y_pred_BL,
            mask_day_test, extra_fields={"N_params": 0}, steps_per_day=T_period,
        )
    )
    log_predictions("BLEND_opti", y_test, y_pred_BL)

    return rows


# ============================================================================
# RUN ONE CONFIG (LB, FH)
# ============================================================================
def _run_one(
    data: np.ndarray,
    is_day_full: np.ndarray,
    LB: int,
    FH: int,
    train_grid,
    use_time_features: bool,
    method_name: str,
    with_baselines: bool,
) -> tuple[list[dict], list[pd.DataFrame]]:
    hours_per_step = 24.0 / T_period
    print(
        f"\n=== LB={LB} ({LB/T_period:g}d) | "
        f"FH={FH} ({FH*hours_per_step:.1f}h) ==="
    )

    split = prepare_split(data, is_day_full, LB, FH, use_time_features)
    X_train, X_test = split["X_train"], split["X_test"]
    y_train, y_test = split["y_train"], split["y_test"]
    in_extra = split["in_extra"]
    Persis_simple_test = split["Persis_simple_test"]
    mask_day_test = split["mask_day_test"]

    rows: list[dict] = []
    pred_rows: list[pd.DataFrame] = []
    log_predictions = make_log_predictions(LB, FH, pred_rows)

    if with_baselines:
        rows.extend(baseline_rows(data, LB, FH, split, log_predictions))

    # ---- ELM
    rng_fh = np.random.default_rng(SEED + FH)
    print(f"  [{method_name}_full]" if with_baselines else f"  [{method_name}]")
    beta, IW, bias, sel_dict, predict_fn = train_grid(
        X_train, y_train, N_ELM_hidden_list, N_ELM_candidates_list, rng_fh
    )
    sel_h = sel_dict["n_hidden"]
    nParams_full = sel_h * (LB + in_extra) + sel_h + sel_h
    y_pred_f = predict_fn(X_test, beta, IW, bias)
    y_pred_f = apply_night_mask(y_pred_f, mask_day_test)
    rows.append(
        build_metric_row(
            method_name, LB, FH, Persis_simple_test, y_test, y_pred_f,
            mask_day_test,
            extra_fields={"N_params": nParams_full, **sel_dict},
            steps_per_day=T_period,
        )
    )
    log_predictions(method_name, y_test, y_pred_f)

    return rows, pred_rows


# ============================================================================
# GENERIC RUNNER
# ============================================================================
def run_elm(
    *,
    slug: str,
    script_name: str,
    train_grid,
    extra_cols: list[str],
    use_time_features: bool = True,
    grid_print: str = "",
    fh_list: list[int] | None = None,
    method_name: str = "ELM",
    with_baselines: bool = True,
) -> None:
    """Generic main(): loads the data, loops LB×FH, writes the CSVs, chrono."""
    t0 = time.time()

    out_file_all, out_file_day, pred_file = build_paths(slug)

    data = load_30min(CSV_FILE, CACHE_NPY, n_rows=Ndata, clip_nonneg=CLIP_NONNEG)
    print(f"Data: {len(data)} points ({len(data)/STEPS_PER_DAY/365.25:.2f} years)")
    if grid_print:
        print(grid_print)

    is_day_full = day_mask(len(data))
    print(
        f"Day mask  : {is_day_full.sum()}/{len(is_day_full)} steps "
        f"({is_day_full.mean()*100:.1f}% day)"
    )

    fhs = fh_list if fh_list is not None else FH_list

    all_rows: list[dict] = []
    all_pred_rows: list[pd.DataFrame] = []
    for LB in LB_list:
        for FH in fhs:
            rows, pred_rows = _run_one(
                data, is_day_full, LB, FH, train_grid, use_time_features,
                method_name, with_baselines,
            )
            all_rows.extend(rows)
            all_pred_rows.extend(pred_rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_all, df_day = split_and_save(
        all_rows,
        extra_cols=extra_cols,
        out_path_all=str(out_file_all),
        out_path_day=str(out_file_day),
    )
    print("\n===== RESULTS (all samples) =====")
    print(df_all.to_string())
    print("\n===== RESULTS (day only) =====")
    print(df_day.to_string())
    print(f"\nResults _all : {out_file_all}")
    print(f"Results _day : {out_file_day}")

    predictions = pd.concat(all_pred_rows, ignore_index=True)
    predictions.to_csv(pred_file, index=False)
    print(f"Predictions saved: {pred_file}")

    dt = time.time() - t0
    print(f"\n[chrono] {script_name} : {dt:.2f} s  ({dt/60:.2f} min)")
