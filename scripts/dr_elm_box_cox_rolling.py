"""
ELM Box-Cox in ROLLING "full-window" mode.

Rolling = instead of one fixed 50/50 train, the model is retrained on a sliding
window of the most recent data as we sweep the test set, so it tracks seasonal
and regime changes. For each window size 'W' (1 month / 6 months / 1 year), the
ELM Box-Cox (Ridge solve on the Box-Cox-transformed target) is trained on ALL
supervised pairs of the window data[T-W:T] (contiguous targets, ~1488 pts for 1
month): any pair whose LB inputs (ending FH steps before the target) AND the
target fit inside the window.

To keep the cost tractable, the model is NOT rebuilt at every step but only every
REFIT_EVERY steps (env var, default STEPS_PER_DAY (48) = 1 day; 336 = 1 week).
In-between steps reuse the last model; the PREDICTION is still made at every step
(input x_pred specific to T). Cost ~ W x (n_test / REFIT_EVERY) x N_candidates x
CV_FOLDS.

At EACH refit, the hidden layer (IW, bias) is RE-SELECTED by drawing N_candidates
random layers, keeping the one with lowest validation RMSE (temporal
expanding-window CV on the current window), then refitting beta on the whole
window. The model therefore fully re-optimises on the recent data at each refit.

Hyperparameters (lam_r, lam_bc) are frozen ONCE per (window, FH) via temporal
expanding-window CV on a warm-up block (validation RMSE, Y scale). The
standardisation (mu, sd) is frozen per FH (stats of the 1st refit) -- neutral
w.r.t. the re-draw and keeps the windows comparable.
"""
import os
import time
from math import sqrt, inf

import numpy as np
import pandas as pd

from elm_common import elm_sigmoid, ridge_solve, temporal_cv_splits
from utils import (
    apply_night_mask,
    build_metric_row,
    load_30min,
    split_and_save,
    time_features_for_targets,
)
from dataset_config import (
    CACHE_NPY,
    CSV_FILE,
    NDATA_FULL,
    RESULTS_DIR,
    STEPS_PER_DAY,
    day_mask,
)

# Box-Cox transform
BC_SHIFT = 1.0


def box_cox(y, lam_bc):
    if lam_bc == 0.0:
        return np.log(y)
    return (np.power(y, lam_bc) - 1.0) / lam_bc


def box_cox_inverse(z, lam_bc):
    if lam_bc == 0.0:
        return np.exp(z)
    base = np.clip(1.0 + lam_bc * z, a_min=1e-12, a_max=None)
    return np.power(base, 1.0 / lam_bc)


def time_features_at(idx_abs, steps_per_day=48):
    """4 sin/cos features keyed on the ABSOLUTE target indices.

    Equivalent to time_features_for_targets but for arbitrary indices (the
    window does not start at 0).
    """
    idx_abs = np.asarray(idx_abs)
    h = (idx_abs % steps_per_day) * (24.0 / steps_per_day)
    j = (idx_abs // steps_per_day) % 365.25
    two_pi = 2.0 * np.pi
    return np.column_stack([
        np.sin(two_pi * h / 24.0), np.cos(two_pi * h / 24.0),
        np.sin(two_pi * j / 365.25), np.cos(two_pi * j / 365.25),
    ])


# ============================================================================
# CONFIG
# ============================================================================
SMOKE_TEST = os.environ.get("SMOKE_TEST", "1") == "1"

LB = 48
SEED = 42

# Spaced refit: we rebuild the model every REFIT_EVERY steps (env var). 
REFIT_EVERY = int(os.environ.get("REFIT_EVERY", STEPS_PER_DAY))
assert REFIT_EVERY >= 1, f"REFIT_EVERY must be >= 1 (got {REFIT_EVERY})"
LAMBDA_GRID = [10e-1, 1.0, 10.0, 25.0, 50.0, 100.0]
LAMBDA_BC_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]

# Windows in steps (48 steps/day).
WINDOWS = {"1m": 31 * 48, "6m": 183 * 48, "1y": 365 * 48}

if SMOKE_TEST:
    print("*** SMOKE TEST MODE ***")
    FH_list = [1, 12]
    N_hidden = 16
    N_candidates = 2
    CV_FOLDS = 3
    N_TEST_DAYS = 5        # SMOKE: truncate the test to 5 days (240 pts)
    WINDOWS = {"1m": 31 * 48, "3m": 90 * 48}   # reduced windows
    Ndata = 8000
else:
    print("*** FULL MODE ***")
    FH_list = [1, 2, 6, 12, 20]
    N_hidden = 500
    N_candidates = 100
    CV_FOLDS = 5
    N_TEST_DAYS = None     # FULL: the whole test
    Ndata = NDATA_FULL

# Window selection via env var (e.g. ROLLING_WINDOWS=1m or ROLLING_WINDOWS=1m,1y).
# The test stays aligned on the largest window of the mode (W_ALIGN), even if it
# is not run, so the NICE stays comparable across partial runs.
W_ALIGN = max(WINDOWS.values())
_sel = os.environ.get("ROLLING_WINDOWS")
if _sel:
    _keys = [k.strip() for k in _sel.split(",")]
    _unknown = [k for k in _keys if k not in WINDOWS]
    assert not _unknown, f"unknown windows: {_unknown} (available: {list(WINDOWS)})"
    WINDOWS = {k: WINDOWS[k] for k in _keys}
    print(f"Selected windows: {list(WINDOWS)}")


def window_targets(T, W, FH):
    """Absolute indices of ALL valid targets of the window, chronological.

    Contiguous targets T-W+LB+FH-1 .. T-1: any pair whose LB inputs (ending FH
    steps before the target) AND the target fit in data[T-W:T] and whose target
    is <= T-1 (causality). ~ W-LB-FH+1 points (1 month ~ 1488).
    """
    return np.arange(T - W + LB + FH - 1, T)


def supervised_pairs(data, tgt, FH):
    """(X_raw, y) pairs for arbitrary target indices."""
    X_raw = data[(tgt - FH - LB + 1)[:, None] + np.arange(LB)]
    return X_raw, data[tgt]


def build_window(data, T, W, FH, mu=None, sd=None):
    """Build the full-window train to predict target T.

    Returns (X_win, y_win, x_pred, mu, sd):
      - X_win, y_win: ALL the supervised pairs of window_targets(T, W, FH)
        (contiguous targets), standardised (mu/sd) + 4 cyclic features
        (in_size = LB + 4).
      - x_pred: standardised input (1, LB+4) to predict target T.
      - mu, sd: standardisation stats used.
    If mu/sd are provided (frozen per FH), they are used as-is; otherwise
    recomputed on this train. All targets of X_win are at an absolute index
    <= T-1 (causality). Cyclic features keyed on the ABSOLUTE indices.
    """
    tgt = window_targets(T, W, FH)
    X_raw, y_win = supervised_pairs(data, tgt, FH)

    if mu is None:
        mu = X_raw.mean(axis=0)
        sd = X_raw.std(axis=0, ddof=1)
        sd = np.where(sd == 0, 1.0, sd)
    X_win = np.concatenate([(X_raw - mu) / sd, time_features_at(tgt)], axis=1)

    x_pred = predict_input(data, T, FH, mu, sd)
    return X_win, y_win, x_pred, mu, sd


def predict_input(data, T, FH, mu, sd):
    """Standardised input (1, LB+4) to predict target T (LB steps ending FH
    steps before T + cyclic features). Lightweight: to call at every step even
    without a refit."""
    x_pred_raw = data[T - FH - LB + 1 : T - FH + 1]
    return np.concatenate(
        [((x_pred_raw - mu) / sd)[None, :], time_features_at([T])], axis=1
    )


def select_hparams_for_window(data, test_start, W, FH, rng):
    """Freeze (lam_r, lam_bc) by temporal expanding-window CV on the warm-up block.

    Block = data[test_start-W : test_start], full train (all the contiguous
    supervised pairs of the window). Standard expanding-window CV
    (temporal_cv_splits, CV_FOLDS folds, validation always after the fit,
    contiguous fit). Criterion = validation RMSE (original Y scale, after inverse
    Box-Cox + clip) summed over all folds. IW/bias drawn once (rng); mu/sd on the
    whole warm-up block.
    """
    in_size = LB + 4
    IW = rng.uniform(-1.0, 1.0, size=(N_hidden, in_size))
    bias = rng.uniform(0.0, 1.0, size=N_hidden)

    tgt = window_targets(test_start, W, FH)
    X_raw, y = supervised_pairs(data, tgt, FH)
    mu = X_raw.mean(axis=0)
    sd = X_raw.std(axis=0, ddof=1)
    sd = np.where(sd == 0, 1.0, sd)
    X = np.concatenate([(X_raw - mu) / sd, time_features_at(tgt)], axis=1)

    combos = [(lr, lbc) for lbc in LAMBDA_BC_GRID for lr in LAMBDA_GRID]
    sse = {c: 0.0 for c in combos}
    n_val = 0
    for fit, val in temporal_cv_splits(len(tgt), CV_FOLDS):
        Hf = elm_sigmoid(X[fit] @ IW.T + bias)
        Hv = elm_sigmoid(X[val] @ IW.T + bias)
        n_val += len(y[val])
        for lam_bc in LAMBDA_BC_GRID:
            z = box_cox(y[fit] + BC_SHIFT, lam_bc)
            for lam_r in LAMBDA_GRID:
                beta = ridge_solve(Hf, z, lam_r)
                yv = np.clip(
                    box_cox_inverse(Hv @ beta, lam_bc) - BC_SHIFT, 0.0, None
                )
                sse[(lam_r, lam_bc)] += float(np.sum((yv - y[val]) ** 2))
    lam_r, lam_bc = min(combos, key=lambda c: sse[c])
    rmse = sqrt(sse[(lam_r, lam_bc)] / n_val)
    print(f"    [hparams W={W} FH={FH}] lam_r={lam_r:g} lam_bc={lam_bc:g} val_RMSE={rmse:.4g}")
    return lam_r, lam_bc


def select_hidden_for_window(X_win, y_win, lam_r, lam_bc, rng):
    """Re-select the hidden layer at frozen lam: draw N_candidates layers
    (IW, bias), keep the one with the lowest validation RMSE (temporal
    expanding-window CV on the current window), then refit beta on the WHOLE
    window with the chosen layer.

    Returns (IW, bias, beta, val_rmse). This is what makes the rolling adaptive:
    at each refit the hidden layer is re-optimised on the recent data (no more
    layer frozen per FH).
    """
    in_size = X_win.shape[1]
    splits = temporal_cv_splits(len(y_win), CV_FOLDS)
    n_val = sum(len(y_win[val]) for _, val in splits)

    best_sse, best_IW, best_bias = inf, None, None
    for _ in range(N_candidates):
        IW = rng.uniform(-1.0, 1.0, size=(N_hidden, in_size))
        bias = rng.uniform(0.0, 1.0, size=N_hidden)
        sse = 0.0
        for fit, val in splits:
            Hf = elm_sigmoid(X_win[fit] @ IW.T + bias)
            beta = ridge_solve(Hf, box_cox(y_win[fit] + BC_SHIFT, lam_bc), lam_r)
            Hv = elm_sigmoid(X_win[val] @ IW.T + bias)
            yv = np.clip(box_cox_inverse(Hv @ beta, lam_bc) - BC_SHIFT, 0.0, None)
            sse += float(np.sum((yv - y_win[val]) ** 2))
        if sse < best_sse:
            best_sse, best_IW, best_bias = sse, IW, bias

    H = elm_sigmoid(X_win @ best_IW.T + best_bias)
    beta = ridge_solve(H, box_cox(y_win + BC_SHIFT, lam_bc), lam_r)
    return best_IW, best_bias, beta, sqrt(best_sse / n_val)


def roll_one_window(data, is_day_full, test_indices, W, win_label, rng_seed_base):
    """Full rolling over a window W: for each FH, predicts every point of
    test_indices and returns (rows, pred_rows).

    rows: build_metric_row lines (ELM + persistence P as a reference).
    pred_rows: DataFrames (Method, LB_days, FH_hours, t_index, y_true, y_pred).
    Full-window train. At each refit (every REFIT_EVERY steps), the hidden layer
    is RE-SELECTED by N_candidates draws (CV on the window) then beta refit on it.
    The in-between steps reuse the last model; the prediction is made at every
    step. lam_r/lam_bc and mu/sd frozen per FH.
    """
    rows, pred_rows = [], []
    for FH in FH_list:
        lam_r, lam_bc = select_hparams_for_window(
            data, test_indices[0], W, FH, np.random.default_rng(SEED + FH)
        )
        # Re-draw RNG specific to the FH (advances from one refit to the next ->
        # different hidden layers at each refit).
        rng = np.random.default_rng(SEED + FH)
        # Standardisation frozen on the train of the 1st refit (neutral vs re-draw).
        _, _, _, mu_fix, sd_fix = build_window(data, test_indices[0], W, FH)

        y_true = np.empty(len(test_indices))
        y_pred = np.empty(len(test_indices))
        y_persis = np.empty(len(test_indices))
        IW = bias = beta = None
        n_refits = 0
        for i, T in enumerate(test_indices):
            if i % REFIT_EVERY == 0:   # refit: re-optimise the hidden layer at T
                X_win, y_win, _, _, _ = build_window(
                    data, T, W, FH, mu=mu_fix, sd=sd_fix
                )
                IW, bias, beta, val_rmse = select_hidden_for_window(
                    X_win, y_win, lam_r, lam_bc, rng
                )
                n_refits += 1
            # Prediction at every step (last refit's model, x_pred specific to T)
            x_pred = predict_input(data, T, FH, mu_fix, sd_fix)
            z = elm_sigmoid(x_pred @ IW.T + bias) @ beta
            y_pred[i] = np.clip(box_cox_inverse(z, lam_bc) - BC_SHIFT, 0.0, None)[0]
            y_true[i] = data[T]
            y_persis[i] = data[T - FH]   # simple persistence = value at emission

        mask_day = is_day_full[test_indices]
        # Post-processing: force predictions to 0 at night (no-op for non-solar sets).
        y_pred = apply_night_mask(y_pred, mask_day)
        y_persis = apply_night_mask(y_persis, mask_day)
        method = f"ELM_rolling_{win_label}"
        extra = {"N_params": N_hidden * (LB + 4) + 2 * N_hidden,
                 "window": win_label, "lambda_ridge": lam_r, "lambda_bc": lam_bc,
                 "refit_every": REFIT_EVERY}
        rows.append(build_metric_row(
            method, LB, FH, y_persis, y_true, y_pred, mask_day, extra_fields=extra,
        ))
        # persistence reference (NICE of P vs P = 1, useful as a check)
        rows.append(build_metric_row(
            "Persistence_P", LB, FH, y_persis, y_true, y_persis, mask_day,
            extra_fields={**extra, "N_params": 0,
                          "lambda_ridge": np.nan, "lambda_bc": np.nan},
        ))
        for m, yp in ((method, y_pred), ("Persistence_P", y_persis)):
            pred_rows.append(pd.DataFrame({
                "Method": m, "LB_days": LB / 48, "FH_hours": FH * 0.5,
                "window": win_label, "t_index": test_indices,
                "y_true": y_true, "y_pred": yp,
            }))
        n_train = len(window_targets(test_indices[0], W, FH))
        print(f"  [{win_label} FH={FH}] done ({len(test_indices)} pts, "
              f"{n_refits} refits (every {REFIT_EVERY}, N_cand={N_candidates}), "
              f"n_train={n_train})")
    return rows, pred_rows


def main():
    data = load_30min(CSV_FILE, CACHE_NPY, n_rows=Ndata)
    print(f"Data: {len(data)} points ({len(data)/48/365.25:.2f} years)")
    is_day_full = day_mask(len(data))

    test_start = W_ALIGN
    test_indices = np.arange(test_start, len(data))
    if N_TEST_DAYS is not None:
        test_indices = test_indices[: N_TEST_DAYS * 48]
    assert len(test_indices) > 0, "1y window too large for the loaded series"
    print(f"Common test: {len(test_indices)} points "
          f"[{test_start}..{test_indices[-1] + 1}) (W_align={W_ALIGN}, "
          f"full-window train, refit every {REFIT_EVERY} steps)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    extra_cols = ["N_params", "window", "lambda_ridge", "lambda_bc", "refit_every"]
    # Version suffix: refit cadence, so that two runs differing only by
    # REFIT_EVERY (e.g. 336 weekly vs 48 daily) do not overwrite their CSVs.
    suffix = f"_r{REFIT_EVERY}"
    all_pred = []
    for win_label, W in WINDOWS.items():
        print(f"\n===== WINDOW {win_label} (W={W} steps, {W/48:g} d) =====")
        rows, pred_rows = roll_one_window(
            data, is_day_full, test_indices, W, win_label, SEED
        )
        # identical t_index across windows (NICE comparability)
        assert np.array_equal(pred_rows[0]["t_index"].to_numpy(), test_indices)
        out_all = RESULTS_DIR / f"Results_DR_ELM_box_cox_rolling_{win_label}{suffix}_all.csv"
        out_day = RESULTS_DIR / f"Results_DR_ELM_box_cox_rolling_{win_label}{suffix}_day.csv"
        df_all, df_day = split_and_save(rows, extra_cols, str(out_all), str(out_day))
        print(df_all.to_string())
        print(f"  -> {out_all}")
        pred = pd.concat(pred_rows, ignore_index=True)
        pred_file = RESULTS_DIR / f"Predictions_DR_ELM_box_cox_rolling_{win_label}{suffix}.csv"
        pred.to_csv(pred_file, index=False)
        print(f"  -> {pred_file}")
        all_pred.append(pred_file)
    print(f"\nPrediction files: {[str(p) for p in all_pred]}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    dt = time.time() - t0
    print(f"\n[chrono] dr_elm_box_cox_rolling.py : {dt:.2f} s  ({dt/60:.2f} min)")
