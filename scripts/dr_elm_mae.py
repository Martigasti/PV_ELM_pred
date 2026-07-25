"""
ELM with MAE (L1) cost on the PV_AC Palaiseau data.

Uses a 2-pass method: Pass 1 is a Ridge initialization, Pass 2 is a single
reweighted solve with W = diag(1 / sqrt(r^(0)^2 + delta^2)) approximating L1.

    Pass 1: beta_0 = (H^T H + lam I)^-1 H^T y      (Ridge init)
    Pass 2: r_i^(0) = y_i - h_i beta_0
             W = diag(1 / sqrt(r_i^(0)^2 + delta^2))
             beta = (H^T W H)^-1 H^T W y            (a single solve)

Grid-searched hyperparameters: (lam, delta).

The target is center-scaled on the train (z = (y - mu_y) / sd_y) before both
passes, prediction mapped back (y_pred = z_pred * sd_y + mu_y); keeps the ridge
penalty of Pass 1 from shrinking a nonzero target mean (meteo). delta is a
fraction of sd_y (grid DELTA_GRID_Z, since r lives in z-space); the reported
delta_mae is re-multiplied by sd_y (physical units). mu_y/sd_y frozen on the
train (fit fold in CV).

Models reported per (LB, FH):
    - Persistence_P : y_pred = PAC(t)
    - Persistence_Pcyclic : y_pred = PAC(t + FH - 24h)
    - BLEND_opti : convex least-squares combination of the two persistences
    - ELM : ELM-MAE on [LB lags + 4 time features]
"""
from math import sqrt
import numpy as np

from elm_common import CLIP_NONNEG, CV_FOLDS, elm_sigmoid, ridge_solve, run_elm, select_by_temporal_cv


LAMBDA_GRID: list[float] = [10.0, 25.0]
# delta grid in center-scaled (z) space: fractions of sd_y. The target is
# solved on z (r ~ O(1)), so the old absolute-watt grid {10,100,500} would make
# the weights flat and degenerate MAE into Ridge; {0.05,0.1,0.5}*sd_y keeps the
# smoothing light (delta < sd_y), preserving the L1 robustness.
DELTA_GRID_Z: list[float] = [0.05, 0.1, 0.5]

# Clip predictions to >=0 only for physical-power targets (skipped for non-solar
# meteo targets, e.g. temperature, which can be negative). Mirrors dr_elm_ridge.
_clip = (lambda a: np.clip(a, a_min=0.0, a_max=None)) if CLIP_NONNEG else (lambda a: a)


# ============================================================================
# ELM-MAE 2-pass
# ============================================================================
def mae_solve(H: np.ndarray, y: np.ndarray, lam: float, delta: float) -> np.ndarray:
    """Strict 2-pass: Pass 1 Ridge, Pass 2 a single smoothed weighted solve."""
    beta0 = ridge_solve(H, y, lam)
    r = y - H @ beta0
    w = 1.0 / np.sqrt(r * r + delta * delta)
    WH = H * w[:, None]
    A = H.T @ WH
    b = H.T @ (w * y)
    return np.linalg.solve(A, b)


def train_elm_mae(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden: int,
    n_candidates: int,
    rng: np.random.Generator,
    lam_grid: list[float] | None = None,
    delta_grid: list[float] | None = None,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    lams = lam_grid if lam_grid else LAMBDA_GRID
    deltas = delta_grid if delta_grid else DELTA_GRID_Z

    def fit_score(X_fit, y_fit, X_val, y_val, IW, bias, combo):
        lam, dlt = combo
        mu, sd = y_fit.mean(), y_fit.std(ddof=1); sd = sd if sd > 0 else 1.0
        beta = mae_solve(elm_sigmoid(X_fit @ IW.T + bias), (y_fit - mu) / sd, lam, dlt)
        z_val = elm_sigmoid(X_val @ IW.T + bias) @ beta
        y_val_pred = _clip(z_val * sd + mu)
        return sqrt(np.mean((y_val_pred - y_val) ** 2))

    def refit(X_full, y_full, IW, bias, combo):
        lam, dlt = combo
        mu, sd = y_full.mean(), y_full.std(ddof=1); sd = sd if sd > 0 else 1.0
        return mae_solve(elm_sigmoid(X_full @ IW.T + bias), (y_full - mu) / sd, lam, dlt), None

    combos = [(lam, dlt) for lam in lams for dlt in deltas]
    beta, IW, bias, combo, _, best_val = select_by_temporal_cv(
        X, y, n_hidden, n_candidates, rng, combos, fit_score, refit, k=k,
    )
    return beta, IW, bias, combo[0], combo[1], best_val


def train_elm_mae_grid(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden_list: list[int],
    n_candidates_list: list[int],
    rng: np.random.Generator,
):
    best_val = np.inf
    best_beta = best_IW = best_bias = None
    best_h = best_c = None
    best_lam = best_delta = None
    for n_hidden in n_hidden_list:
        for n_candidates in n_candidates_list:
            beta, IW, bias, lam_sel, delta_sel, val_rmse = train_elm_mae(
                X, y, n_hidden, n_candidates, rng,
                lam_grid=LAMBDA_GRID, delta_grid=DELTA_GRID_Z, k=CV_FOLDS,
            )
            print(
                f"    n_hidden={n_hidden:4d}  n_cand={n_candidates:4d}  "
                f"lam={lam_sel:g}  delta={delta_sel:g}  val_RMSE={val_rmse:.4g}"
            )
            if val_rmse < best_val:
                best_val, best_beta, best_IW, best_bias = val_rmse, beta, IW, bias
                best_h, best_c = n_hidden, n_candidates
                best_lam, best_delta = lam_sel, delta_sel
    print(
        f"    -> selected: n_hidden={best_h}  n_cand={best_c}  "
        f"lam={best_lam:g}  delta={best_delta:g}  val_RMSE={best_val:.4g}"
    )
    # delta is a fraction of sd_y in z-space; report it in physical units (x sd_y).
    mu_y, sd_y = y.mean(), y.std(ddof=1); sd_y = sd_y if sd_y > 0 else 1.0
    sel_dict = {
        "n_hidden": best_h, "n_candidates": best_c,
        "lambda_mae": best_lam, "delta_mae": best_delta * sd_y,
    }

    def predict_fn(Xte, beta, IW, bias):
        return elm_predict(Xte, beta, IW, bias, mu_y, sd_y)

    return best_beta, best_IW, best_bias, sel_dict, predict_fn


def elm_predict(
    X: np.ndarray, beta: np.ndarray, IW: np.ndarray, bias: np.ndarray,
    mu: float, sd: float,
) -> np.ndarray:
    # beta lives in center-scaled target space: de-standardize, then _clip.
    z = elm_sigmoid(X @ IW.T + bias) @ beta
    return _clip(z * sd + mu)


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    run_elm(
        slug="mae",
        script_name="dr_elm_mae.py",
        train_grid=train_elm_mae_grid,
        extra_cols=["N_params", "n_hidden", "n_candidates", "lambda_mae", "delta_mae"],
        grid_print=f"MAE 2-pass: lam_grid={LAMBDA_GRID}, delta_grid(z)={DELTA_GRID_Z}",
    )


if __name__ == "__main__":
    main()
