"""
Robust Risk-regularized ELM forecasting on the PV_AC Palaiseau data.

Closed-form Ridge solve where lam = eps^2, interpreting eps as the bounded
uncertainty magnitude on H (worst-case min-max formulation, El Ghaoui 1997):

    Ridge        : beta = (H^T H + lam I)^-1 H^T y     (lam free)
    Robust Risk  : beta = (H^T H + eps^2 I)^-1 H^T y   (lam = eps^2)

The target is center-scaled on the train (z = (y - mu_y) / sd_y) before the
solve, prediction mapped back (y_pred = z_pred * sd_y + mu_y); keeps the eps^2
penalty from shrinking a nonzero target mean (meteo). mu_y/sd_y frozen on the
train (fit fold in CV).

Eps is selected per candidate by minimal validation RMSE over EPS_GRID.

Models reported per (LB, FH):
    - Persistence_P : y_pred = PAC(t)
    - Persistence_Pcyclic : y_pred = PAC(t + FH - 24h)
    - BLEND_opti : convex least-squares combination of the two persistences
    - ELM : ELM Robust Risk on [LB lags + 4 time features]
"""
from math import sqrt

import numpy as np

# The common part (config, baselines, runner) lives in elm_common.
from elm_common import CLIP_NONNEG, CV_FOLDS, elm_sigmoid, run_elm, select_by_temporal_cv


EPS_GRID: list[float] = [1e-2, 0.1, 0, sqrt(10), 5.0, 10]

# Clip predictions to >=0 only for physical-power targets (skipped for non-solar
# meteo targets, e.g. temperature, which can be negative). Mirrors dr_elm_ridge.
_clip = (lambda a: np.clip(a, a_min=0.0, a_max=None)) if CLIP_NONNEG else (lambda a: a)


# ============================================================================
# ROBUST RISK-REGULARIZED ELM (= Ridge with lam = eps^2)
# ============================================================================
def robust_risk_solve(H: np.ndarray, y: np.ndarray, eps: float) -> np.ndarray:
    """Solve beta = (H^T H + eps^2 I)^-1 H^T y."""
    n_hidden = H.shape[1]
    A = H.T @ H + (eps ** 2) * np.eye(n_hidden)
    b = H.T @ y
    return np.linalg.solve(A, b)


def train_elm_robust_risk(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden: int,
    n_candidates: int,
    rng: np.random.Generator,
    eps_grid: list[float],
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Select (IW, bias, eps) by temporal CV, then refit beta on the full train set."""
    def fit_score(X_fit, y_fit, X_val, y_val, IW, bias, combo):
        (eps,) = combo
        mu, sd = y_fit.mean(), y_fit.std(ddof=1); sd = sd if sd > 0 else 1.0
        beta = robust_risk_solve(elm_sigmoid(X_fit @ IW.T + bias), (y_fit - mu) / sd, eps)
        z_val = elm_sigmoid(X_val @ IW.T + bias) @ beta
        y_val_pred = _clip(z_val * sd + mu)
        return sqrt(np.mean((y_val_pred - y_val) ** 2))

    def refit(X_full, y_full, IW, bias, combo):
        (eps,) = combo
        mu, sd = y_full.mean(), y_full.std(ddof=1); sd = sd if sd > 0 else 1.0
        beta = robust_risk_solve(elm_sigmoid(X_full @ IW.T + bias), (y_full - mu) / sd, eps)
        return beta, None

    combos = [(e,) for e in eps_grid]
    beta, IW, bias, combo, _, best_val = select_by_temporal_cv(
        X, y, n_hidden, n_candidates, rng, combos, fit_score, refit, k=k,
    )
    return beta, IW, bias, combo[0], best_val


def train_elm_robust_risk_grid(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden_list: list[int],
    n_candidates_list: list[int],
    rng: np.random.Generator,
):
    best_val = np.inf
    best_beta = best_IW = best_bias = None
    best_h = best_c = best_eps = None
    for n_hidden in n_hidden_list:
        for n_candidates in n_candidates_list:
            beta, IW, bias, eps_sel, val_rmse = train_elm_robust_risk(
                X, y, n_hidden, n_candidates, rng,
                eps_grid=EPS_GRID, k=CV_FOLDS,
            )
            print(
                f"    n_hidden={n_hidden:4d}  n_cand={n_candidates:4d}  "
                f"eps={eps_sel:g}  val_RMSE={val_rmse:.4g}"
            )
            if val_rmse < best_val:
                best_val, best_beta, best_IW, best_bias = val_rmse, beta, IW, bias
                best_h, best_c, best_eps = n_hidden, n_candidates, eps_sel
    print(
        f"    -> selected: n_hidden={best_h}  n_cand={best_c}  eps={best_eps:g}  "
        f"val_RMSE={best_val:.4g}"
    )
    sel_dict = {"n_hidden": best_h, "n_candidates": best_c, "eps_robust": best_eps}
    mu_y, sd_y = y.mean(), y.std(ddof=1); sd_y = sd_y if sd_y > 0 else 1.0

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
        slug="robust_risk",
        script_name="dr_elm_robust_risk.py",
        train_grid=train_elm_robust_risk_grid,
        extra_cols=["N_params", "n_hidden", "n_candidates", "eps_robust"],
        grid_print=f"Robust Risk: eps grid={EPS_GRID}",
    )


if __name__ == "__main__":
    main()
