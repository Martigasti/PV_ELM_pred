"""
Anisotropic Tikhonov-regularized ELM forecasting on the PV_AC Palaiseau data.

Closed-form solve with L = diag(sqrt(lam * s_j)), s_j = mean(H[:, j]^2), so each
neuron is penalized proportionally to its mean energy (active neurons regularized more):

    Ridge     : beta = (H^T H + lam I)^-1 H^T y          (isotropic)
    Tikhonov  : beta = (H^T H + lam * diag(s_j))^-1 H^T y (anisotropic)

Lambda is selected per candidate by minimal validation RMSE over LAMBDA_GRID.

Models reported per (LB, FH):
    - Persistence_P : y_pred = PAC(t)
    - Persistence_Pcyclic : y_pred = PAC(t + FH - 24h)
    - BLEND_opti : convex least-squares combination of the two persistences
    - ELM : ELM Tikhonov anisotrope on [LB lags + 4 time features]
"""
from math import sqrt
import numpy as np

from elm_common import CV_FOLDS, elm_sigmoid, run_elm, select_by_temporal_cv


LAMBDA_GRID: list[float] = [10.0, 25.0]


# ============================================================================
# ANISOTROPIC TIKHONOV ELM
# ============================================================================
def tikhonov_solve(H: np.ndarray, y: np.ndarray, lam: float, s: np.ndarray) -> np.ndarray:
    """Solve beta = (H^T H + lam * diag(s))^-1 H^T y.

    s : per-neuron energy vector (s_j = mean(H[:, j]^2)), computed on the
        same H as the one passed here (fit or full train).
    """
    A = H.T @ H + lam * np.diag(s)
    b = H.T @ y
    return np.linalg.solve(A, b)


def train_elm_tikhonov(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden: int,
    n_candidates: int,
    rng: np.random.Generator,
    lam_grid: list[float],
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Select (IW, bias, lambda) by temporal CV, then refit beta on the full train set."""
    def fit_score(X_fit, y_fit, X_val, y_val, IW, bias, combo):
        (lam,) = combo
        H_fit = elm_sigmoid(X_fit @ IW.T + bias)
        s_fit = np.mean(H_fit ** 2, axis=0)
        beta = tikhonov_solve(H_fit, y_fit, lam, s_fit)
        y_val_pred = np.clip(elm_sigmoid(X_val @ IW.T + bias) @ beta, a_min=0.0, a_max=None)
        return sqrt(np.mean((y_val_pred - y_val) ** 2))

    def refit(X_full, y_full, IW, bias, combo):
        (lam,) = combo
        H_full = elm_sigmoid(X_full @ IW.T + bias)
        s_full = np.mean(H_full ** 2, axis=0)
        return tikhonov_solve(H_full, y_full, lam, s_full), None

    combos = [(lam,) for lam in lam_grid]
    beta, IW, bias, combo, _, best_val = select_by_temporal_cv(
        X, y, n_hidden, n_candidates, rng, combos, fit_score, refit, k=k,
    )
    return beta, IW, bias, combo[0], best_val


def train_elm_tikhonov_grid(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden_list: list[int],
    n_candidates_list: list[int],
    rng: np.random.Generator,
):
    best_val = np.inf
    best_beta = best_IW = best_bias = None
    best_h = best_c = best_lam = None
    for n_hidden in n_hidden_list:
        for n_candidates in n_candidates_list:
            beta, IW, bias, lam_sel, val_rmse = train_elm_tikhonov(
                X, y, n_hidden, n_candidates, rng,
                lam_grid=LAMBDA_GRID, k=CV_FOLDS,
            )
            print(
                f"    n_hidden={n_hidden:4d}  n_cand={n_candidates:4d}  "
                f"lam={lam_sel:g}  val_RMSE={val_rmse:.4g}"
            )
            if val_rmse < best_val:
                best_val, best_beta, best_IW, best_bias = val_rmse, beta, IW, bias
                best_h, best_c, best_lam = n_hidden, n_candidates, lam_sel
    print(
        f"    -> selected: n_hidden={best_h}  n_cand={best_c}  lam={best_lam:g}  "
        f"val_RMSE={best_val:.4g}"
    )
    sel_dict = {"n_hidden": best_h, "n_candidates": best_c, "lambda_tikhonov": best_lam}
    return best_beta, best_IW, best_bias, sel_dict, elm_predict


def elm_predict(
    X: np.ndarray, beta: np.ndarray, IW: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    # PAC is a physical power: clip negative values to 0.
    return np.clip(elm_sigmoid(X @ IW.T + bias) @ beta, a_min=0.0, a_max=None)


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    run_elm(
        slug="tikhonov",
        script_name="dr_elm_tikhonov.py",
        train_grid=train_elm_tikhonov_grid,
        extra_cols=["N_params", "n_hidden", "n_candidates", "lambda_tikhonov"],
        grid_print=f"Tikhonov anisotrope: lambda grid={LAMBDA_GRID}",
    )


if __name__ == "__main__":
    main()
