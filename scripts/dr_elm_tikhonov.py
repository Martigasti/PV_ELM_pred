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
from math import floor, sqrt
import numpy as np

from elm_common import VAL_RATIO, elm_sigmoid, run_elm


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
    val_ratio: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Select (IW, bias, lambda) by validation RMSE, then refit beta on the full train set."""
    in_size = X.shape[1]
    n_train = X.shape[0]
    n_fit = max(1, floor((1.0 - val_ratio) * n_train))
    X_fit, X_val = X[:n_fit], X[n_fit:]
    y_fit, y_val = y[:n_fit], y[n_fit:]

    best_val, best_lam, best_IW, best_bias = np.inf, None, None, None

    for _ in range(n_candidates):
        IW = rng.uniform(-1.0, 1.0, size=(n_hidden, in_size))
        bias = rng.uniform(0.0, 1.0, size=n_hidden)
        H_fit = elm_sigmoid(X_fit @ IW.T + bias)
        H_val = elm_sigmoid(X_val @ IW.T + bias)
        s_fit = np.mean(H_fit ** 2, axis=0)

        for l in lam_grid:
            beta = tikhonov_solve(H_fit, y_fit, l, s_fit)
            y_val_pred = np.clip(H_val @ beta, a_min=0.0, a_max=None)
            val_rmse = sqrt(np.mean((y_val_pred - y_val) ** 2))
            if val_rmse < best_val:
                best_val, best_lam, best_IW, best_bias = val_rmse, l, IW, bias

    H_full = elm_sigmoid(X @ best_IW.T + best_bias)
    s_full = np.mean(H_full ** 2, axis=0)
    best_beta = tikhonov_solve(H_full, y, best_lam, s_full)
    return best_beta, best_IW, best_bias, best_lam, best_val


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
                lam_grid=LAMBDA_GRID, val_ratio=VAL_RATIO,
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
