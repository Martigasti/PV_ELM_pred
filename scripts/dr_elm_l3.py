"""
ELM with L3 (cubic norm) cost on the PV_AC Palaiseau data.

Uses a 2-pass method: Pass 1 is a Ridge initialization, Pass 2 is a single
reweighted solve with W = diag(sqrt(r^(0)^2 + delta^2)) upweighting large residuals.

    Pass 1: beta_0 = (H^T H + lam I)^-1 H^T y      (Ridge init)
    Pass 2: r_i^(0) = y_i - h_i beta_0
             W = diag(sqrt(r_i^(0)^2 + delta^2))    (upweights large residuals)
             beta = (H^T W H)^-1 H^T W y            (a single solve)

Unlike MAE/Huber (which reduce the influence of large residuals),
L3 increases their weight: the cubic norm penalizes large errors
more strongly than L2 (motivation: abrupt transitions, cloud passages).

Grid-searched hyperparameters: (lam, delta).

Models reported per (LB, FH):
    - Persistence_P : y_pred = PAC(t)
    - Persistence_Pcyclic : y_pred = PAC(t + FH - 24h)
    - BLEND_opti : convex least-squares combination of the two persistences
    - ELM : ELM-L3 on [LB lags + 4 time features]
"""
from math import floor, sqrt
import numpy as np

from elm_common import VAL_RATIO, elm_sigmoid, ridge_solve, run_elm


LAMBDA_GRID: list[float] = [10.0, 25.0]
DELTA_GRID: list[float] = [10.0, 100.0, 500.0]


# ============================================================================
# ELM-L3 2-pass
# ============================================================================
def l3_solve(H: np.ndarray, y: np.ndarray, lam: float, delta: float) -> np.ndarray:
    """Strict 2-pass: Pass 1 Ridge init, Pass 2 a single weighted solve.

    W_L3 = diag(sqrt(r^2 + delta^2)) emphasizes the large residuals.
    """
    beta0 = ridge_solve(H, y, lam)
    r = y - H @ beta0
    w = np.sqrt(r * r + delta * delta)
    WH = H * w[:, None]
    A = H.T @ WH
    b = H.T @ (w * y)
    return np.linalg.solve(A, b)


def train_elm_l3(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden: int,
    n_candidates: int,
    rng: np.random.Generator,
    lam_grid: list[float] | None = None,
    delta_grid: list[float] | None = None,
    val_ratio: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    in_size = X.shape[1]
    n_train = X.shape[0]
    n_fit = max(1, floor((1.0 - val_ratio) * n_train))
    X_fit, X_val = X[:n_fit], X[n_fit:]
    y_fit, y_val = y[:n_fit], y[n_fit:]

    lams = lam_grid if lam_grid else LAMBDA_GRID
    deltas = delta_grid if delta_grid else DELTA_GRID

    best_val = np.inf
    best_IW = best_bias = None
    best_lam = best_delta = None

    for _ in range(n_candidates):
        IW = rng.uniform(-1.0, 1.0, size=(n_hidden, in_size))
        bias = rng.uniform(0.0, 1.0, size=n_hidden)
        H_fit = elm_sigmoid(X_fit @ IW.T + bias)
        H_val = elm_sigmoid(X_val @ IW.T + bias)

        for lam in lams:
            for dlt in deltas:
                beta = l3_solve(H_fit, y_fit, lam, dlt)
                y_val_pred = np.clip(H_val @ beta, a_min=0.0, a_max=None)
                val_rmse = sqrt(np.mean((y_val_pred - y_val) ** 2))
                if val_rmse < best_val:
                    best_val = val_rmse
                    best_IW, best_bias = IW, bias
                    best_lam, best_delta = lam, dlt

    H_full = elm_sigmoid(X @ best_IW.T + best_bias)
    best_beta = l3_solve(H_full, y, best_lam, best_delta)
    return best_beta, best_IW, best_bias, best_lam, best_delta, best_val


def train_elm_l3_grid(
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
            beta, IW, bias, lam_sel, delta_sel, val_rmse = train_elm_l3(
                X, y, n_hidden, n_candidates, rng,
                lam_grid=LAMBDA_GRID, delta_grid=DELTA_GRID, val_ratio=VAL_RATIO,
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
    sel_dict = {
        "n_hidden": best_h, "n_candidates": best_c,
        "lambda_l3": best_lam, "delta_l3": best_delta,
    }
    return best_beta, best_IW, best_bias, sel_dict, elm_predict


def elm_predict(
    X: np.ndarray, beta: np.ndarray, IW: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    return np.clip(elm_sigmoid(X @ IW.T + bias) @ beta, a_min=0.0, a_max=None)


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    run_elm(
        slug="l3",
        script_name="dr_elm_l3.py",
        train_grid=train_elm_l3_grid,
        extra_cols=["N_params", "n_hidden", "n_candidates", "lambda_l3", "delta_l3"],
        grid_print=f"L3 2-pass: lam_grid={LAMBDA_GRID}, delta_grid={DELTA_GRID}",
    )


if __name__ == "__main__":
    main()
