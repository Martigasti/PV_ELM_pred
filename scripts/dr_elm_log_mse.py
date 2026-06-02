"""
ELM with Log-MSE cost on the PV_AC Palaiseau data.

Uses a 2-pass method: Pass 1 is a Ridge initialization on y, Pass 2 is a single
linearized solve fitting the residuals in log space around yhat^(0). 

The loss is:
    J(beta) = sum_i (log(y_i + c) - log(h_i beta + c))^2 + lam * ||beta||^2

with shift c > 0 (defined even at night, y = 0) grid-searched in C_GRID.

    Pass 1: beta_0 = (H^T H + lam I)^-1 H^T Y                  (Ridge on y)
    Pass 2: yhat_i^(0) = h_i beta_0                            (in y space)
            ytilde_i = log(y_i+c) - log(yhat_i^(0)+c) + (h_i beta_0) / (yhat_i^(0)+c)
            W_log = diag(1 / (yhat_i^(0) + c)^2)
            beta = (H^T W_log H + lam I)^-1 H^T W_log ytilde   (a single solve)
    
Grid-searched hyperparameters: lam, c.

Models reported per (LB, FH):
    - Persistence_P : y_pred = PAC(t)
    - Persistence_Pcyclic : y_pred = PAC(t + FH - 24h)
    - BLEND_opti : convex least-squares combination of the two persistences
    - ELM : ELM-Log-MSE on [LB lags + 4 time features]
"""
from math import floor, sqrt
import numpy as np

from elm_common import VAL_RATIO, elm_sigmoid, ridge_solve, run_elm


LAMBDA_GRID: list[float] = [10.0, 25.0]
C_GRID: list[float] = [0.1, 1.0, 10.0, 100.0]


# ============================================================================
# ELM-Log-MSE 2-pass 
# ============================================================================
def log_mse_solve(
    H: np.ndarray,
    y: np.ndarray,
    lam: float,
    c: float,
) -> np.ndarray:
    """
    Strict 2-pass: Pass 1 Ridge on y, Pass 2 a single linearized solve.
    """
    n_hidden = H.shape[1]
    log_y = np.log(y + c)
    # Pass 1: Ridge on y (original space)
    beta0 = ridge_solve(H, y, lam)
    # Pass 2: linearization around yhat_0 (assumed >= 0 in original space).
    # yhat0 is clipped to 0 before adding c, otherwise the samples where Ridge
    # predicts negatively blow up ytilde (tiny yhat0_c, very negative yhat0).
    yhat0 = np.maximum(H @ beta0, 0.0)
    yhat0_c = yhat0 + c
    w = 1.0 / (yhat0_c * yhat0_c)
    ytilde = log_y - np.log(yhat0_c) + yhat0 / yhat0_c
    WH = H * w[:, None]
    A = H.T @ WH + lam * np.eye(n_hidden)
    b = H.T @ (w * ytilde)
    return np.linalg.solve(A, b)


def log_mse_predict_raw(H: np.ndarray, beta: np.ndarray, c: float) -> np.ndarray:
    """Direct prediction: H beta lives in original space (PDF §A.2)."""
    return np.clip(H @ beta, a_min=0.0, a_max=None)


def train_elm_log_mse(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden: int,
    n_candidates: int,
    rng: np.random.Generator,
    lam_grid: list[float] | None = None,
    c_grid: list[float] | None = None,
    val_ratio: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    in_size = X.shape[1]
    n_train = X.shape[0]
    n_fit = max(1, floor((1.0 - val_ratio) * n_train))
    X_fit, X_val = X[:n_fit], X[n_fit:]
    y_fit, y_val = y[:n_fit], y[n_fit:]

    grid_l = lam_grid if lam_grid else LAMBDA_GRID
    grid_c = c_grid if c_grid else C_GRID

    best_val, best_lam, best_c_shift, best_IW, best_bias = np.inf, None, None, None, None

    for _ in range(n_candidates):
        IW = rng.uniform(-1.0, 1.0, size=(n_hidden, in_size))
        bias = rng.uniform(0.0, 1.0, size=n_hidden)
        H_fit = elm_sigmoid(X_fit @ IW.T + bias)
        H_val = elm_sigmoid(X_val @ IW.T + bias)

        for l in grid_l:
            for c_val in grid_c:
                beta = log_mse_solve(H_fit, y_fit, l, c_val)
                y_val_pred = log_mse_predict_raw(H_val, beta, c_val)
                val_rmse = sqrt(np.mean((y_val_pred - y_val) ** 2))
                if val_rmse < best_val:
                    best_val = val_rmse
                    best_lam, best_c_shift = l, c_val
                    best_IW, best_bias = IW, bias

    H_full = elm_sigmoid(X @ best_IW.T + best_bias)
    best_beta = log_mse_solve(H_full, y, best_lam, best_c_shift)
    return best_beta, best_IW, best_bias, best_lam, best_c_shift, best_val


def train_elm_log_mse_grid(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden_list: list[int],
    n_candidates_list: list[int],
    rng: np.random.Generator,
):
    best_val = np.inf
    best_beta = best_IW = best_bias = None
    best_h = best_n_cand = best_lam = best_c_shift = None
    for n_hidden in n_hidden_list:
        for n_candidates in n_candidates_list:
            beta, IW, bias, lam_sel, c_sel, val_rmse = train_elm_log_mse(
                X, y, n_hidden, n_candidates, rng,
                lam_grid=LAMBDA_GRID, c_grid=C_GRID, val_ratio=VAL_RATIO,
            )
            print(
                f"    n_hidden={n_hidden:4d}  n_cand={n_candidates:4d}  "
                f"lam={lam_sel:g}  c={c_sel:g}  val_RMSE={val_rmse:.4g}"
            )
            if val_rmse < best_val:
                best_val, best_beta, best_IW, best_bias = val_rmse, beta, IW, bias
                best_h, best_n_cand, best_lam, best_c_shift = n_hidden, n_candidates, lam_sel, c_sel
    print(
        f"    -> selected: n_hidden={best_h}  n_cand={best_n_cand}  "
        f"lam={best_lam:g}  c={best_c_shift:g}  val_RMSE={best_val:.4g}"
    )
    sel_dict = {
        "n_hidden": best_h, "n_candidates": best_n_cand,
        "lambda_log_mse": best_lam, "c_log_mse": best_c_shift,
    }
    return best_beta, best_IW, best_bias, sel_dict, elm_predict


def elm_predict(
    X: np.ndarray, beta: np.ndarray, IW: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    H = elm_sigmoid(X @ IW.T + bias)
    return log_mse_predict_raw(H, beta, c=0.0)


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    run_elm(
        slug="log_mse",
        script_name="dr_elm_log_mse.py",
        train_grid=train_elm_log_mse_grid,
        extra_cols=["N_params", "n_hidden", "n_candidates", "lambda_log_mse", "c_log_mse"],
        grid_print=f"Log-MSE 2-pass: lam_grid={LAMBDA_GRID}, c_grid={C_GRID}",
    )


if __name__ == "__main__":
    main()
