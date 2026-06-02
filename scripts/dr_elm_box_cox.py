"""
Ridge-regularized ELM on Box-Cox transformed target.

Box-Cox transforms the target to make it more "Gaussian" / variance more
homogeneous, runs a linear regression on the transformed target, then inverts
the transformation to predict.

Transformation:
    Y_bc = (Y^lam_bc - 1) / lam_bc   if lam_bc != 0
    Y_bc = log(Y)                    if lam_bc == 0

Analytical solution (Ridge on the transformed target):
    beta = (H^T H + lam_r I)^-1 H^T Y_bc

Two hyperparameters:
    - lam_r  : regularization strength (Ridge)
    - lam_bc : Box-Cox exponent (lam_bc = 1 ≈ Ridge on Y, lam_bc = 0.5 ≈ Ridge
               on sqrt(Y), lam_bc = 0 ≈ Ridge on log(Y))

Inversion:
    Y = (1 + lam_bc * Z)^(1/lam_bc)   if lam_bc != 0
    Y = exp(Z)                        if lam_bc == 0

Lam_r and lam_bc are selected jointly by minimal validation RMSE
over LAMBDA_GRID x LAMBDA_BC_GRID.

Models reported per (LB, FH):
    - Persistence_P : y_pred = PAC(t)
    - Persistence_Pcyclic : y_pred = PAC(t + FH - 24h)
    - BLEND_opti : convex least-squares combination of the two persistences
    - ELM : ELM Ridge on Box-Cox transformed Y on [LB lags + 4 time features]
"""
from math import floor, sqrt
import numpy as np

from elm_common import VAL_RATIO, elm_sigmoid, ridge_solve, run_elm


LAMBDA_GRID: list[float] = [10.0, 25.0]
LAMBDA_BC_GRID: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]
BC_SHIFT: float = 1.0  # shift c added to Y to guarantee Y + c > 0


# ============================================================================
# BOX-COX TRANSFORMATION
# ============================================================================
def box_cox(y: np.ndarray, lam_bc: float) -> np.ndarray:
    """Apply Box-Cox to y (assumes y > 0)."""
    if lam_bc == 0.0:
        return np.log(y)
    return (np.power(y, lam_bc) - 1.0) / lam_bc


def box_cox_inverse(z: np.ndarray, lam_bc: float) -> np.ndarray:
    """Box-Cox inverse. Clip to a small epsilon to avoid negative bases."""
    if lam_bc == 0.0:
        return np.exp(z)
    base = 1.0 + lam_bc * z
    base = np.clip(base, a_min=1e-12, a_max=None)
    return np.power(base, 1.0 / lam_bc)


# ============================================================================
# RIDGE-REGULARIZED ELM ON BOX-COX TARGET
# ============================================================================
def train_elm_box_cox(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden: int,
    n_candidates: int,
    rng: np.random.Generator,
    lam_grid: list[float] | None = None,
    lam_bc_grid: list[float] | None = None,
    val_ratio: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Select (IW, bias, lam_r, lam_bc) by validation RMSE (on the original
    Y scale, after Box-Cox inversion), then refit beta on the full train set."""
    in_size = X.shape[1]
    n_train = X.shape[0]
    n_fit = max(1, floor((1.0 - val_ratio) * n_train))
    X_fit, X_val = X[:n_fit], X[n_fit:]
    y_fit, y_val = y[:n_fit], y[n_fit:]

    grid_r = lam_grid if lam_grid else LAMBDA_GRID
    grid_bc = lam_bc_grid if lam_bc_grid else LAMBDA_BC_GRID

    # Shift for positivity — applied to Y, not to X
    y_fit_pos = y_fit + BC_SHIFT
    y_val_pos = y_val + BC_SHIFT

    best_val, best_lam_r, best_lam_bc = np.inf, None, None
    best_IW, best_bias = None, None

    for _ in range(n_candidates):
        IW = rng.uniform(-1.0, 1.0, size=(n_hidden, in_size))
        bias = rng.uniform(0.0, 1.0, size=n_hidden)
        H_fit = elm_sigmoid(X_fit @ IW.T + bias)
        H_val = elm_sigmoid(X_val @ IW.T + bias)

        for lam_bc in grid_bc:
            y_fit_bc = box_cox(y_fit_pos, lam_bc)
            for lam_r in grid_r:
                beta = ridge_solve(H_fit, y_fit_bc, lam_r)
                z_val = H_val @ beta
                y_val_pred = np.clip(box_cox_inverse(z_val, lam_bc) - BC_SHIFT, a_min=0.0, a_max=None)
                val_rmse = sqrt(np.mean((y_val_pred - y_val) ** 2))
                if val_rmse < best_val:
                    best_val = val_rmse
                    best_lam_r, best_lam_bc = lam_r, lam_bc
                    best_IW, best_bias = IW, bias

    H_full = elm_sigmoid(X @ best_IW.T + best_bias)
    y_full_bc = box_cox(y + BC_SHIFT, best_lam_bc)
    best_beta = ridge_solve(H_full, y_full_bc, best_lam_r)
    return best_beta, best_IW, best_bias, best_lam_r, best_lam_bc, best_val


def train_elm_box_cox_grid(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden_list: list[int],
    n_candidates_list: list[int],
    rng: np.random.Generator,
):
    best_val = np.inf
    best_beta = best_IW = best_bias = None
    best_h = best_c = best_lam_r = best_lam_bc = None
    for n_hidden in n_hidden_list:
        for n_candidates in n_candidates_list:
            beta, IW, bias, lam_r_sel, lam_bc_sel, val_rmse = train_elm_box_cox(
                X, y, n_hidden, n_candidates, rng,
                lam_grid=LAMBDA_GRID, lam_bc_grid=LAMBDA_BC_GRID, val_ratio=VAL_RATIO,
            )
            print(
                f"    n_hidden={n_hidden:4d}  n_cand={n_candidates:4d}  "
                f"lam_r={lam_r_sel:g}  lam_bc={lam_bc_sel:g}  val_RMSE={val_rmse:.4g}"
            )
            if val_rmse < best_val:
                best_val, best_beta, best_IW, best_bias = val_rmse, beta, IW, bias
                best_h, best_c = n_hidden, n_candidates
                best_lam_r, best_lam_bc = lam_r_sel, lam_bc_sel
    print(
        f"    -> selected: n_hidden={best_h}  n_cand={best_c}  "
        f"lam_r={best_lam_r:g}  lam_bc={best_lam_bc:g}  val_RMSE={best_val:.4g}"
    )
    sel_dict = {
        "n_hidden": best_h, "n_candidates": best_c,
        "lambda_ridge": best_lam_r, "lambda_bc": best_lam_bc,
    }
    # The prediction depends on the selected lam_bc: specialized closure.
    def predict_fn(Xte, beta, IW, bias):
        return elm_predict(Xte, beta, IW, bias, best_lam_bc)

    return best_beta, best_IW, best_bias, sel_dict, predict_fn


def elm_predict(
    X: np.ndarray, beta: np.ndarray, IW: np.ndarray, bias: np.ndarray, lam_bc: float
) -> np.ndarray:
    # PAC is a physical power: clip negative values to 0.
    z = elm_sigmoid(X @ IW.T + bias) @ beta
    return np.clip(box_cox_inverse(z, lam_bc) - BC_SHIFT, a_min=0.0, a_max=None)


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    run_elm(
        slug="box_cox",
        script_name="dr_elm_box_cox.py",
        train_grid=train_elm_box_cox_grid,
        extra_cols=["N_params", "n_hidden", "n_candidates", "lambda_ridge", "lambda_bc"],
        grid_print=f"Box-Cox: lam_r grid={LAMBDA_GRID}  lam_bc grid={LAMBDA_BC_GRID}  shift c={BC_SHIFT}",
    )


if __name__ == "__main__":
    main()
