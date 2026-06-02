"""
Ablation variant of dr_elm_ridge.py — Ridge ELM without the 4 cyclic features.

Model input: only the 48 lags (in_size = 48), no sin/cos
features.

Model reported per (LB, FH):
    - ELM_nocyclic : Ridge ELM on [LB lags only, without cyclic features]
"""
from math import floor, sqrt
import numpy as np

from elm_common import VAL_RATIO, elm_sigmoid, ridge_solve, run_elm


LAMBDA_GRID: list[float] = [10.0, 25.0]


# ============================================================================
# RIDGE-REGULARIZED ELM
# ============================================================================
def train_elm_ridge(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden: int,
    n_candidates: int,
    rng: np.random.Generator,
    lam_grid: list[float] | None = None,
    val_ratio: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    in_size = X.shape[1]
    n_train = X.shape[0]
    n_fit = max(1, floor((1.0 - val_ratio) * n_train))
    X_fit, X_val = X[:n_fit], X[n_fit:]
    y_fit, y_val = y[:n_fit], y[n_fit:]

    grid = lam_grid if lam_grid else LAMBDA_GRID

    best_val, best_lam, best_IW, best_bias = np.inf, None, None, None

    for _ in range(n_candidates):
        IW = rng.uniform(-1.0, 1.0, size=(n_hidden, in_size))
        bias = rng.uniform(0.0, 1.0, size=n_hidden)
        H_fit = elm_sigmoid(X_fit @ IW.T + bias)
        H_val = elm_sigmoid(X_val @ IW.T + bias)

        for l in grid:
            beta = ridge_solve(H_fit, y_fit, l)
            y_val_pred = np.clip(H_val @ beta, a_min=0.0, a_max=None)
            val_rmse = sqrt(np.mean((y_val_pred - y_val) ** 2))
            if val_rmse < best_val:
                best_val, best_lam, best_IW, best_bias = val_rmse, l, IW, bias

    H_full = elm_sigmoid(X @ best_IW.T + best_bias)
    best_beta = ridge_solve(H_full, y, best_lam)
    return best_beta, best_IW, best_bias, best_lam, best_val


def train_elm_ridge_grid(
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
            beta, IW, bias, lam_sel, val_rmse = train_elm_ridge(
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
    sel_dict = {"n_hidden": best_h, "n_candidates": best_c, "lambda_ridge": best_lam}
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
        slug="ridge_nocyclic",
        script_name="dr_elm_znocyclic_ridge.py",
        train_grid=train_elm_ridge_grid,
        extra_cols=["N_params", "n_hidden", "n_candidates", "lambda_ridge"],
        grid_print=f"Ridge: grid={LAMBDA_GRID}",
        use_time_features=False,
        method_name="ELM_nocyclic",
        with_baselines=False,
    )


if __name__ == "__main__":
    main()
