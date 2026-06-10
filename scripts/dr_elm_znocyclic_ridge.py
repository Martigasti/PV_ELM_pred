"""
Ablation variant of dr_elm_ridge.py — Ridge ELM without the 4 cyclic features.

Model input: only the 48 lags (in_size = 48), no sin/cos
features.

Model reported per (LB, FH):
    - ELM_nocyclic : Ridge ELM on [LB lags only, without cyclic features]
"""
from math import sqrt
import numpy as np

from elm_common import CV_FOLDS, elm_sigmoid, ridge_solve, run_elm, select_by_temporal_cv


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
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    grid = lam_grid if lam_grid else LAMBDA_GRID

    def fit_score(X_fit, y_fit, X_val, y_val, IW, bias, combo):
        (lam,) = combo
        beta = ridge_solve(elm_sigmoid(X_fit @ IW.T + bias), y_fit, lam)
        y_val_pred = np.clip(elm_sigmoid(X_val @ IW.T + bias) @ beta, a_min=0.0, a_max=None)
        return sqrt(np.mean((y_val_pred - y_val) ** 2))

    def refit(X_full, y_full, IW, bias, combo):
        (lam,) = combo
        return ridge_solve(elm_sigmoid(X_full @ IW.T + bias), y_full, lam), None

    combos = [(lam,) for lam in grid]
    beta, IW, bias, combo, _, best_val = select_by_temporal_cv(
        X, y, n_hidden, n_candidates, rng, combos, fit_score, refit, k=k,
    )
    return beta, IW, bias, combo[0], best_val


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
