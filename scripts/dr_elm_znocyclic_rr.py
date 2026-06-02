"""
Ablation variant of dr_elm_robust_risk.py — Robust Risk ELM without the 4
cyclic features.

Model input: only the 48 lags (in_size = 48), no sin/cos
features.

Model reported per (LB, FH):
    - ELM_nocyclic : Robust Risk ELM on [LB lags only, without cyclic features]
"""
from math import floor, sqrt
import numpy as np


from elm_common import VAL_RATIO, elm_sigmoid, run_elm


EPS_GRID: list[float] = [sqrt(10.0), 5.0]


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
    eps_grid: list[float] | None = None,
    val_ratio: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    in_size = X.shape[1]
    n_train = X.shape[0]
    n_fit = max(1, floor((1.0 - val_ratio) * n_train))
    X_fit, X_val = X[:n_fit], X[n_fit:]
    y_fit, y_val = y[:n_fit], y[n_fit:]

    grid = eps_grid if eps_grid else EPS_GRID

    best_val, best_eps, best_IW, best_bias = np.inf, None, None, None

    for _ in range(n_candidates):
        IW = rng.uniform(-1.0, 1.0, size=(n_hidden, in_size))
        bias = rng.uniform(0.0, 1.0, size=n_hidden)
        H_fit = elm_sigmoid(X_fit @ IW.T + bias)
        H_val = elm_sigmoid(X_val @ IW.T + bias)

        for e in grid:
            beta = robust_risk_solve(H_fit, y_fit, e)
            y_val_pred = np.clip(H_val @ beta, a_min=0.0, a_max=None)
            val_rmse = sqrt(np.mean((y_val_pred - y_val) ** 2))
            if val_rmse < best_val:
                best_val, best_eps, best_IW, best_bias = val_rmse, e, IW, bias

    H_full = elm_sigmoid(X @ best_IW.T + best_bias)
    best_beta = robust_risk_solve(H_full, y, best_eps)
    return best_beta, best_IW, best_bias, best_eps, best_val


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
                eps_grid=EPS_GRID, val_ratio=VAL_RATIO,
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
        slug="robust_risk_nocyclic",
        script_name="dr_elm_znocyclic_rr.py",
        train_grid=train_elm_robust_risk_grid,
        extra_cols=["N_params", "n_hidden", "n_candidates", "eps_robust"],
        grid_print=f"Robust Risk: eps grid={EPS_GRID}",
        use_time_features=False,
        method_name="ELM_nocyclic",
        with_baselines=False,
    )


if __name__ == "__main__":
    main()
