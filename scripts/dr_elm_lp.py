"""
ELM with general L_p cost on the PV_AC Palaiseau data.

Uses a 2-pass IRLS method with p in [1, 2]: Pass 1 is a Ridge initialization,
Pass 2 is a single reweighted solve using residuals from Pass 1:

    J_p(beta) = sum_i |y_i - h_i^T beta|^p + lam ||beta||_2^2

    Pass 1: beta_0 = (H^T H + lam I)^-1 H^T y           (Ridge init)
    Pass 2: r_i^(0) = y_i - h_i beta_0
             W_p    = diag(max(|r_i^(0)|, eps)^(p - 2))
             beta   = (H^T W_p H)^-1 H^T W_p y           (a single solve)


Generalization: continuously interpolates between MAE (p=1) and Ridge (p=2). For
p < 2, the weights w_i = |r_i|^(p-2) downweight the large residuals
(robust behavior, intermediate between MAE/Ridge).

Grid-searched hyperparameters: (lam, p).

Models reported per (LB, FH):
    - Persistence_P : y_pred = PAC(t)
    - Persistence_Pcyclic : y_pred = PAC(t + FH - 24h)
    - BLEND_opti : convex least-squares combination of the two persistences
    - ELM : ELM-L_p on [LB lags + 4 time features]
"""
from math import floor, sqrt
import numpy as np

from elm_common import VAL_RATIO, elm_sigmoid, ridge_solve, run_elm


LAMBDA_GRID: list[float] = [10.0, 25.0]
P_GRID: list[float] = [1.25, 1.5, 1.75]
EPS_W: float = 1e-6


# ============================================================================
# ELM-L_p 2-pass
# ============================================================================
def lp_solve(
    H: np.ndarray, y: np.ndarray, lam: float, p: float, eps: float = EPS_W
) -> np.ndarray:
    """Strict 2-pass: Pass 1 Ridge init, Pass 2 a single weighted solve.

    W_p = diag(|r|^(p-2)) with floor max(|r|, eps) to avoid
    division by zero when p<2 (negative exponent).
    """
    beta0 = ridge_solve(H, y, lam)
    r = y - H @ beta0
    abs_r = np.maximum(np.abs(r), eps)
    w = abs_r ** (p - 2.0)
    WH = H * w[:, None]
    A = H.T @ WH
    b = H.T @ (w * y)
    return np.linalg.solve(A, b)


def train_elm_lp(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden: int,
    n_candidates: int,
    rng: np.random.Generator,
    lam_grid: list[float] | None = None,
    p_grid: list[float] | None = None,
    val_ratio: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    in_size = X.shape[1]
    n_train = X.shape[0]
    n_fit = max(1, floor((1.0 - val_ratio) * n_train))
    X_fit, X_val = X[:n_fit], X[n_fit:]
    y_fit, y_val = y[:n_fit], y[n_fit:]

    lams = lam_grid if lam_grid else LAMBDA_GRID
    ps = p_grid if p_grid else P_GRID

    best_val = np.inf
    best_IW = best_bias = None
    best_lam = best_p = None

    for _ in range(n_candidates):
        IW = rng.uniform(-1.0, 1.0, size=(n_hidden, in_size))
        bias = rng.uniform(0.0, 1.0, size=n_hidden)
        H_fit = elm_sigmoid(X_fit @ IW.T + bias)
        H_val = elm_sigmoid(X_val @ IW.T + bias)

        for lam in lams:
            for p in ps:
                beta = lp_solve(H_fit, y_fit, lam, p)
                y_val_pred = np.clip(H_val @ beta, a_min=0.0, a_max=None)
                val_rmse = sqrt(np.mean((y_val_pred - y_val) ** 2))
                if val_rmse < best_val:
                    best_val = val_rmse
                    best_IW, best_bias = IW, bias
                    best_lam, best_p = lam, p

    H_full = elm_sigmoid(X @ best_IW.T + best_bias)
    best_beta = lp_solve(H_full, y, best_lam, best_p)
    return best_beta, best_IW, best_bias, best_lam, best_p, best_val


def train_elm_lp_grid(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden_list: list[int],
    n_candidates_list: list[int],
    rng: np.random.Generator,
):
    best_val = np.inf
    best_beta = best_IW = best_bias = None
    best_h = best_c = None
    best_lam = best_p = None
    for n_hidden in n_hidden_list:
        for n_candidates in n_candidates_list:
            beta, IW, bias, lam_sel, p_sel, val_rmse = train_elm_lp(
                X, y, n_hidden, n_candidates, rng,
                lam_grid=LAMBDA_GRID, p_grid=P_GRID, val_ratio=VAL_RATIO,
            )
            print(
                f"    n_hidden={n_hidden:4d}  n_cand={n_candidates:4d}  "
                f"lam={lam_sel:g}  p={p_sel:g}  val_RMSE={val_rmse:.4g}"
            )
            if val_rmse < best_val:
                best_val, best_beta, best_IW, best_bias = val_rmse, beta, IW, bias
                best_h, best_c = n_hidden, n_candidates
                best_lam, best_p = lam_sel, p_sel
    print(
        f"    -> selected: n_hidden={best_h}  n_cand={best_c}  "
        f"lam={best_lam:g}  p={best_p:g}  val_RMSE={best_val:.4g}"
    )
    sel_dict = {
        "n_hidden": best_h, "n_candidates": best_c,
        "lambda_lp": best_lam, "p_lp": best_p,
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
        slug="lp",
        script_name="dr_elm_lp.py",
        train_grid=train_elm_lp_grid,
        extra_cols=["N_params", "n_hidden", "n_candidates", "lambda_lp", "p_lp"],
        grid_print=f"L_p 2-pass: lam_grid={LAMBDA_GRID}, p_grid={P_GRID}",
    )


if __name__ == "__main__":
    main()
