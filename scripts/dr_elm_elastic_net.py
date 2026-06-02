"""
Elastic Net-regularized ELM forecasting on the PV_AC Palaiseau data.

Uses a 2-pass method: Pass 1 is a Ridge(lam2) initialization, Pass 2 is a single
reweighted solve linearizing the L1 term via W_EN = diag(1 / (|beta_0,j| + eps)).

Elastic Net cost:
    J_EN(beta) = || H beta - y ||_2^2 + lam2 * ||beta||_2^2 + lam1 * ||beta||_1


Pass 1: beta_0 = (H^T H + lam2 I)^-1 H^T y                       (Ridge init)
Pass 2: W_EN = diag( 1 / (|beta_0,j| + eps) )
             beta = (H^T H + lam2 I + lam1 W_EN)^-1 H^T y              (a single solve)

Hyperparameters: (lam1, lam2) selected by validation RMSE.

Models reported per (LB, FH):
    - Persistence_P : y_pred = PAC(t)
    - Persistence_Pcyclic : y_pred = PAC(t + FH - 24h)
    - BLEND_opti : convex least-squares combination of the two persistences
    - ELM : ELM Elastic Net on [LB lags + 4 time features]
"""
from math import floor, sqrt
import numpy as np

from elm_common import VAL_RATIO, elm_sigmoid, ridge_solve, run_elm


LAMBDA1_GRID: list[float] = [1.0, 10.0]
LAMBDA2_GRID: list[float] = [10.0, 25.0]
EPS_W: float = 1e-6


# ============================================================================
# ELASTIC NET-REGULARIZED ELM (strict 2-pass)
# ============================================================================
def elastic_net_solve(
    H: np.ndarray,
    y: np.ndarray,
    lam1: float,
    lam2: float,
    eps: float = EPS_W,
) -> np.ndarray:
    """Strict 2-pass (PDF §A.5).
    Pass 1: beta_0 = Ridge(lam2). Pass 2: a single solve with W_EN = diag(1/(|beta_0,j|+eps))."""
    beta0 = ridge_solve(H, y, lam2)
    w_en = 1.0 / (np.abs(beta0) + eps)
    n_hidden = H.shape[1]
    A = H.T @ H + lam2 * np.eye(n_hidden) + lam1 * np.diag(w_en)
    b = H.T @ y
    return np.linalg.solve(A, b)


def train_elm_elastic_net(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden: int,
    n_candidates: int,
    rng: np.random.Generator,
    lam1_grid: list[float] | None = None,
    lam2_grid: list[float] | None = None,
    val_ratio: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    in_size = X.shape[1]
    n_train = X.shape[0]
    n_fit = max(1, floor((1.0 - val_ratio) * n_train))
    X_fit, X_val = X[:n_fit], X[n_fit:]
    y_fit, y_val = y[:n_fit], y[n_fit:]

    l1s = lam1_grid if lam1_grid else LAMBDA1_GRID
    l2s = lam2_grid if lam2_grid else LAMBDA2_GRID

    best_val = np.inf
    best_IW = best_bias = None
    best_lam1 = best_lam2 = None

    for _ in range(n_candidates):
        IW = rng.uniform(-1.0, 1.0, size=(n_hidden, in_size))
        bias = rng.uniform(0.0, 1.0, size=n_hidden)
        H_fit = elm_sigmoid(X_fit @ IW.T + bias)
        H_val = elm_sigmoid(X_val @ IW.T + bias)

        for l1 in l1s:
            for l2 in l2s:
                beta = elastic_net_solve(H_fit, y_fit, l1, l2)
                y_val_pred = np.clip(H_val @ beta, a_min=0.0, a_max=None)
                val_rmse = sqrt(np.mean((y_val_pred - y_val) ** 2))
                if val_rmse < best_val:
                    best_val = val_rmse
                    best_IW, best_bias = IW, bias
                    best_lam1, best_lam2 = l1, l2

    H_full = elm_sigmoid(X @ best_IW.T + best_bias)
    best_beta = elastic_net_solve(H_full, y, best_lam1, best_lam2)
    return best_beta, best_IW, best_bias, best_lam1, best_lam2, best_val


def train_elm_elastic_net_grid(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden_list: list[int],
    n_candidates_list: list[int],
    rng: np.random.Generator,
):
    best_val = np.inf
    best_beta = best_IW = best_bias = None
    best_h = best_c = None
    best_lam1 = best_lam2 = None
    for n_hidden in n_hidden_list:
        for n_candidates in n_candidates_list:
            beta, IW, bias, l1_sel, l2_sel, val_rmse = train_elm_elastic_net(
                X, y, n_hidden, n_candidates, rng,
                lam1_grid=LAMBDA1_GRID, lam2_grid=LAMBDA2_GRID, val_ratio=VAL_RATIO,
            )
            print(
                f"    n_hidden={n_hidden:4d}  n_cand={n_candidates:4d}  "
                f"lam1={l1_sel:g}  lam2={l2_sel:g}  val_RMSE={val_rmse:.4g}"
            )
            if val_rmse < best_val:
                best_val, best_beta, best_IW, best_bias = val_rmse, beta, IW, bias
                best_h, best_c = n_hidden, n_candidates
                best_lam1, best_lam2 = l1_sel, l2_sel
    print(
        f"    -> selected: n_hidden={best_h}  n_cand={best_c}  "
        f"lam1={best_lam1:g}  lam2={best_lam2:g}  val_RMSE={best_val:.4g}"
    )
    sel_dict = {
        "n_hidden": best_h, "n_candidates": best_c,
        "lambda1_en": best_lam1, "lambda2_en": best_lam2,
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
        slug="elastic_net",
        script_name="dr_elm_elastic_net.py",
        train_grid=train_elm_elastic_net_grid,
        extra_cols=["N_params", "n_hidden", "n_candidates", "lambda1_en", "lambda2_en"],
        grid_print=f"Elastic Net 2-pass: lam1_grid={LAMBDA1_GRID}, lam2_grid={LAMBDA2_GRID}",
    )


if __name__ == "__main__":
    main()
