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
from math import sqrt
import numpy as np

from elm_common import CV_FOLDS, elm_sigmoid, ridge_solve, run_elm, select_by_temporal_cv


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
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    l1s = lam1_grid if lam1_grid else LAMBDA1_GRID
    l2s = lam2_grid if lam2_grid else LAMBDA2_GRID

    def fit_score(X_fit, y_fit, X_val, y_val, IW, bias, combo):
        l1, l2 = combo
        beta = elastic_net_solve(elm_sigmoid(X_fit @ IW.T + bias), y_fit, l1, l2)
        y_val_pred = np.clip(elm_sigmoid(X_val @ IW.T + bias) @ beta, a_min=0.0, a_max=None)
        return sqrt(np.mean((y_val_pred - y_val) ** 2))

    def refit(X_full, y_full, IW, bias, combo):
        l1, l2 = combo
        return elastic_net_solve(elm_sigmoid(X_full @ IW.T + bias), y_full, l1, l2), None

    combos = [(l1, l2) for l1 in l1s for l2 in l2s]
    beta, IW, bias, combo, _, best_val = select_by_temporal_cv(
        X, y, n_hidden, n_candidates, rng, combos, fit_score, refit, k=k,
    )
    return beta, IW, bias, combo[0], combo[1], best_val


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
                lam1_grid=LAMBDA1_GRID, lam2_grid=LAMBDA2_GRID, k=CV_FOLDS,
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
