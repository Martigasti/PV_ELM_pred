"""
ELM with Log-MSE cost on the PV_AC Palaiseau data.

The loss is:
    J(beta) = sum_i (log(y_i + c) - log(h_i beta + c))^2 + lam * ||beta||^2

with shift c > 0 (defined even at night, y = 0) grid-searched in C_GRID.

DEVIATION FROM PDF Sujet_ELM_OPTI-6 (Annexe A.2). The PDF prescribes a 2-pass
linearization of log(h_i beta + c) around a Ridge init, with beta living in
*original* space (y_pred = H beta). On PV data (hard zeros at night), that
formula is **degenerate**: it collapses every prediction to ~0 for any c in
the grid (verified empirically -- nMBE = -0.995, RMSE pinned across horizons,
NICE_Sigma > 1 i.e. worse than persistence). Two compounding causes:
  1. log(y+c) is dominated by the night mass (y=0), and the W = 1/(yhat0+c)^2
     weights blow up wherever yhat0 ~ 0, dragging H beta negative -> clipped 0.
  2. The PDF's RHS uses W = G^2 (G = 1/(yhat0+c)); the true Gauss-Newton RHS
     for design matrix G H is H^T G ytilde (one power of G). Even with that
     fixed, cause #1 still collapses the fit.

Instead we put beta in **log space** (the coherent reading noted in CLAUDE.md):
the loss is then quadratic in beta -> pure Ridge on z = log(y + c), no
linearization, single closed-form solve, and the log link is genuinely active.

    z_i  = log(y_i + c)
    beta = (H^T H + lam I)^-1 H^T z              (Ridge on z, log space)
    y_pred = exp(H beta) - c                     (clipped at 0)

Grid-searched hyperparameters: lam, c.

Models reported per (LB, FH):
    - Persistence_P : y_pred = PAC(t)
    - Persistence_Pcyclic : y_pred = PAC(t + FH - 24h)
    - BLEND_opti : convex least-squares combination of the two persistences
    - ELM : ELM-Log-MSE on [LB lags + 4 time features]
"""
from math import sqrt
import numpy as np

from elm_common import CV_FOLDS, elm_sigmoid, ridge_solve, run_elm, select_by_temporal_cv


LAMBDA_GRID: list[float] = [25.0]
C_GRID: list[float] = [0.1, 1.0, 10.0, 100.0]


# ============================================================================
# ELM-Log-MSE (log-space beta, closed-form Ridge on log(y+c))
# ============================================================================
def log_mse_solve(
    H: np.ndarray,
    y: np.ndarray,
    lam: float,
    c: float,
) -> np.ndarray:
    """Ridge on z = log(y + c): closed-form, beta lives in log space."""
    return ridge_solve(H, np.log(y + c), lam)


def log_mse_predict_raw(H: np.ndarray, beta: np.ndarray, c: float) -> np.ndarray:
    """beta lives in log space: y_pred = exp(H beta) - c, clipped at 0."""
    return np.clip(np.exp(H @ beta) - c, a_min=0.0, a_max=None)


def train_elm_log_mse(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden: int,
    n_candidates: int,
    rng: np.random.Generator,
    lam_grid: list[float] | None = None,
    c_grid: list[float] | None = None,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    grid_l = lam_grid if lam_grid else LAMBDA_GRID
    grid_c = c_grid if c_grid else C_GRID

    def fit_score(X_fit, y_fit, X_val, y_val, IW, bias, combo):
        lam, c_val = combo
        beta = log_mse_solve(elm_sigmoid(X_fit @ IW.T + bias), y_fit, lam, c_val)
        y_val_pred = log_mse_predict_raw(elm_sigmoid(X_val @ IW.T + bias), beta, c_val)
        return sqrt(np.mean((y_val_pred - y_val) ** 2))

    def refit(X_full, y_full, IW, bias, combo):
        lam, c_val = combo
        return log_mse_solve(elm_sigmoid(X_full @ IW.T + bias), y_full, lam, c_val), None

    combos = [(l, c_val) for l in grid_l for c_val in grid_c]
    beta, IW, bias, combo, _, best_val = select_by_temporal_cv(
        X, y, n_hidden, n_candidates, rng, combos, fit_score, refit, k=k,
    )
    return beta, IW, bias, combo[0], combo[1], best_val


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
                lam_grid=LAMBDA_GRID, c_grid=C_GRID, k=CV_FOLDS,
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

    def predict_fn(
        X: np.ndarray, beta: np.ndarray, IW: np.ndarray, bias: np.ndarray
    ) -> np.ndarray:
        H = elm_sigmoid(X @ IW.T + bias)
        return log_mse_predict_raw(H, beta, c=best_c_shift)

    return best_beta, best_IW, best_bias, sel_dict, predict_fn


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    run_elm(
        slug="log_mse",
        script_name="dr_elm_log_mse.py",
        train_grid=train_elm_log_mse_grid,
        extra_cols=["N_params", "n_hidden", "n_candidates", "lambda_log_mse", "c_log_mse"],
        grid_print=f"Log-MSE (log-space beta, Ridge on log(y+c)): lam_grid={LAMBDA_GRID}, c_grid={C_GRID}",
    )


if __name__ == "__main__":
    main()
