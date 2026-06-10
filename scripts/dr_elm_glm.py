"""
ELM with linear GLM cost on the PV_AC Palaiseau data.

Uses a 2-pass Fisher linearization: Pass 1 is a Ridge initialization on y,
Pass 2 is a single reweighted solve in log space using the adjusted GLM variable

The model:
    g(E[Y]) = H beta,    g(mu) = log(mu),    Var(Y) ~ mu^2  (Gamma)

is solved by Fisher linearization around mu_hat = H beta_0:

    Pass 1: beta_0 = (H^T H + lam I)^-1 H^T y                  (Ridge on y)
    Pass 2: mu_hat_i = max(H_i beta_0, eps_mu)                 (clip for log)
             z_i     = log(mu_hat_i) + (y_i + c - mu_hat_i) / mu_hat_i
                                                                (adjusted variable)
             W_eta_i = mu_hat_i^2                               (see note below)
             beta    = (H^T W_eta H + lam I)^-1 H^T W_eta z     (a single solve)
    Prediction: y_pred = max(exp(H beta) - c, 0)

Grid-searched hyperparameters: (lam, c).

Models reported per (LB, FH):
    - Persistence_P : y_pred = PAC(t)
    - Persistence_Pcyclic : y_pred = PAC(t + FH - 24h)
    - BLEND_opti : convex least-squares combination of the two persistences
    - ELM : ELM-GLM (Gamma + log link) on [LB lags + 4 time features]
"""
from math import floor, sqrt
import numpy as np

from elm_common import CV_FOLDS, elm_sigmoid, ridge_solve, run_elm, select_by_temporal_cv


LAMBDA_GRID: list[float] = [25.0]
C_GRID: list[float] = [0.1, 1.0, 10.0, 100.0]
EPS_MU: float = 1e-3  # floor on mu_hat before log


# ============================================================================
# ELM GLM (Gamma + log link) via 2-pass
# ============================================================================
def glm_solve(
    H: np.ndarray,
    y: np.ndarray,
    lam: float,
    c: float,
) -> np.ndarray:
    """Strict 2-pass : Pass 1 Ridge on y, Pass 2 a single GLM solve.

    Gamma + log link: mu_hat = H beta_0 (clip to eps_mu),
    z = log(mu_hat) + (y + c - mu_hat) / mu_hat,
    W_eta = diag(mu_hat^2) (proportional form, see docstring),
    beta = (H^T W H + lam I)^-1 H^T W z   (beta lives in log space).
    """
    n_hidden = H.shape[1]
    # Pass 1: Ridge on y (original space)
    beta0 = ridge_solve(H, y, lam)
    # Pass 2: Fisher linearization around mu_hat (clip for well-defined log)
    mu_hat = np.maximum(H @ beta0, EPS_MU)
    z = np.log(mu_hat) + (y + c - mu_hat) / mu_hat
    w = mu_hat * mu_hat
    WH = H * w[:, None]
    A = H.T @ WH + lam * np.eye(n_hidden)
    b = H.T @ (w * z)
    return np.linalg.solve(A, b)


def glm_predict_raw(H: np.ndarray, beta: np.ndarray, c: float) -> np.ndarray:
    """Prediction: H beta lives in log space, y_pred = exp(H beta) - c."""
    eta = H @ beta
    # clip eta to avoid overflow in exp; PV max ~ 1e4 W, log(1e4 + 100) ~ 9.2
    eta = np.minimum(eta, 20.0)
    return np.clip(np.exp(eta) - c, a_min=0.0, a_max=None)


def train_elm_glm(
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
        beta = glm_solve(elm_sigmoid(X_fit @ IW.T + bias), y_fit, lam, c_val)
        y_val_pred = glm_predict_raw(elm_sigmoid(X_val @ IW.T + bias), beta, c_val)
        return sqrt(np.mean((y_val_pred - y_val) ** 2))

    def refit(X_full, y_full, IW, bias, combo):
        lam, c_val = combo
        return glm_solve(elm_sigmoid(X_full @ IW.T + bias), y_full, lam, c_val), None

    combos = [(l, c_val) for l in grid_l for c_val in grid_c]
    beta, IW, bias, combo, _, best_val = select_by_temporal_cv(
        X, y, n_hidden, n_candidates, rng, combos, fit_score, refit, k=k,
    )
    return beta, IW, bias, combo[0], combo[1], best_val


def train_elm_glm_grid(
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
            beta, IW, bias, lam_sel, c_sel, val_rmse = train_elm_glm(
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
        "lambda_glm": best_lam, "c_glm": best_c_shift,
    }
    # The prediction depends on the selected c: specialized closure.
    def predict_fn(Xte, beta, IW, bias):
        return elm_predict(Xte, beta, IW, bias, best_c_shift)

    return best_beta, best_IW, best_bias, sel_dict, predict_fn


def elm_predict(
    X: np.ndarray, beta: np.ndarray, IW: np.ndarray, bias: np.ndarray, c: float
) -> np.ndarray:
    H = elm_sigmoid(X @ IW.T + bias)
    return glm_predict_raw(H, beta, c)


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    run_elm(
        slug="glm",
        script_name="dr_elm_glm.py",
        train_grid=train_elm_glm_grid,
        extra_cols=["N_params", "n_hidden", "n_candidates", "lambda_glm", "c_glm"],
        grid_print=f"GLM (Gamma+log) 2-pass: lam_grid={LAMBDA_GRID}, c_grid={C_GRID}",
    )


if __name__ == "__main__":
    main()
