"""
ELM with M-Estimator (Welsch) cost on the PV_AC Palaiseau data.

Uses a 2-pass method, piecewise version: Pass 1 is a Ridge initialization, Pass 2
is a single reweighted solve with w = exp(-(r^(0)/c)^2) redescending on outliers.

Cost (smooth Welsch, single branch):
    rho(r) = (c^2 / 2) (1 - exp(-(r/c)^2))    for all r

IRLS weights w = psi(r)/r (psi = rho'):
    w_i = exp(-(r_i/c)^2)                     for all r_i

    Pass 1: beta_0 = (H^T H + lam I)^-1 H^T y                  (Ridge init)
    Pass 2: r_i^(0) = y_i - h_i beta_0
             c = 2.985 * MAD(r^(0)) / 0.6745                    (computed once)
             W = diag(exp(-(r_i^(0)/c)^2))
             beta = (H^T W H)^-1 H^T W y                        (a single solve)

Grid-searched hyperparameter: lam. c is derived from the initial Ridge residual.

Models reported per (LB, FH):
    - Persistence_P : y_pred = PAC(t)
    - Persistence_Pcyclic : y_pred = PAC(t + FH - 24h)
    - BLEND_opti : convex least-squares combination of the two persistences
    - ELM : ELM M-Estimator (Welsch) on [LB lags + 4 time features]
"""
from math import floor, sqrt
import numpy as np

from elm_common import VAL_RATIO, elm_sigmoid, ridge_solve, run_elm


K_WELSCH: float = 2.985
LAMBDA_GRID: list[float] = [10.0, 25.0]
EPS_W: float = 1e-6


# ============================================================================
# ELM-M-Estimator (smooth Welsch) 2-pass
# ============================================================================
def _mad_sigma(r: np.ndarray) -> float:
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    return float(mad / 0.6745)


def m_estimator_solve(
    H: np.ndarray,
    y: np.ndarray,
    lam: float,
    k: float = K_WELSCH,
    eps: float = EPS_W,
) -> tuple[np.ndarray, float]:
    """Strict 2-pass (smooth Welsch): Pass 1 Ridge, Pass 2 a single weighted solve.

    c = k * MAD(r^(0)) / 0.6745: robust sigma_hat (MAD resists the outliers
    that std would inflate), k=2.985 is the Welsch constant (95% efficiency vs
    OLS under Gaussian), computed once on the Ridge residuals."""
    beta0 = ridge_solve(H, y, lam)
    r = y - H @ beta0
    sigma = _mad_sigma(r)
    c = max(k * sigma, eps)
    w = np.exp(-(r / c) ** 2)
    WH = H * w[:, None]
    A = H.T @ WH
    b = H.T @ (w * y)
    beta = np.linalg.solve(A, b)
    return beta, c


def train_elm_m_estimator(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden: int,
    n_candidates: int,
    rng: np.random.Generator,
    lam_grid: list[float] | None = None,
    val_ratio: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    in_size = X.shape[1]
    n_train = X.shape[0]
    n_fit = max(1, floor((1.0 - val_ratio) * n_train))
    X_fit, X_val = X[:n_fit], X[n_fit:]
    y_fit, y_val = y[:n_fit], y[n_fit:]

    lams = lam_grid if lam_grid else LAMBDA_GRID

    best_val = np.inf
    best_IW = best_bias = None
    best_lam = best_c = None

    for _ in range(n_candidates):
        IW = rng.uniform(-1.0, 1.0, size=(n_hidden, in_size))
        bias = rng.uniform(0.0, 1.0, size=n_hidden)
        H_fit = elm_sigmoid(X_fit @ IW.T + bias)
        H_val = elm_sigmoid(X_val @ IW.T + bias)

        for lam in lams:
            beta, c_sel = m_estimator_solve(H_fit, y_fit, lam)
            y_val_pred = np.clip(H_val @ beta, a_min=0.0, a_max=None)
            val_rmse = sqrt(np.mean((y_val_pred - y_val) ** 2))
            if val_rmse < best_val:
                best_val = val_rmse
                best_IW, best_bias = IW, bias
                best_lam, best_c = lam, c_sel

    H_full = elm_sigmoid(X @ best_IW.T + best_bias)
    best_beta, best_c = m_estimator_solve(H_full, y, best_lam)
    return best_beta, best_IW, best_bias, best_lam, best_c, best_val


def train_elm_m_estimator_grid(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden_list: list[int],
    n_candidates_list: list[int],
    rng: np.random.Generator,
):
    best_val = np.inf
    best_beta = best_IW = best_bias = None
    best_h = best_cand = None
    best_lam = best_c = None
    for n_hidden in n_hidden_list:
        for n_candidates in n_candidates_list:
            beta, IW, bias, lam_sel, c_sel, val_rmse = train_elm_m_estimator(
                X, y, n_hidden, n_candidates, rng,
                lam_grid=LAMBDA_GRID, val_ratio=VAL_RATIO,
            )
            print(
                f"    n_hidden={n_hidden:4d}  n_cand={n_candidates:4d}  "
                f"lam={lam_sel:g}  c={c_sel:.4g}  val_RMSE={val_rmse:.4g}"
            )
            if val_rmse < best_val:
                best_val, best_beta, best_IW, best_bias = val_rmse, beta, IW, bias
                best_h, best_cand = n_hidden, n_candidates
                best_lam, best_c = lam_sel, c_sel
    print(
        f"    -> selected: n_hidden={best_h}  n_cand={best_cand}  "
        f"lam={best_lam:g}  c={best_c:.4g}  val_RMSE={best_val:.4g}"
    )
    sel_dict = {
        "n_hidden": best_h, "n_candidates": best_cand,
        "lambda_m_est": best_lam, "c_welsch": best_c,
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
        slug="m_estimator",
        script_name="dr_elm_m_estimator.py",
        train_grid=train_elm_m_estimator_grid,
        extra_cols=["N_params", "n_hidden", "n_candidates", "lambda_m_est", "c_welsch"],
        grid_print=f"M-Estimator 2-pass (Welsch lisse): k={K_WELSCH} (c = k * MAD), lam_grid={LAMBDA_GRID}",
    )


if __name__ == "__main__":
    main()
