"""
ELM with Huber cost on the PV_AC Palaiseau data.

Uses a 2-pass method: Pass 1 is a Ridge initialization, Pass 2 is a single
reweighted solve with W = diag(min(1, delta / |r^(0)|)) downweighting outliers.

Huber cost:
    rho_delta(r) = (1/2) r^2              if |r| <= delta
                   delta (|r| - delta/2)  if |r| > delta

    Pass 1: beta_0 = (H^T H + lam I)^-1 H^T y                  (Ridge init)
    Pass 2: r_i^(0) = y_i - h_i beta_0
             delta = 1.345 * MAD(r^(0)) / 0.6745                (computed once)
             W = diag(min(1, delta / |r_i^(0)|))
             beta = (H^T W H)^-1 H^T W y                        (a single solve)

Grid-searched hyperparameter: lam. (delta is derived from the initial Ridge residual)

Models reported per (LB, FH):
    - Persistence_P : y_pred = PAC(t)
    - Persistence_Pcyclic : y_pred = PAC(t + FH - 24h)
    - BLEND_opti : convex least-squares combination of the two persistences
    - ELM : ELM-Huber on [LB lags + 4 time features]
"""
from math import sqrt
import numpy as np

from elm_common import CV_FOLDS, elm_sigmoid, ridge_solve, run_elm, select_by_temporal_cv


HUBER_K: float = 1.345
LAMBDA_GRID: list[float] = [10.0, 25.0]
EPS_W: float = 1e-6


# ============================================================================
# ELM-Huber via IRLS (delta adaptive via MAD)
# ============================================================================
def _mad_sigma(r: np.ndarray) -> float:
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    return float(mad / 0.6745)


def huber_solve(
    H: np.ndarray,
    y: np.ndarray,
    lam: float,
    k: float = HUBER_K,
    eps: float = EPS_W,
) -> tuple[np.ndarray, float]:
    """Strict 2-pass (PDF §A.3): Pass 1 Ridge, Pass 2 a single Huber-weighted solve.

    delta = k * MAD(r^(0)) / 0.6745: robust sigma_hat (MAD resists the outliers
    that std would inflate), k=1.345 is the Huber constant (95% efficiency vs OLS
    under Gaussian), computed once on the Ridge residuals"""
    beta0 = ridge_solve(H, y, lam)
    r = y - H @ beta0
    sigma = _mad_sigma(r)
    delta = max(k * sigma, eps)
    abs_r = np.abs(r)
    w = np.where(abs_r <= delta, 1.0, delta / np.maximum(abs_r, eps))
    WH = H * w[:, None]
    A = H.T @ WH
    b = H.T @ (w * y)
    beta = np.linalg.solve(A, b)
    return beta, delta


def train_elm_huber(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden: int,
    n_candidates: int,
    rng: np.random.Generator,
    lam_grid: list[float] | None = None,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    lams = lam_grid if lam_grid else LAMBDA_GRID

    def fit_score(X_fit, y_fit, X_val, y_val, IW, bias, combo):
        (lam,) = combo
        beta, _ = huber_solve(elm_sigmoid(X_fit @ IW.T + bias), y_fit, lam)
        y_val_pred = np.clip(elm_sigmoid(X_val @ IW.T + bias) @ beta, a_min=0.0, a_max=None)
        return sqrt(np.mean((y_val_pred - y_val) ** 2))

    def refit(X_full, y_full, IW, bias, combo):
        (lam,) = combo
        return huber_solve(elm_sigmoid(X_full @ IW.T + bias), y_full, lam)  # (beta, delta)

    combos = [(lam,) for lam in lams]
    beta, IW, bias, combo, delta, best_val = select_by_temporal_cv(
        X, y, n_hidden, n_candidates, rng, combos, fit_score, refit, k=k,
    )
    return beta, IW, bias, combo[0], delta, best_val


def train_elm_huber_grid(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden_list: list[int],
    n_candidates_list: list[int],
    rng: np.random.Generator,
    lam_grid: list[float] | None = None,
    k: int = CV_FOLDS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float, float]:
    best_val = np.inf
    best_beta = best_IW = best_bias = None
    best_h = best_c = None
    best_lam = best_delta = None
    for n_hidden in n_hidden_list:
        for n_candidates in n_candidates_list:
            beta, IW, bias, lam_sel, delta_sel, val_rmse = train_elm_huber(
                X, y, n_hidden, n_candidates, rng,
                lam_grid=lam_grid, k=k,
            )
            print(
                f"    n_hidden={n_hidden:4d}  n_cand={n_candidates:4d}  "
                f"lam={lam_sel:g}  delta={delta_sel:.4g}  val_RMSE={val_rmse:.4g}"
            )
            if val_rmse < best_val:
                best_val, best_beta, best_IW, best_bias = val_rmse, beta, IW, bias
                best_h, best_c = n_hidden, n_candidates
                best_lam, best_delta = lam_sel, delta_sel
    print(
        f"    -> selected: n_hidden={best_h}  n_cand={best_c}  "
        f"lam={best_lam:g}  delta={best_delta:.4g}  val_RMSE={best_val:.4g}"
    )
    sel_dict = {
        "n_hidden": best_h, "n_candidates": best_c,
        "lambda_huber": best_lam, "delta_huber": best_delta,
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
        slug="huber",
        script_name="dr_elm_huber.py",
        train_grid=train_elm_huber_grid,
        extra_cols=["N_params", "n_hidden", "n_candidates", "lambda_huber", "delta_huber"],
        grid_print=f"Huber 2-pass: k={HUBER_K} (delta = k * MAD), lam_grid={LAMBDA_GRID}",
    )


if __name__ == "__main__":
    main()
