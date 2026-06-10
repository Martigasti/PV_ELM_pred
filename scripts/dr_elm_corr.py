"""
ELM with temporal correlation (Corr-ELM) on the PV_AC Palaiseau data.

Loss:
    J_C(beta) = (Y - H beta)^T C (Y - H beta) + sigma^2 ||beta||^2
with C_{ij} = exp(-|t_i - t_j| / tau) (stationary correlation of the residuals).

Analytical solution (closed-form, 1-pass):
    beta_C = (H^T C H + sigma^2 I)^-1 H^T C Y

C acts as the inverse of the noise covariance (GLS).

Hyperparameters: joint grid (sigma^2, tau) selected by validation
RMSE (chronological 20%). Rebuilt once per tau and
reused for each sigma^2.

Environment variables:
    CORR_BANDED=1  : banded approximation C_ij=0 if |i-j| > 5*tau/Δt (default, fast)
    CORR_BANDED=0  : dense C matrix (very slow: O(N^2 * N_h) per tau)

Models reported per (LB, FH):
    - Persistence_P : y_pred = PAC(t)
    - Persistence_Pcyclic : y_pred = PAC(t + FH - 24h)
    - BLEND_opti : convex least-squares combination of the two persistences
    - ELM : Corr-ELM on [LB lags + 4 time features]
"""
import os
from math import sqrt
import numpy as np

from elm_common import CV_FOLDS, elm_sigmoid, run_elm, select_by_temporal_cv


SIGMA2_GRID: list[float] = [10.0, 25.0]
TAU_GRID:    list[float] = [0.5, 2.0, 6.0]  # hours
STEP_HOURS:  float       = 0.5              # data at 30-min step

# Banded approximation: C_ij = 0 if |i-j| > BAND_MULT * tau / STEP_HOURS.
# Switchable from the terminal: CORR_BANDED=1 (default) or CORR_BANDED=0 (dense).
CORR_BANDED: bool = os.environ.get("CORR_BANDED", "1") == "1"
BAND_MULT:   float = 5.0
print(f"*** C MATRIX MODE : {'BANDED' if CORR_BANDED else 'DENSE'} ***")


# ============================================================================
# ELM CORR-ELM
# ============================================================================
def build_C_dense(n: int, tau: float, step_hours: float = STEP_HOURS) -> np.ndarray:
    """C_{ij} = exp(-|t_i - t_j| / tau) in float32. Memory O(n^2)."""
    t = (np.arange(n) * step_hours).astype(np.float32)
    dt = np.abs(t[:, None] - t[None, :])
    return np.exp(-dt / np.float32(tau))


def band_halfwidth(tau: float, step_hours: float = STEP_HOURS, band_mult: float = BAND_MULT) -> int:
    """Half-width K such that C_{ij} ≈ 0 for |i-j| > K (residual weight ≤ exp(-band_mult))."""
    return max(1, int(np.ceil(band_mult * tau / step_hours)))


def banded_matmul(M: np.ndarray, tau: float, K: int, step_hours: float = STEP_HOURS) -> np.ndarray:
    """Compute C @ M where C is banded: C_{ij} = exp(-|i-j| * step / tau) if |i-j| <= K, 0 otherwise.

    Complexity: O(N * K * m) instead of O(N^2 * m) for the dense case.
    M has shape (N,) or (N, m).
    """
    M = np.ascontiguousarray(M, dtype=np.float64)
    n = M.shape[0]
    out = M.copy()  # diagonal k=0: w=1
    offsets = np.arange(1, K + 1)
    weights = np.exp(-offsets * step_hours / tau)
    for k, w in zip(offsets, weights):
        # diagonal +k: out[i] += w * M[i+k]
        out[: n - k] += w * M[k:]
        # diagonal -k: out[i] += w * M[i-k]
        out[k:] += w * M[: n - k]
    return out


def corr_solve_from_precomputed(
    HtCH: np.ndarray, HtCy: np.ndarray, sigma2: float
) -> np.ndarray:
    """Solve (H^T C H + sigma^2 I) beta = H^T C y with H^T C H, H^T C y precomputed."""
    n_hidden = HtCH.shape[0]
    return np.linalg.solve(HtCH + sigma2 * np.eye(n_hidden), HtCy)


def _apply_C(M: np.ndarray, tau: float) -> np.ndarray:
    """Apply the stationary correlation operator C(tau) to M (rows = contiguous
    time steps). Banded approximation by default, dense if CORR_BANDED is off.
    The dense matrix is sized to len(M), so it stays valid on any contiguous
    slice (an expanding-window CV fit set or the whole train set)."""
    if CORR_BANDED:
        return banded_matmul(M, tau, band_halfwidth(tau, STEP_HOURS, BAND_MULT), STEP_HOURS)
    C = build_C_dense(M.shape[0], tau, STEP_HOURS)
    return (C @ M.astype(np.float32)).astype(np.float64)


def train_elm_corr(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden: int,
    n_candidates: int,
    rng: np.random.Generator,
    sigma2_grid: list[float] | None = None,
    tau_grid: list[float] | None = None,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Select (IW, bias, sigma^2, tau) by temporal CV, refit on the full train set."""
    s_grid = sigma2_grid if sigma2_grid else SIGMA2_GRID
    t_grid = tau_grid if tau_grid else TAU_GRID

    def fit_score(X_fit, y_fit, X_val, y_val, IW, bias, combo):
        tau, sigma2 = combo
        H_fit = elm_sigmoid(X_fit @ IW.T + bias)
        HtCH = H_fit.T @ _apply_C(H_fit, tau)
        HtCy = H_fit.T @ _apply_C(y_fit, tau)
        beta = corr_solve_from_precomputed(HtCH, HtCy, sigma2)
        y_val_pred = np.clip(elm_sigmoid(X_val @ IW.T + bias) @ beta, a_min=0.0, a_max=None)
        return sqrt(np.mean((y_val_pred - y_val) ** 2))

    def refit(X_full, y_full, IW, bias, combo):
        tau, sigma2 = combo
        H_full = elm_sigmoid(X_full @ IW.T + bias)
        HtCH = H_full.T @ _apply_C(H_full, tau)
        HtCy = H_full.T @ _apply_C(y_full, tau)
        return corr_solve_from_precomputed(HtCH, HtCy, sigma2), None

    combos = [(tau, sigma2) for tau in t_grid for sigma2 in s_grid]
    beta, IW, bias, combo, _, best_val = select_by_temporal_cv(
        X, y, n_hidden, n_candidates, rng, combos, fit_score, refit, k=k,
    )
    best_tau, best_sigma2 = combo
    return beta, IW, bias, best_sigma2, best_tau, best_val


def train_elm_corr_grid(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden_list: list[int],
    n_candidates_list: list[int],
    rng: np.random.Generator,
):
    best_val = np.inf
    best_beta = best_IW = best_bias = None
    best_h = best_c = best_sigma2 = best_tau = None
    for n_hidden in n_hidden_list:
        for n_candidates in n_candidates_list:
            beta, IW, bias, sigma2_sel, tau_sel, val_rmse = train_elm_corr(
                X, y, n_hidden, n_candidates, rng,
                sigma2_grid=SIGMA2_GRID, tau_grid=TAU_GRID, k=CV_FOLDS,
            )
            print(
                f"    n_hidden={n_hidden:4d}  n_cand={n_candidates:4d}  "
                f"sigma2={sigma2_sel:g}  tau={tau_sel:g}h  val_RMSE={val_rmse:.4g}"
            )
            if val_rmse < best_val:
                best_val, best_beta, best_IW, best_bias = val_rmse, beta, IW, bias
                best_h, best_c = n_hidden, n_candidates
                best_sigma2, best_tau = sigma2_sel, tau_sel
    print(
        f"    -> selected: n_hidden={best_h}  n_cand={best_c}  "
        f"sigma2={best_sigma2:g}  tau={best_tau:g}h  val_RMSE={best_val:.4g}"
    )
    sel_dict = {
        "n_hidden": best_h, "n_candidates": best_c,
        "sigma2_corr": best_sigma2, "tau_corr": best_tau,
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
        slug="corr",
        script_name="dr_elm_corr.py",
        train_grid=train_elm_corr_grid,
        extra_cols=["N_params", "n_hidden", "n_candidates", "sigma2_corr", "tau_corr"],
        grid_print=f"Corr-ELM: sigma2_grid={SIGMA2_GRID}  tau_grid={TAU_GRID} h",
    )


if __name__ == "__main__":
    main()
