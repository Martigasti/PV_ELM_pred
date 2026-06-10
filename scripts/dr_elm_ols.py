"""
ELM forecasting on the PV_AC Palaiseau data, using the pseudo-inverse (OLS).

Output weights obtained with the Moore-Penrose pseudo-inverse:

    beta = pinv(H) @ y minimizes ||H beta - y||^2

Models reported per (LB, FH):
    - Persistence_P : y_pred = PAC(t)
    - Persistence_Pcyclic : y_pred = PAC(t + FH - 24h)
    - BLEND_opti : convex least-squares combination of the two persistences
    - ELM : pseudo-inverse ELM on [LB lags + 4 time features]
"""
from math import sqrt
import numpy as np

from elm_common import CV_FOLDS, elm_sigmoid, run_elm, select_by_temporal_cv


# ============================================================================
# ELM TRAINING (pseudo-inverse / OLS)
# ============================================================================
def train_elm(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden: int,
    n_candidates: int,
    rng: np.random.Generator,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # Temporal CV on expanding windows prevents the (IW, bias) selection from
    # being rewarded merely by a larger n_hidden (which mechanically reduces
    # the training RMSE), while averaging over several seasons.
    def fit_score(X_fit, y_fit, X_val, y_val, IW, bias, combo):
        H_fit = elm_sigmoid(X_fit @ IW.T + bias)
        beta, *_ = np.linalg.lstsq(H_fit, y_fit, rcond=None)
        y_val_pred = np.clip(elm_sigmoid(X_val @ IW.T + bias) @ beta, a_min=0.0, a_max=None)
        return sqrt(np.mean((y_val_pred - y_val) ** 2))

    def refit(X_full, y_full, IW, bias, combo):
        H_full = elm_sigmoid(X_full @ IW.T + bias)
        beta, *_ = np.linalg.lstsq(H_full, y_full, rcond=None)
        return beta, None

    beta, IW, bias, _, _, best_val = select_by_temporal_cv(
        X, y, n_hidden, n_candidates, rng, [()], fit_score, refit, k=k,
    )
    return beta, IW, bias, best_val


def train_elm_grid(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden_list: list[int],
    n_candidates_list: list[int],
    rng: np.random.Generator,
):
    best_val = np.inf
    best_beta = best_IW = best_bias = None
    best_h = best_c = None
    for n_hidden in n_hidden_list:
        for n_candidates in n_candidates_list:
            beta, IW, bias, val_rmse = train_elm(
                X, y, n_hidden, n_candidates, rng, k=CV_FOLDS
            )
            print(
                f"    n_hidden={n_hidden:4d}  n_cand={n_candidates:4d}  "
                f"val_RMSE={val_rmse:.4g}"
            )
            if val_rmse < best_val:
                best_val, best_beta, best_IW, best_bias = val_rmse, beta, IW, bias
                best_h, best_c = n_hidden, n_candidates
    print(
        f"    -> selected: n_hidden={best_h}  n_cand={best_c}  val_RMSE={best_val:.4g}"
    )
    sel_dict = {"n_hidden": best_h, "n_candidates": best_c}
    return best_beta, best_IW, best_bias, sel_dict, elm_predict


def elm_predict(
    X: np.ndarray, beta: np.ndarray, IW: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    # PAC is a physical power: clip negative values to 0.
    return np.clip(elm_sigmoid(X @ IW.T + bias) @ beta, a_min=0.0, a_max=None)


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    run_elm(
        slug="ols",
        script_name="dr_elm_ols.py",
        train_grid=train_elm_grid,
        extra_cols=["N_params", "n_hidden", "n_candidates"],
    )


if __name__ == "__main__":
    main()
