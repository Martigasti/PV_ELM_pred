"""
Shared utils for the PV_AC Palaiseau forecasting scripts.

Two groups of functions, all used by blend_optimisation, blend_correlation,
dr_ar_ols, dr_elm_ols and dr_elm_ridge:

  * Data / features / predictors: load_30min, sertomat,
    time_features_for_targets, predict_cyclic_persistence, predict_blend.
  * Evaluation: compute_is_day_mask (solar elevation > 0 deg at Palaiseau),
    metrics_on_subset, build_metric_row, split_and_save. These functions
    produce two parallel benchmark tables:
        - `_all`: metrics over all test samples (nights included).
        - `_day`: metrics restricted to the samples where the sun is
                  above the geometric horizon at Palaiseau.
"""
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib  # For day/night detection (sun angle)


# ============================================================================
# DATA LOADING
# ============================================================================
def load_30min(csv_path: Path, cache_path: Path, n_rows: int | None = None) -> np.ndarray:
    if cache_path.exists():
        arr = np.load(cache_path)
    else:
        df = pd.read_csv(csv_path)
        df["datetime"] = pd.to_datetime(
            df["datetime"].astype(str).str.split("+").str[0],
            format="%Y-%m-%d %H:%M:%S",
        )
        df = df.sort_values("datetime").set_index("datetime")
        arr = df["PAC"].resample("30min").mean().to_numpy()
    arr = np.where(np.isnan(arr), 0.0, arr)
    arr = np.clip(arr, a_min=0.0, a_max=None)
    if n_rows is not None:
        arr = arr[:n_rows]
    return arr


# ============================================================================
# SUPERVISED MATRIX
# ============================================================================
def sertomat(x: np.ndarray, LB: int, FH: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert a time series into a supervised matrix (X, y).

    Each row i of X holds the LB past values x[i..i+LB-1],
    and y[i] is the target x[i+LB+FH-1] (i.e. FH steps beyond the
    last input). Used to build the train/test sets of all the
    models (persistence, BLEND, AR-OLS, ELM).
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    n = LB + FH
    m = N + 1 - n
    Y = np.lib.stride_tricks.sliding_window_view(x, n)[:m]
    return Y[:, :LB].copy(), Y[:, n - 1].copy()


# ============================================================================
# CYCLIC TIME FEATURES
# ============================================================================
def time_features_for_targets(
    n_samples: int, LB: int, FH: int, steps_per_day: int = 48
) -> np.ndarray:
    # sin/cos encoding so that 23h<->0h and Dec 31<->Jan 1 are close
    idx_abs = np.arange(n_samples) + LB + FH - 1
    h = (idx_abs % steps_per_day) * (24.0 / steps_per_day)
    j = (idx_abs // steps_per_day) % 365.25
    two_pi = 2.0 * np.pi
    return np.column_stack(
        [
            np.sin(two_pi * h / 24.0),
            np.cos(two_pi * h / 24.0),
            np.sin(two_pi * j / 365.25),
            np.cos(two_pi * j / 365.25),
        ]
    )


# ============================================================================
# PERSISTENCE / BLEND PREDICTORS
# ============================================================================
def predict_cyclic_persistence(
    data: np.ndarray,
    offset_base: int,
    n_test: int,
    T_period: int,
    fallback: np.ndarray,
) -> np.ndarray:
    y_pred = np.empty(n_test)
    for k in range(1, n_test + 1):
        idx_d = offset_base + k - T_period
        if 1 <= idx_d <= len(data):
            y_pred[k - 1] = data[idx_d - 1]
        else:
            y_pred[k - 1] = fallback[k - 1]
    return y_pred


def predict_blend(
    y_pred_P: np.ndarray,
    y_pred_Pc: np.ndarray,
    lam_phase: np.ndarray,
    offset_base: int,
    T_period: int,
) -> np.ndarray:
    """Convex combination lam*P + (1-lam)*Pc, with lam chosen per target phase.

    lam_phase must be pre-clipped to [0, 1] and indexed 0..T_period-1
    where phase = (target_index - 1) % T_period.
    """
    n_test = len(y_pred_P)
    y_pred_BL = np.empty(n_test)
    for k in range(1, n_test + 1):
        idx_d = offset_base + k
        phase_k = (idx_d - 1) % T_period
        lam = lam_phase[phase_k]
        y_pred_BL[k - 1] = lam * y_pred_P[k - 1] + (1.0 - lam) * y_pred_Pc[k - 1]
    return y_pred_BL


# ============================================================================
# DAY/NIGHT MASK (solar elevation > 0 deg at Palaiseau)
# ============================================================================
LAT_PALAISEAU = 48.7128
LON_PALAISEAU = 2.2188

# arr[0] corresponds to the first 30-min slot of the series (2020-08-01 00:00 UTC).
# The mask is cached on first computation.
_DEBUT_SERIE_UTC = "2020-08-01 00:00:00"
_FREQ = "30min"
_MASK_CACHE_NPY = Path(__file__).resolve().parent.parent / "data" / "Palaiseau" / "is_day_mask.npy"


def compute_is_day_mask(
    n_steps: int,
    elevation_threshold_deg: float = 0.0,
    latitude: float = LAT_PALAISEAU,
    longitude: float = LON_PALAISEAU,
    start_utc: str = _DEBUT_SERIE_UTC,
    cache_path: Path = _MASK_CACHE_NPY,
) -> np.ndarray:
    """Boolean array, True where the sun > threshold over the 30-min slot.

    Defaults to Palaiseau (lat/lon/start date) for backward compatibility.
    Pass latitude/longitude/start_utc/cache_path to use another site (e.g.
    Oxelar), with a distinct cache file so masks do not overwrite each other.
    Cached on disk so the pvlib call is paid only once per N.
    """
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.size >= n_steps:
            return cached[:n_steps]

    times = pd.date_range(start_utc, periods=n_steps, freq=_FREQ, tz="UTC")
    sp = pvlib.solarposition.get_solarposition(
        times, latitude=latitude, longitude=longitude
    )
    mask = sp["elevation"].to_numpy() > elevation_threshold_deg
    try:
        np.save(cache_path, mask)
    except OSError:
        pass
    return mask


# ============================================================================
# METRICS
# ============================================================================
_METRIC_KEYS = ["RMSE", "nRMSE", "nMBE", "nMAE", "R2",
                "NICE1", "NICE2", "NICE3", "NICE_Sigma"]


def _Lk(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    return float(np.mean(np.abs(y_true - y_pred) ** k) ** (1.0 / k))


def metrics_on_subset(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    persis_ref: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, ...]:
    """RMSE/nRMSE/nMBE/nMAE/R2/NICE1-3/NICE_Sigma restricted to `mask`.

    NICE_k is computed against the persistence reference recomputed on the
    same subset — using the global reference would mix the populations and
    make the numbers incomparable.
    """
    y_test = np.asarray(y_test).ravel()
    y_pred = np.asarray(y_pred).ravel()
    persis_ref = np.asarray(persis_ref).ravel()
    if mask is not None:
        y_test = y_test[mask]
        y_pred = y_pred[mask]
        persis_ref = persis_ref[mask]

    n = y_test.size
    if n == 0:
        nan = float("nan")
        return nan, nan, nan, nan, nan, nan, nan, nan, nan

    err = y_pred - y_test
    rmse = sqrt(np.mean(err ** 2))
    mean_y = y_test.mean() if y_test.mean() != 0 else float("nan")
    nrmse = rmse / mean_y
    nmbe = np.mean(err) / mean_y
    nmae = np.mean(np.abs(err)) / mean_y
    var_y = np.sum((y_test - y_test.mean()) ** 2)
    r2 = 1.0 - np.sum(err ** 2) / var_y if var_y > 0 else float("nan")

    MAE_P = _Lk(y_test, persis_ref, 1)
    RMSE_P = _Lk(y_test, persis_ref, 2)
    RMCE_P = _Lk(y_test, persis_ref, 3)
    n1 = _Lk(y_test, y_pred, 1) / MAE_P if MAE_P > 0 else float("nan")
    n2 = _Lk(y_test, y_pred, 2) / RMSE_P if RMSE_P > 0 else float("nan")
    n3 = _Lk(y_test, y_pred, 3) / RMCE_P if RMCE_P > 0 else float("nan")
    nS = (n1 + n2 + n3) / 3.0
    return rmse, nrmse, nmbe, nmae, r2, n1, n2, n3, nS


def build_metric_row(
    method: str,
    LB: int,
    FH: int,
    persis_ref: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    mask_day: np.ndarray,
    extra_fields: dict | None = None,
) -> dict:
    """Result row with both metric blocks `_all` and `_day`."""
    m_all = metrics_on_subset(y_test, y_pred, persis_ref, mask=None)
    m_day = metrics_on_subset(y_test, y_pred, persis_ref, mask=mask_day)
    row: dict = dict(Method=method, LB_days=LB / 48, FH_hours=FH * 0.5)
    if extra_fields:
        row.update(extra_fields)
    for k, v in zip(_METRIC_KEYS, m_all):
        row[f"{k}_all"] = v
    for k, v in zip(_METRIC_KEYS, m_day):
        row[f"{k}_day"] = v
    return row


def split_and_save(
    rows: list[dict],
    extra_cols: list[str] | None,
    out_path_all: str,
    out_path_day: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the merged rows into two CSVs with un-suffixed metric columns."""
    # Lazy import to keep utils import-cycle-free (dataset_config imports utils).
    from dataset_config import QUANTITY

    base = ["Method", "quantity", "LB_days", "FH_hours"]
    extras = list(extra_cols or [])

    def project(rows: list[dict], suffix: str) -> pd.DataFrame:
        out = []
        for r in rows:
            d = {c: r.get(c, QUANTITY if c == "quantity" else None)
                 for c in base + extras}
            for k in _METRIC_KEYS:
                d[k] = r.get(f"{k}_{suffix}")
            out.append(d)
        return pd.DataFrame(out, columns=base + extras + _METRIC_KEYS)

    df_all = project(rows, "all").sort_values(
        ["FH_hours", "LB_days", "Method"]
    ).reset_index(drop=True)
    df_day = project(rows, "day").sort_values(
        ["FH_hours", "LB_days", "Method"]
    ).reset_index(drop=True)
    df_all.to_csv(out_path_all, index=False)
    df_day.to_csv(out_path_day, index=False)
    return df_all, df_day
