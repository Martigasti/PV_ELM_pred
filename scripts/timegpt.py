"""
PV Palaiseau forecasting with TimeGPT (Nixtla) — zero-shot foundation model.

Unlike the other bench scripts (ELM, AR-OLS, BLEND) which are trained
on the train set, TimeGPT is called zero-shot: it is given a history
window (LB steps) and predicts the next h steps. No fine-tuning.

Call strategy:
    - A single call per window, h = max(FH_list) steps (covers all horizons).
    - Subsampling of the test set by STRIDE to stay within reasonable
      quotas (Nixtla free tier: ~1000 calls/month in practice).
    - No exogenous variables: TimeGPT extracts the seasonality on its own (in the
      spirit of a foundation model, comparable to the zero-shot paradigm).

Baselines reported on the SAME subsampled subset to make the
NICE metrics comparable: Persistence P, Persistence P_cyclic, BLEND_opti.

Prerequisites:
    - pip install nixtla
    - export NIXTLA_API_KEY=<your key obtained from https://dashboard.nixtla.io/>
"""
import os
import time
from math import floor

import numpy as np
import pandas as pd

from blend_optimisation import fit_blend_lambda_per_phase
from dataset_config import CACHE_NPY, CSV_FILE, NDATA_FULL, RESULTS_DIR, day_mask
from utils import (
    apply_night_mask,
    build_metric_row,
    load_30min,
    predict_blend,
    predict_cyclic_persistence,
    sertomat,
    split_and_save,
)


# ============================================================================
# CONFIG
# ============================================================================
SMOKE_TEST = os.environ.get("SMOKE_TEST", "1") == "1"

if SMOKE_TEST:
    print("*** SMOKE TEST MODE ***")
    Ndata    = 2000
    LB_list  = [48]
    FH_list  = [1, 12]
    ratio    = 0.50
    T_period = 48
    STRIDE   = 50          # 1 window out of 50
else:
    print("*** FULL MODE ***")
    Ndata    = NDATA_FULL
    LB_list  = [48]
    FH_list  = [1, 2, 6, 12, 20]
    ratio    = 0.50
    T_period = 48
    STRIDE   = 5          # 1 window out of 20 (~875 calls for 17,500 test pts)

# Start date of the dataset (first point at 00:00 UTC per the file name).
START_TIMESTAMP = "2020-08-01 00:00:00"
FREQ            = "30min"
TIMEGPT_MODEL   = "timegpt-1"   # alternatives: "timegpt-1-long-horizon"

OUT_FILE_ALL = RESULTS_DIR / "Results_TimeGPT_NICE_simple_all.csv"
OUT_FILE_DAY = RESULTS_DIR / "Results_TimeGPT_NICE_simple_day.csv"
PRED_FILE    = RESULTS_DIR / "Predictions_TimeGPT_NICE_simple.csv"


# ============================================================================
# TIMEGPT CLIENT
# ============================================================================
def get_timegpt_client():
    """Create the Nixtla client. Fail clearly if the lib or key are missing."""
    try:
        from nixtla import NixtlaClient
    except ImportError as exc:
        raise SystemExit(
            "The 'nixtla' SDK is not installed. Run: pip install nixtla"
        ) from exc

    api_key = os.environ.get("NIXTLA_API_KEY")
    if not api_key:
        raise SystemExit(
            "Environment variable NIXTLA_API_KEY not set. "
            "Get a key from https://dashboard.nixtla.io/ then: "
            "export NIXTLA_API_KEY=<your_key>"
        )
    return NixtlaClient(api_key=api_key)


# Per-call retry on transient network errors (DNS drop, 408/timeout, conn reset).
# WSL2 connections to the Nixtla API drop intermittently; without this a single
# blip after ~1h of calls kills the whole run.
_MAX_CALL_RETRIES = 6
_BACKOFF_BASE_S = 2.0          # 2,4,8,16,32,64 s
_CKPT_EVERY = 25               # flush checkpoint to disk every N windows


def _ckpt_paths(sig: str):
    """Checkpoint files keyed by a run signature (invalidated if params change)."""
    stem = RESULTS_DIR / f".timegpt_ckpt_{sig}"
    return stem.with_suffix(".preds.npy"), stem.with_suffix(".done.npy")


def _forecast_one(client, hist, t0, h):
    """One TimeGPT call, retried on transient network errors with backoff.

    Raises the last exception if all retries are exhausted (caller checkpoints
    and exits cleanly so a re-run resumes)."""
    ts = pd.date_range(start=t0, periods=len(hist), freq=FREQ)
    df = pd.DataFrame({"ds": ts, "y": hist, "unique_id": "PAC"})
    last_exc = None
    for attempt in range(_MAX_CALL_RETRIES):
        try:
            fcst = client.forecast(df=df, h=h, freq=FREQ, model=TIMEGPT_MODEL)
            return fcst["TimeGPT"].to_numpy()
        except Exception as exc:  # network/HTTP errors from httpx/nixtla
            last_exc = exc
            wait = _BACKOFF_BASE_S * (2 ** attempt)
            print(f"      [retry {attempt+1}/{_MAX_CALL_RETRIES}] {type(exc).__name__}: "
                  f"{exc} — waiting {wait:.0f}s")
            time.sleep(wait)
    raise last_exc


def timegpt_forecast_batch(
    client, histories: list[np.ndarray], start_ts_list: list[pd.Timestamp], h: int,
    sig: str = "default",
) -> np.ndarray:
    """Predict h steps for each history window. Returns (n_windows, h).

    Each forecast call is independent: a single series is passed per window
    with its absolute timestamp (so that TimeGPT picks up the correct seasonal
    features: hour of the day, day of the year).

    Resumable: predictions are checkpointed to disk as they complete. If the
    network drops and per-call retries are exhausted, the checkpoint is saved
    and the script exits cleanly — re-running resumes from the last saved
    window (no lost API calls). `sig` keys the checkpoint to the run params.
    """
    n = len(histories)
    preds_path, done_path = _ckpt_paths(sig)

    if preds_path.exists() and done_path.exists():
        out = np.load(preds_path)
        done = np.load(done_path)
        if out.shape == (n, h) and done.shape == (n,):
            print(f"    TimeGPT : resuming from checkpoint "
                  f"({int(done.sum())}/{n} windows already done)")
        else:  # stale checkpoint (params changed) → start fresh
            out, done = np.full((n, h), np.nan), np.zeros(n, dtype=bool)
    else:
        out, done = np.full((n, h), np.nan), np.zeros(n, dtype=bool)

    def _flush():
        np.save(preds_path, out)
        np.save(done_path, done)

    for i, (hist, t0) in enumerate(zip(histories, start_ts_list)):
        if done[i]:
            continue
        try:
            out[i] = _forecast_one(client, hist, t0, h)
        except Exception as exc:
            _flush()
            done_n = int(done.sum())
            raise SystemExit(
                f"\nNetwork failure at window {i+1}/{n} after retries: {exc}\n"
                f"Checkpoint saved ({done_n}/{n} done). Re-run the SAME command "
                f"to resume from window {done_n+1}."
            ) from exc
        done[i] = True
        if (i + 1) % _CKPT_EVERY == 0:
            _flush()
        if (i + 1) % 50 == 0:
            print(f"    TimeGPT : {i+1}/{n} windows processed")

    _flush()
    return np.clip(out, a_min=0.0, a_max=None)


# ============================================================================
# EXECUTION
# ============================================================================
def run_one(
    data: np.ndarray,
    is_day_full: np.ndarray,
    start_dt: pd.Timestamp,
    client,
    LB: int,
    timegpt_preds: np.ndarray,
    timegpt_idx: np.ndarray,
    h_max: int,
    FH: int,
) -> tuple[list[dict], list[pd.DataFrame]]:
    """Build the metric rows for a given FH.

    timegpt_preds : (n_windows, h_max) — predictions already computed for h_max.
    timegpt_idx   : (n_windows,) — start index of each target window on the
                                   absolute axis (= idx_split + LB + ... ).
    """
    print(f"\n=== LB={LB} ({LB/48:g}d) | FH={FH} ({FH*0.5:.1f}h) ===")

    PVin, PVout = sertomat(data, LB, FH)
    idx_split = floor(ratio * PVin.shape[0])
    Persis_simple_test = PVin[idx_split:, -1]
    offset_base = idx_split + LB + FH - 1
    n_test = len(PVout) - idx_split
    mask_day_test = is_day_full[offset_base : offset_base + n_test]

    # Subsampling: keep only the TimeGPT indices.
    # timegpt_idx points to the absolute index data[k] which is the 1st prediction
    # of the window. The prediction at step FH is timegpt_preds[:, FH-1] and
    # corresponds to data[timegpt_idx + FH - 1].
    # For the baselines we must fetch y_true and persis at the same indices.
    abs_pred_idx = timegpt_idx + FH - 1                       # absolute indices in `data`
    local_idx    = abs_pred_idx - offset_base                  # indices in the test set
    valid        = (local_idx >= 0) & (local_idx < n_test)
    local_idx    = local_idx[valid]
    y_pred_tg    = timegpt_preds[valid, FH - 1]

    y_test_sub        = PVout[idx_split:][local_idx]
    persis_simple_sub = Persis_simple_test[local_idx]
    mask_day_sub      = mask_day_test[local_idx]

    # Post-processing: force predictions to 0 at night (no-op for non-solar sets).
    y_pred_tg = apply_night_mask(y_pred_tg, mask_day_sub)

    # ---- Persistences and BLEND: computed on the whole test set then subsampled.
    y_pred_P  = apply_night_mask(Persis_simple_test[local_idx], mask_day_sub)
    y_pred_Pc_full = predict_cyclic_persistence(
        data, offset_base, n_test, T_period, fallback=Persis_simple_test
    )
    y_pred_Pc = apply_night_mask(y_pred_Pc_full[local_idx], mask_day_sub)

    data_tr_raw = data[: idx_split + LB + FH]
    lam_phase = fit_blend_lambda_per_phase(data_tr_raw, FH, T_period)
    y_pred_BL_full = predict_blend(
        Persis_simple_test, y_pred_Pc_full, lam_phase, offset_base, T_period
    )
    y_pred_BL = apply_night_mask(y_pred_BL_full[local_idx], mask_day_sub)

    rows: list[dict] = []
    pred_rows: list[pd.DataFrame] = []

    def log_predictions(method: str, y_pred: np.ndarray) -> None:
        pred_rows.append(
            pd.DataFrame(
                {
                    "Method": method,
                    "LB_days": LB / 48,
                    "FH_hours": FH * 0.5,
                    "t_index": local_idx,
                    "y_true": y_test_sub,
                    "y_pred": y_pred,
                }
            )
        )

    for name, y_pred in [
        ("Persistence_P",       y_pred_P),
        ("Persistence_Pcyclic", y_pred_Pc),
        ("BLEND_opti",          y_pred_BL),
        ("TimeGPT",             y_pred_tg),
    ]:
        rows.append(
            build_metric_row(
                name, LB, FH, persis_simple_sub, y_test_sub, y_pred,
                mask_day_sub, extra_fields={"N_params": 0, "n_test_sub": len(y_test_sub)},
            )
        )
        log_predictions(name, y_pred)

    print(f"    [TimeGPT] {len(y_pred_tg)} predictions evaluated (stride={STRIDE})")
    return rows, pred_rows


def main() -> None:
    data = load_30min(CSV_FILE, CACHE_NPY, n_rows=Ndata)
    print(f"Data: {len(data)} points ({len(data)/48/365.25:.2f} years)")
    start_dt = pd.Timestamp(START_TIMESTAMP)

    is_day_full = day_mask(len(data))
    print(
        f"Day mask  : {is_day_full.sum()}/{len(is_day_full)} steps "
        f"({is_day_full.mean()*100:.1f}% day)"
    )

    client = get_timegpt_client()

    # ------- TimeGPT window preparation (computed ONCE for all FH).
    LB = LB_list[0]
    h_max = max(FH_list)
    n_total = len(data)
    PVin, _ = sertomat(data, LB, h_max)
    idx_split = floor(ratio * PVin.shape[0])
    # For each test window: history = data[k-LB : k], target = data[k : k+h_max]
    # k ranges over offset_base_for_h1 = idx_split + LB  ...  n_total - h_max
    k_min = idx_split + LB
    k_max = n_total - h_max
    k_values = np.arange(k_min, k_max + 1, STRIDE)
    print(f"TimeGPT : {len(k_values)} windows to predict (h={h_max}, stride={STRIDE})")

    histories = [data[k - LB : k] for k in k_values]
    start_ts_list = [start_dt + pd.Timedelta(minutes=30 * (k - LB)) for k in k_values]
    # Signature keys the resume-checkpoint to the run params: changing Ndata,
    # stride, LB, horizon or window count starts a fresh checkpoint.
    sig = f"N{n_total}_s{STRIDE}_LB{LB}_h{h_max}_w{len(k_values)}"
    timegpt_preds = timegpt_forecast_batch(
        client, histories, start_ts_list, h_max, sig=sig,
    )
    # timegpt_idx[j] = absolute index of the 1st prediction (= k_values[j])
    timegpt_idx = k_values.astype(int)

    all_rows: list[dict] = []
    all_pred_rows: list[pd.DataFrame] = []
    for FH in FH_list:
        rows, pred_rows = run_one(
            data, is_day_full, start_dt, client,
            LB, timegpt_preds, timegpt_idx, h_max, FH,
        )
        all_rows.extend(rows)
        all_pred_rows.extend(pred_rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_all, df_day = split_and_save(
        all_rows,
        extra_cols=["N_params", "n_test_sub"],
        out_path_all=str(OUT_FILE_ALL),
        out_path_day=str(OUT_FILE_DAY),
    )
    print("\n===== RESULTS (all samples) =====")
    print(df_all.to_string())
    print("\n===== RESULTS (day only) =====")
    print(df_day.to_string())
    print(f"\nResults _all : {OUT_FILE_ALL}")
    print(f"Results _day : {OUT_FILE_DAY}")

    predictions = pd.concat(all_pred_rows, ignore_index=True)
    predictions.to_csv(PRED_FILE, index=False)
    print(f"Predictions saved: {PRED_FILE}")

    # Results are durably saved → drop the resume-checkpoint.
    for p in _ckpt_paths(sig):
        p.unlink(missing_ok=True)


if __name__ == "__main__":
    _t0 = time.time()
    main()
    _dt = time.time() - _t0
    print(f"\n[chrono] timegpt.py : {_dt:.2f} s  ({_dt/60:.2f} min)")
