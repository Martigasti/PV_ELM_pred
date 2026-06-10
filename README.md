# PV Forecasting — Palaiseau

Benchmark of photovoltaic power forecasting models (AC power output)
on 30-minute data from Palaiseau (August 2020 -- August 2022).
Persistence baselines, two variants of convex BLEND, a linear AR-OLS model,
and several ELM variants are compared across multiple forecast horizons.

## Repository structure

```
python_prediction/
├── data/
│   ├── Palaiseau/
│   │   ├── PV_AC_20200801_20250706_Palaiseau.csv   # Raw data (AC power, 15-min steps)
│   │   ├── data_30min.npy                          # Cache: series resampled to 30 min
│   │   └── is_day_mask.npy                         # Cache: day/night mask (solar elevation > 0°)
│   ├── cs/                                         # Oxelar + Signes datasets (raw CSV + 30-min cache + mask)
│   ├── Solete/                                     # SOLETE dataset (DimRed_PAC_Solete.nc + 30-min cache + mask)
│   └── Alice/                                      # Alice Springs dataset (DimRed_PAC_Alice.nc + 30-min cache + mask)
├── requirements.txt                            # Python dependencies
├── scripts/
│   ├── observations.ipynb         # Exploration notebook / figures
│   ├── preprocessing.py           # Builds data/Palaiseau/data_30min.npy from the CSV
│   ├── preprocessing_oxelar.py    # Builds data/cs/oxelar_30min.npy from the Oxelar CSV
│   ├── preprocessing_signes.py    # Builds data/cs/signes_30min.npy from the Signes CSV (same structure as Oxelar)
│   ├── preprocessing_nc.py        # Builds the 30-min cache for the NetCDF datasets (Solete, Alice), driven by DATASET
│   ├── dataset_config.py          # Dataset selection via DATASET env var (Palaiseau/oxelar/signes/solete/alice)
│   ├── run_full.py                # Entry point: runs all scripts in full mode
│   ├── utils.py                   # Shared helpers: data loading, predictors, NICE metrics, day mask
│   ├── elm_common.py              # Factored ELM core: config, elm_sigmoid, ridge_solve, generic run_elm runner
│   ├── blend_optimisation.py      # BLEND with λ estimated by least squares (analytical solution)
│   ├── blend_correlation.py       # BLEND with λ via correlation formula (paper)
│   ├── dr_ar_ols.py               # Linear AR-OLS baseline (pseudo-inverse)
│   ├── dr_elm_ols.py              # ELM with output weights via pseudo-inverse
│   ├── dr_elm_ridge.py            # ELM with output weights via Ridge regression
│   ├── dr_elm_robust_risk.py      # ELM Robust Risk (Ridge with lam = ε², bounded-uncertainty interpretation)
│   ├── dr_elm_tikhonov.py         # Anisotropic Tikhonov ELM (penalty proportional to per-neuron energy)
│   ├── dr_elm_box_cox.py          # Ridge ELM on Box-Cox-transformed target (shift c=1 W)
│   ├── dr_elm_box_cox_rolling.py  # Box-Cox ELM, rolling refit each step over fixed windows {1m,6m,1y} (off run_full)
│   ├── dr_elm_mae.py              # ELM-MAE (L1 loss, strict 2-pass, smoothing √(r²+δ²))
│   ├── dr_elm_log_mse.py          # ELM-Log-MSE (log-quadratic loss, 2-pass, shift c grid)
│   ├── dr_elm_huber.py            # ELM-Huber (Huber loss, 2-pass, adaptive δ via MAD)
│   ├── dr_elm_l3.py               # ELM-L3 (cubic norm, 2-pass with smooth |r|³ ≈ r²·√(r²+δ²))
│   ├── dr_elm_corr.py             # Corr-ELM (temporal correlation C_ij=exp(-|Δt|/τ), closed-form; env CORR_BANDED=0/1)
│   ├── dr_elm_elastic_net.py      # ELM Elastic Net (L1+L2, 2-pass linearised, joint grid (λ₁, λ₂))
│   ├── dr_elm_m_estimator.py      # ELM M-Estimator (Welsch redescending weights, 2-pass, c via MAD)
│   ├── dr_elm_lp.py               # ELM-Lp (general L^p norm, p in (1,2)), 2-pass, joint grid (λ, p)
│   ├── dr_elm_glm.py              # ELM-GLM linearised Fisher (Gamma + log link), 2-pass, joint grid (λ, c)
│   ├── timegpt.py                 # TimeGPT (Nixtla foundation model, zero-shot; needs NIXTLA_API_KEY)
│   ├── score_neural_prophet.py    # Score external NeuralProphet forecasts with benchmark metrics (NICE)
│   ├── dr_elm_znocyclic_ols.py    # Ablation: ELM OLS without the 4 cyclic features (FH=1)
│   ├── dr_elm_znocyclic_ridge.py  # Ablation: ELM Ridge without the 4 cyclic features (FH=1)
│   └── dr_elm_znocyclic_rr.py     # Ablation: ELM Robust Risk without the 4 cyclic features (FH=1)
├── results/                       # Result and prediction CSV files
└── figures/                       # Plots (predictions vs ground truth, slides, etc.)
```

## Installation

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `pvlib` (used in `utils.py` to compute solar
elevation and generate the day/night mask).

## Data preparation

First, run **`preprocessing.py`** once. It reads the raw CSV, resamples to
30-minute steps, and writes `data/Palaiseau/data_30min.npy`. All scripts use this cache.

```bash
python scripts/preprocessing.py
```

The day/night mask (`data/Palaiseau/is_day_mask.npy`) is generated automatically on the
first call to `compute_is_day_mask` and cached to disk.

### Other datasets

The `DATASET` env var selects the series (default `Palaiseau`; also `oxelar`,
`signes`, `solete`, `alice`), via `dataset_config.py`. Build each dataset's cache
once with its preprocessing, then everything else is driven by `DATASET`:

```bash
DATASET=oxelar python scripts/preprocessing_oxelar.py   # Oxelar CSV
DATASET=signes python scripts/preprocessing_signes.py   # Signes CSV
DATASET=solete python scripts/preprocessing_nc.py       # SOLETE NetCDF
DATASET=alice  python scripts/preprocessing_nc.py       # Alice NetCDF
```

NetCDF datasets (Solete, Alice) require `xarray` + `netCDF4` and have no CSV
fallback, so their cache must be built first.

## Running the models

### Run all models (full mode)

```bash
python scripts/run_full.py                       # default dataset (Palaiseau)
DATASET=solete python scripts/run_full.py   # results land in results/Solete/
```

### Run a specific model

Pass one or more target names to `run_full.py` (run with no argument to run all):

```bash
python scripts/run_full.py blend_opt      # blend_optimisation.py   (BLEND, λ by OLS per phase)
python scripts/run_full.py blend_corr     # blend_correlation.py    (BLEND, λ via correlation)
python scripts/run_full.py ar             # dr_ar_ols.py            (linear AR-OLS baseline)
python scripts/run_full.py ols            # dr_elm_ols.py           (ELM, pseudo-inverse)
python scripts/run_full.py ridge          # dr_elm_ridge.py         (ELM, Ridge)
python scripts/run_full.py rr             # dr_elm_robust_risk.py   (ELM Robust Risk, λ = ε²)
python scripts/run_full.py tikhonov       # dr_elm_tikhonov.py      (anisotropic Tikhonov)
python scripts/run_full.py box_cox        # dr_elm_box_cox.py       (Ridge on Box-Cox target)
python scripts/run_full.py mae            # dr_elm_mae.py           (L1 loss, 2-pass)
python scripts/run_full.py log_mse        # dr_elm_log_mse.py       (log-MSE loss, 2-pass)
python scripts/run_full.py huber          # dr_elm_huber.py         (Huber loss, 2-pass)
python scripts/run_full.py l3             # dr_elm_l3.py            (cubic norm, 2-pass)
python scripts/run_full.py corr           # dr_elm_corr.py          (Corr-ELM, closed-form)
python scripts/run_full.py elastic_net    # dr_elm_elastic_net.py   (L1+L2, 2-pass)
python scripts/run_full.py m_estimator    # dr_elm_m_estimator.py   (Welsch M-estimator, 2-pass)
python scripts/run_full.py lp             # dr_elm_lp.py            (general L^p norm, 2-pass)
python scripts/run_full.py glm            # dr_elm_glm.py           (GLM Gamma+log, 2-pass Fisher)
python scripts/run_full.py timegpt        # timegpt.py              (TimeGPT, zero-shot)
python scripts/run_full.py ols_nocyclic   # dr_elm_znocyclic_ols.py    (ablation, no cyclic features)
python scripts/run_full.py ridge_nocyclic # dr_elm_znocyclic_ridge.py  (ablation, no cyclic features)
python scripts/run_full.py rr_nocyclic    # dr_elm_znocyclic_rr.py     (ablation, no cyclic features)
```

### Smoke-test mode (fast, 1000 points)

Running a script directly activates the smoke test by default:

```bash
python scripts/dr_elm_ridge.py
```

`run_full.py` forces `SMOKE_TEST=0` (full mode).

#### Corr-ELM banded approximation

`dr_elm_corr.py` builds an $N \times N$ correlation matrix, which becomes the
bottleneck for $N \approx 17\,000$ train samples. To keep it manageable, a
banded approximation is used by default: entries beyond $5\tau / \Delta t$
steps from the diagonal are set to zero (residual weight $e^{-5} \approx 0.7\%$),
bringing the cost down from $O(N^2 \cdot N_h)$ to $O(N \cdot K \cdot N_h)$ which is
roughly 250× faster at $\tau = 6$ h.

```bash
CORR_BANDED=1 python scripts/dr_elm_corr.py   # banded (default, fast)
CORR_BANDED=0 python scripts/dr_elm_corr.py   # dense (slow)
```

The same applies under `run_full.py`:

```bash
CORR_BANDED=0 python scripts/run_full.py corr   # full mode + dense C
```

## Models evaluated

| Model               | Description                                                                  |
|---------------------|------------------------------------------------------------------------------|
| Persistence_P       | $\hat{y}(t+h) = y(t)$                                                        |
| Persistence_Pcyclic | $\hat{y}(t+h) = y(t+h-24h)$                                                  |
| BLEND_opti          | Convex blend $\lambda P + (1-\lambda) P_c$, $\lambda$ learned by least squares (per phase) |
| BLEND_corre         | Convex blend with $\lambda = 0.5 (1 + \rho)$ via empirical correlation       |
| AR-OLS              | OLS linear regression on 48 lags + 4 cyclic temporal features                |
| ELM (OLS)           | ELM, output weights via pseudo-inverse                                       |
| ELM (Ridge)         | ELM, output weights via regularised least squares ($\lambda$ grid-search)    |
| ELM (Robust Risk)   | Like Ridge but $\lambda = \varepsilon^2$ (bounded uncertainty on $H$)        |
| ELM (Tikhonov)      | Anisotropic Ridge: penalty $\propto$ energy of each hidden neuron            |
| ELM (Box-Cox)       | Ridge on Box-Cox-transformed target $P_{AC}$; two hyperparams ($\lambda_r$, $\lambda_{bc}$) |
| ELM (MAE)           | $L^1$ cost $\|H\beta - y\|_1$, strict 2-pass: Pass 1 Ridge init, Pass 2 a single weighted solve with smoothing $\sqrt{r^2 + \delta^2}$; joint grid $(\lambda, \delta)$ |
| ELM (Log-MSE)       | Cost $\|\log(H\beta + c) - \log(y + c)\|^2 + \lambda\|\beta\|^2$ with $\beta$ in **log space** (deviation from PDF §A.2): closed-form Ridge on $z = \log(y + c)$, $\hat y = \exp(H\beta) - c$. The PDF's literal 2-pass ($\beta$ in original space) collapses to $\sim 0$ on PV data. Grid $(\lambda, c)$ |
| ELM (Huber)         | Huber cost $\rho_\delta$ (quadratic near 0, linear beyond), strict 2-pass: Pass 1 Ridge init, Pass 2 a single weighted solve with $W = \min(1, \delta / \lvert r^{(0)}\rvert)$, $\delta = 1.345 \cdot \mathrm{MAD}(r^{(0)})$ |
| ELM (L3)            | Cubic-norm cost $\sum_i \lvert r_i\rvert^3$ (upweights large residuals, opposite of MAE/Huber), strict 2-pass: Pass 1 Ridge init, Pass 2 a single weighted solve with $W = \mathrm{diag}\big(\sqrt{(r^{(0)})^2 + \delta^2}\big)$; joint grid $(\lambda, \delta)$ |
| ELM (Corr)          | GLS with stationary temporal correlation $C_{ij} = \exp(-\lvert t_i - t_j\rvert / \tau)$ on residuals: $\beta = (H^\top C H + \sigma^2 I)^{-1} H^\top C Y$. Closed-form, joint grid $(\sigma^2, \tau)$ |
| ELM (Elastic Net)   | Cost $\lVert H\beta - y\rVert_2^2 + \lambda_2\lVert\beta\rVert_2^2 + \lambda_1\lVert\beta\rVert_1$, strict 2-pass: Pass 1 Ridge($\lambda_2$) init, Pass 2 a single weighted solve linearising the $L^1$ term with $W_{EN} = \mathrm{diag}(1 / (\lvert\beta^{(0)}_j\rvert + \varepsilon))$; joint grid $(\lambda_1, \lambda_2)$ |
| ELM (M-Estimator)   | Welsch redescending cost $\rho(r) = \tfrac{c^2}{2}(1 - e^{-(r/c)^2})$, strict 2-pass: Pass 1 Ridge init, Pass 2 a single weighted solve with $W = \mathrm{diag}(e^{-(r^{(0)}/c)^2})$, $c = 2.985 \cdot \mathrm{MAD}(r^{(0)}) / 0.6745$; grid over $\lambda$ |
| ELM ($L_p$)         | General $L^p$ norm cost $\sum_i \lvert r_i\rvert^p + \lambda\lVert\beta\rVert_2^2$ with $p \in (1,2)$, strict 2-pass: Pass 1 Ridge init, Pass 2 a single weighted solve with $W_p = \mathrm{diag}(\lvert r^{(0)}\rvert^{p-2})$ (floor at $\varepsilon$ to avoid division by zero); joint grid $(\lambda, p)$ |
| ELM (GLM)           | Linearised GLM (Fisher scoring, one step) with Gamma + log link: $g(\mu) = \log\mu$, $\mathrm{Var}(Y) \propto \mu^2$. Strict 2-pass: Pass 1 Ridge on $y$, Pass 2 a single weighted solve with adjusted variable $z = \log(\hat\mu) + (y + c - \hat\mu)/\hat\mu$ and weights $W_\eta = \mathrm{diag}(\hat\mu^2)$. $\beta$ lives in log-space, $\hat y = \exp(H\beta) - c$. Joint grid $(\lambda, c)$ |
| TimeGPT             | Nixtla foundation model, zero-shot inference (no training, no exogenous features). 1 API call per window covering $h_{\max}$, subsampled at `STRIDE` to fit free-tier quotas. Requires `pip install nixtla` and `NIXTLA_API_KEY` |
| NeuralProphet       | External model (forecasts supplied by a teammate), scored a posteriori with the benchmark metrics (NICE) via `score_neural_prophet.py`. Two variants: with / without daily seasonality |

All models share the same inputs (lookback window of $LB=48$ lags +
sin/cos features for the hour of day and day of year) and the same
forecast horizons: 0.5 h; 1 h; 3 h; 6 h; 10 h.

## Outputs

Each script writes to `results/`:

- `Results_<name>_all.csv`: metrics over all test steps
- `Results_<name>_day.csv`: metrics restricted to daytime steps (nights excluded), computed based on when solar elevation exceeds 0° at the Palaiseau site coordinates (using the pvlib library)
- `Predictions_<name>.csv`: point-by-point predictions for each method and horizon

### Reported metrics

Let $y_i$ be the ground truth, $\hat{y}_i$ the prediction, and $N$ the number
of test steps. Denote $\bar{y} = \frac{1}{N}\sum_i y_i$.

| Metric | Formula | Unit | Interpretation |
|---|---|---|---|
| RMSE | $\sqrt{\frac{1}{N}\sum_i (\hat{y}_i - y_i)^2}$ | W | Root mean squared error. Heavily penalises large errors. Lower is better. |
| nRMSE | $\text{RMSE} / \bar{y}$ | — | RMSE normalised by the mean. Enables comparison across datasets or horizons with different scales. |
| nMBE | $\frac{1}{N\bar{y}}\sum_i (\hat{y}_i - y_i)$ | — | Normalised mean bias error. Positive = systematic over-estimation, negative = under-estimation. Ideally close to 0. |
| nMAE | $\frac{1}{N\bar{y}}\sum_i \|\hat{y}_i - y_i\|$ | — | Normalised MAE. Less sensitive to outliers than RMSE. |
| $R^2$ | $1 - \frac{\sum_i (\hat{y}_i - y_i)^2}{\sum_i (y_i - \bar{y})^2}$ | — | Explained variance. $R^2 = 1$: perfect forecast; $R^2 = 0$: equivalent to predicting $\bar{y}$; $R^2 < 0$: worse than the mean. |
| $NICE^1$ | $L^1(\hat{y}) / L^1(\hat{y}_P)$ | — | Skill score based on MAE. $<1$: model beats simple persistence; $>1$: worse. |
| $NICE^2$ | $L^2(\hat{y}) / L^2(\hat{y}_P)$ | — | Skill score based on RMSE. Same, more sensitive to large errors. |
| $NICE^3$ | $L^3(\hat{y}) / L^3(\hat{y}_P)$ | — | Skill score based on the $L^3$ norm. Even more penalising for extreme errors. |
| $NICE_\Sigma$ | $(NICE^1 + NICE^2 + NICE^3) / 3$ | — | Average of the 3 NICE scores. Synthetic indicator of quality relative to persistence. |

Where $L^k(\hat{y}) = \left(\frac{1}{N}\sum_i \|\hat{y}_i - y_i\|^k\right)^{1/k}$ and
$\hat{y}_P$ is **simple persistence** ($\hat{y}_P(t+h) = y(t)$) recomputed on
the same sub-population (all samples / daytime only).

### Practical reading guide

- **To quantify absolute error**: look at `RMSE` (in W) or `nRMSE` (dimensionless, comparable across horizons).
- **To detect systematic bias**: `nMBE`. A blend that consistently predicts too low will have a negative `nMBE`.
- **To compare against persistence**: use the `NICE` scores. They are the only way to judge whether a model adds value over the trivial baseline.
  - $NICE^k < 1$: the model beats persistence on the order-$k$ error.
  - $NICE^k \approx 1$: no better than persistence.
  - $NICE^k > 1$: worse than persistence (to be avoided).
- **`_all` vs `_day`**: `_all` metrics include nights (where $y = 0$ and persistence is trivial), which artificially inflates scores. `_day` metrics are more representative of the true difficulty of PV forecasting.
