# Prédiction PV — Palaiseau

Benchmark de modèles de prévision de la production photovoltaïque (PAC)
sur les données 30 min de Palaiseau (août 2020 -- juillet 2025).
On compare des baselines de persistance, deux variantes de BLEND convexe,
un modèle linéaire AR-OLS et deux variantes d'ELM (OLS et Ridge) sur
plusieurs horizons de prédiction.

## Structure du dépôt

```
python_prediction/
├── data/
│   ├── PV_AC_20200801_20250706_Palaiseau.csv   # Données brutes (PAC, pas 15 min)
│   ├── data_30min.npy                          # Cache : série rééchantillonnée à 30 min
│   └── is_day_mask.npy                         # Cache : masque jour/nuit (élévation solaire > 0°)
├── observations.ipynb                          # Notebook d'exploration / figures
├── requirements.txt                            # Dépendances Python
├── scripts/
│   ├── preprocessing.py           # Construit data/data_30min.npy depuis le CSV
│   ├── run_full.py                # Point d'entrée : lance tous les scripts en mode complet
│   ├── utils.py                   # Helpers partagés : données, prédicteurs, métriques (NICE), masque jour
│   ├── blend_optimisation.py      # BLEND avec λ estimé par moindres carrés (solution analytique)
│   ├── blend_correlation.py       # BLEND avec λ via formule de corrélation (article)
│   ├── dr_ar_ols.py               # Baseline linéaire AR-OLS (pseudo-inverse)
│   ├── dr_elm_ols.py              # ELM avec poids de sortie par pseudo-inverse
│   ├── dr_elm_ridge.py            # ELM avec poids de sortie par régression Ridge
│   └── bench_elm.py               # Mesure les temps d'entraînement / inférence de l'ELM
├── results/                       # CSV de résultats et de prédictions
└── figures/                       # Graphiques (prédictions vs vérité, slides, etc.)
```

## Installation

Python 3.10+ recommandé.

```bash
pip install -r requirements.txt
```

Dépendances : `numpy`, `pandas`, `pvlib` (utilisé dans `utils.py` pour
calculer l'élévation solaire et générer le masque jour/nuit).

## Préparation des données

Avant tout, exécutez **`preprocessing.py`** une fois. Il lit le CSV brut,
rééchantillonne en pas de 30 min et écrit `data/data_30min.npy`.
Tous les scripts utilisent ce cache.

```bash
python scripts/preprocessing.py
```

Le masque jour/nuit (`data/is_day_mask.npy`) est généré automatiquement au
premier appel à `compute_is_day_mask` et mis en cache.

## Exécution

### Lancer tous les modèles (mode complet, 2 ans de données)

```bash
python scripts/run_full.py
```

### Lancer un modèle en particulier

```bash
python scripts/run_full.py blend_opt    # blend_optimisation.py
python scripts/run_full.py blend_corr   # blend_correlation.py
python scripts/run_full.py ar           # dr_ar_ols.py
python scripts/run_full.py ols          # dr_elm_ols.py
python scripts/run_full.py ridge        # dr_elm_ridge.py
```

### Mode test (rapide, sur 1000 points)

Lancer un script directement active le smoke test par défaut :

```bash
python scripts/dr_elm_ridge.py
```

`run_full.py` force `SMOKE_TEST=0` (mode complet).

### Benchmark des temps de calcul (ELM)

`bench_elm.py` mesure les temps de `fit` et de `predict` de l'ELM
(variantes OLS et Ridge) avec les meilleurs hyperparamètres figés,
sans grid search.

```bash
python scripts/bench_elm.py
```

## Modèles évalués

| Modèle              | Description                                                                  |
|---------------------|------------------------------------------------------------------------------|
| Persistence_P       | $\hat{y}(t+h) = y(t)$                                                        |
| Persistence_Pcyclic | $\hat{y}(t+h) = y(t+h-24h)$                                                  |
| BLEND_opti          | Mélange convexe $\lambda P + (1-\lambda) P_c$, $\lambda$ appris par moindres carrés (par phase) |
| BLEND_corre         | Mélange convexe avec $\lambda = 0.5 (1 + \rho)$ via corrélation empirique    |
| AR-OLS              | Régression linéaire OLS sur 48 retards + 4 features temporelles cycliques    |
| ELM (OLS)           | ELM, poids de sortie par pseudo-inverse                                      |
| ELM (Ridge)         | ELM, poids de sortie par moindres carrés régularisés ($\lambda$ grid-search) |

Tous les modèles partagent les mêmes entrées (fenêtre $LB=48$ retards +
features sin/cos pour l'heure et le jour de l'année) et les mêmes
horizons : 0,5 h ; 1 h ; 3 h ; 6 h ; 10 h.

## Sorties

Chaque script écrit dans `results/` :

- `Results_<nom>_all.csv` : métriques sur tous les pas de test
- `Results_<nom>_day.csv` : métriques restreintes aux pas du jour (nuit absente), calculé par rapport à quand l'élévation
  solaire dépasse 0° aux coordonnées du site dans Palaiseau (en utilisant la bibliothèque pvlib)
- `Predictions_<nom>.csv` : prédictions point par point pour chaque
  méthode et chaque horizon

### Métriques rapportées

Soit $y_i$ la vérité terrain, $\hat{y}_i$ la prédiction et $N$ le nombre
de pas de test. On note $\bar{y} = \frac{1}{N}\sum_i y_i$.

| Métrique | Formule | Unité | Interprétation |
|---|---|---|---|
| RMSE | $\sqrt{\frac{1}{N}\sum_i (\hat{y}_i - y_i)^2}$ | W | Erreur quadratique moyenne. Pénalise fortement les grosses erreurs. Plus c'est petit, mieux c'est. |
| nRMSE | $\text{RMSE} / \bar{y}$ | — | RMSE normalisé par la moyenne. Permet de comparer entre jeux de données ou horizons d'amplitude différente. |
| nMBE | $\frac{1}{N\bar{y}}\sum_i (\hat{y}_i - y_i)$ | — | Biais moyen normalisé. Positif = sur-estimation systématique, négatif = sous-estimation. Idéal : proche de 0. |
| nMAE | $\frac{1}{N\bar{y}}\sum_i \|\hat{y}_i - y_i\|$ | — | MAE normalisé. Moins sensible aux valeurs extrêmes que la RMSE. |
| $R^2$ | $1 - \frac{\sum_i (\hat{y}_i - y_i)^2}{\sum_i (y_i - \bar{y})^2}$ | — | Variance expliquée. $R^2 = 1$ : prédiction parfaite ; $R^2 = 0$ : équivaut à prédire $\bar{y}$ ; $R^2 < 0$ : pire que la moyenne. |
| $NICE^1$ | $L^1(\hat{y}) / L^1(\hat{y}_P)$ | — | Score skill basé sur MAE. $<1$ : modèle meilleur que la persistance simple ; $>1$ : pire. |
| $NICE^2$ | $L^2(\hat{y}) / L^2(\hat{y}_P)$ | — | Score skill basé sur RMSE. Idem, plus sensible aux grosses erreurs. |
| $NICE^3$ | $L^3(\hat{y}) / L^3(\hat{y}_P)$ | — | Score skill basé sur la norme $L^3$. Encore plus pénalisant pour les erreurs extrêmes. |
| $NICE_\Sigma$ | $(NICE^1 + NICE^2 + NICE^3) / 3$ | — | Moyenne des 3 NICE. Indicateur synthétique de la qualité relative à la persistance. |

Avec $L^k(\hat{y}) = \left(\frac{1}{N}\sum_i \|\hat{y}_i - y_i\|^k\right)^{1/k}$ et
$\hat{y}_P$ la **persistance simple** ($\hat{y}_P(t+h) = y(t)$) recalculée sur
la même sous-population (tous échantillons / jour uniquement).

### Conseils pour une lecture pratique

- **Pour quantifier l'erreur absolue** : regarder `RMSE` (en W) ou `nRMSE` (sans unité, comparable entre horizons).
- **Pour détecter un biais systématique** : `nMBE`. Un blend qui prédit toujours trop bas aura un `nMBE` négatif.
- **Pour comparer à la persistance** : utiliser les `NICE`. C'est le seul moyen de juger si le modèle apporte quelque chose par rapport à la baseline triviale.
  - $NICE^k < 1$ : le modèle bat la persistance sur l'erreur d'ordre $k$.
  - $NICE^k \approx 1$ : pas mieux que persistance.
  - $NICE^k > 1$ : pire que persistance (à éviter).
- **`_all` vs `_day`** : les métriques `_all` incluent les nuits (où $y = 0$ et où la persistance est triviale), ce qui gonfle artificiellement les scores. Les métriques `_day` sont plus représentatives de la difficulté réelle de prédiction PV.
