# LeBron Points Prediction

A machine-learning pipeline that predicts LeBron James's points scored in the next game, trained on eight seasons (2018–2026) of NBA box scores combined with behavioral and game-context features.

---

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env        # add SPORTS_API_KEY if you have one (optional)

python main.py run-all      # fetch data → build features → train → evaluate → predict
```

Individual steps:

```bash
python main.py fetch-data       # pull NBA game logs and team stats
python main.py build-features   # engineer all features
python main.py train            # walk-forward cross-validation + model selection
python main.py evaluate         # holdout metrics, charts, SHAP analysis
python main.py predict-next     # predict next game points
```

---

## Prediction Formula

The model predicts:

```
ŷ = f(X_rolling, X_efficiency, X_opponent, X_fatigue, X_vegas, X_pt, X_pressure)
```

Where `f` is the best-performing model selected by walk-forward cross-validation across:
Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, XGBoost, LightGBM.

---

## Feature Groups

### 1. Rolling Window Statistics

For each stat `s ∈ {PTS, MIN, FGA, FG3A, FTA, FG_PCT, TS_PCT, USG_PROXY, ...}` and window `w ∈ {3, 5, 10, 20}`:

```
ROLL{w}_{s}[t] = mean( s[t-w : t-1] )   # shifted by 1 — no data leakage
```

Volatility (coefficient of variation):
```
ROLL{w}_PTS_VOLATILITY[t] = ROLL{w}_PTS_STD[t] / ROLL{w}_PTS[t]
```

Z-score within window:
```
ROLL{w}_PTS_ZSCORE[t] = (PTS[t-1] - μ_w) / σ_w
```

### 2. Exponentially-Weighted Moving Average

For span `α ∈ {3, 5, 10}` and `col ∈ {PTS, MIN, USG_PROXY, FGA, TS_PCT}`:

```
EWMA{α}_{col}[t] = (1-λ) · col[t-1] + λ · EWMA{α}_{col}[t-1]
  where λ = 1 - 2/(α+1)
```

### 3. Efficiency Stats

True Shooting Percentage:
```
TS% = PTS / (2 × (FGA + 0.44 × FTA))
```

Effective Field Goal Percentage:
```
eFG% = (FGM + 0.5 × FG3M) / FGA
```

Usage Rate Proxy:
```
USG_PROXY = (FGA + 0.44×FTA + TOV) × 100 / (MIN × 5)
```

### 4. Vegas / Market Features

Implied team total from spread and over/under:
```
IMPLIED_TEAM_TOTAL = (O/U / 2) ± (|spread| / 2)
  + if Lakers are favored (spread < 0), − if underdogs
```

Implied win probability from American moneyline:
```
              100 / (ML + 100)        if ML > 0  (underdog)
IMPLIED_P =
              |ML| / (|ML| + 100)     if ML < 0  (favorite)
```

### 5. Interaction Terms

```
OPP_DEF_X_USG   = OPP_DEF_RATING × ROLL5_USG_PROXY
PACE_X_MIN      = OPP_PACE × LAST_GAME_MIN
SPREAD_X_MIN    = |LAKERS_SPREAD| × LAST_GAME_MIN
TEAM_TOTAL_X_USG= IMPLIED_TEAM_TOTAL × ROLL5_USG_PROXY
HOME_X_OPP_DEF  = HOME_GAME × OPP_DEF_RATING
FATIGUE_X_B2B   = BACK_TO_BACK × AVG_MIN_L5
```

---

## Prospect Theory Index

Based on Kahneman & Tversky (1979): decision-makers evaluate outcomes relative to a reference point and weight losses more heavily than equivalent gains. Applied here, LeBron's scoring effort may shift based on how recent performance compares to expectations.

### Reference Points

```
SEASON_AVG_PTS[t]  = expanding mean of PTS[1 : t-1] within season
ROLL10_PTS_REF[t]  = mean( PTS[t-10 : t-1] )
PREV_PROP_LINE[t]  = Vegas player prop from game t-1 (if available)
```

### Gains and Losses

```
PREV_GAME_PTS_GAP[t]     = PTS[t-1] − ROLL10_PTS_REF[t]

SCORING_DISAPPOINTMENT[t] = max(0,  ROLL10_PTS_REF[t] − PTS[t-1])   # below expectation
SCORING_OVERPERFORMANCE[t]= max(0,  PTS[t-1] − ROLL10_PTS_REF[t])   # above expectation
```

Z-scored across all games:
```
SCORING_DISAPPOINTMENT_Z[t] = (SCORING_DISAPPOINTMENT[t] − μ_D) / σ_D
```

Loss aversion amplification (λ = 2.25, per Kahneman & Tversky):
```
LOSS_AVERSION_SCORE[t] = SCORING_DISAPPOINTMENT_Z[t] × 2.25
```

### Shot Aggressiveness After Disappointment

```
SHOT_AGGRESSIVENESS_CHANGE[t] = ROLL3_FGA[t] − ROLL10_FGA[t]
USG_INCREASE_AFTER_LOSS[t]    = COMING_OFF_LOSS[t] × (ROLL5_USG − ROLL20_USG)[t]
```

### Composite Prospect Theory Index

```
PROSPECT_THEORY_INDEX =
    0.30 × SCORING_DISAPPOINTMENT_Z
  + 0.25 × COMING_OFF_LOSS
  + 0.20 × USG_INCREASE_AFTER_LOSS
  + 0.15 × SHOT_AGG_CHANGE_Z
  + 0.10 × CLOSE_LOSS_FLAG
```

Weights are configurable in `config.yaml` under `prospect_theory.weights`.

---

## Pressure Index

Quantifies structural game-context pressure independent of prior performance.

### Composite Pressure Index

```
PRESSURE_INDEX =
    0.20 × PLAYOFF_FLAG
  + 0.15 × CLOSE_SPREAD_FLAG      (|spread| ≤ 4.0)
  + 0.15 × STRONG_OPP_FLAG        (OPP_OFF_RATING > 110)
  + 0.10 × LATE_SEASON_FLAG       (March–April, non-playoff)
  + 0.10 × LOSING_STREAK_FLAG     (streak ≥ 2)
  + 0.10 × NATIONAL_TV_FLAG       (high-market opponent proxy)
  + 0.10 × RIVALRY_FLAG           (GSW, BOS, MIA, CLE, DAL)
  + 0.10 × HIGH_TOTAL_FLAG        (O/U > 225)
```

### Cross-Index Interactions

```
PRESSURE_X_PROSPECT = PRESSURE_INDEX × PROSPECT_THEORY_INDEX
PRESSURE_X_OPP_DEF  = PRESSURE_INDEX × OPP_DEF_RATING
FATIGUE_X_PRESSURE  = BACK_TO_BACK × PRESSURE_INDEX
HOME_X_PRESSURE     = HOME_GAME × PRESSURE_INDEX
```

---

## Model Training

### Walk-Forward Cross-Validation

Time-series safe — never trains on future games.

```
For fold k in [0, n_splits):
  test_start = N - n_splits×test_size + k×test_size
  train  = games[0 : test_start]          # all history up to fold
  test   = games[test_start : test_start + test_size]   # next 20 games
  MAE_k  = mean_absolute_error(y_test, model.predict(X_test))
```

Default: `n_splits = 5`, `test_size = 20 games` (100-game holdout across 5 non-overlapping folds).

### Model Selection

The model with the lowest mean walk-forward MAE is selected, saved, and used for prediction. All models use `SimpleImputer(strategy="median")` to handle missing odds columns.

### Candidate Models

| Model | Key Hyperparameters |
|---|---|
| Linear Regression | — |
| Ridge | α = 1.0 |
| Lasso | α = 0.5 |
| ElasticNet | α = 0.5, l1\_ratio = 0.5 |
| Random Forest | 200 trees |
| Gradient Boosting | 200 trees, lr = 0.05 |
| XGBoost | 300 trees, lr = 0.05, max\_depth = 5 |
| LightGBM | 300 trees, lr = 0.05, num\_leaves = 31 |

---

## Evaluation Metrics

```
MAE  = (1/N) Σ |y_i − ŷ_i|
RMSE = √( (1/N) Σ (y_i − ŷ_i)² )
R²   = 1 − Σ(y_i − ŷ_i)² / Σ(y_i − ȳ)²
```

Compared against a rolling-average baseline:
```
baseline_pred[t] = mean( PTS[t-5 : t-1] )
```

Segment metrics are computed for: home vs. away, back-to-back vs. rested, playoff vs. regular season, tough vs. weak opponent defense.

---

## Project Structure

```
lebron-points-prediction/
├── main.py                    # CLI entry point
├── config.yaml                # all tunable parameters
├── .env                       # SPORTS_API_KEY (never committed)
├── src/
│   ├── config.py              # config loader
│   ├── nba_client.py          # nba_api wrapper with caching
│   ├── odds_client.py         # Sports API wrapper (optional)
│   ├── data_pipeline.py       # data orchestration + merging
│   ├── feature_engineering.py # all feature groups
│   ├── prospect_theory.py     # behavioral features (K&T 1979)
│   ├── pressure_index.py      # game-context pressure features
│   ├── modeling.py            # walk-forward CV + model training
│   ├── evaluate.py            # metrics, charts, SHAP
│   ├── predict_next_game.py   # next-game prediction report
│   └── sample_data.py         # synthetic fallback when API blocked
├── data/
│   ├── raw/nba/               # cached JSON from nba_api (gitignored)
│   └── processed/             # parquet files (gitignored)
└── outputs/
    ├── model_artifacts/       # saved models + feature lists
    ├── charts/                # evaluation plots
    └── reports/               # metrics CSV
```

---

## Configuration

Key settings in `config.yaml`:

```yaml
odds:
  historical_enabled: false   # true = fetch historical odds (most APIs don't support this)
  use_for_prediction: true    # try odds API on predict-next

model:
  n_splits: 5
  test_size: 20               # games per fold

rolling_windows: [3, 5, 10, 20]
ewma_spans: [3, 5, 10]
```

---

## Disclaimer

This is a statistical/research project. Predictions are for analytical purposes only and do not constitute financial or betting advice.
