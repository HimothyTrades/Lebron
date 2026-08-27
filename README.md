# LeBron James Points Prediction Model
# LeBron Points Prediction

A machine learning pipeline that predicts LeBron James's points scored in his next NBA game. The project integrates traditional box-score modeling, market-implied expectations from Vegas odds, opponent-adjusted defensive context, schedule fatigue analysis, and behavioral variables inspired by Prospect Theory.
A machine-learning pipeline that predicts LeBron James's points scored in the next game, trained on eight seasons (2018–2026) of NBA box scores combined with behavioral and game-context features.

---

## Project Goal
## Quick Start

Can we build a statistically rigorous model that predicts how many points LeBron James will score on a given night — combining what the box score tells us with what the betting market implies?
```bash
pip install -r requirements.txt
cp .env.example .env        # add SPORTS_API_KEY if you have one (optional)

This project answers that question by combining:
- **NBA historical performance data** (game logs, advanced stats, box scores)
- **Opponent defensive context** (defensive rating, pace, paint/perimeter defense)
- **Team context** (Lakers offensive rating, lineup, fatigue)
- **Vegas market data** (spread, moneyline, over/under, player points props)
- **Behavioral variables** (Prospect Theory, pressure index, momentum)
python main.py run-all      # fetch data → build features → train → evaluate → predict
```

The result is a multi-model ensemble pipeline with walk-forward validation, explainability via feature importances and SHAP values, and a clean next-game prediction interface.
Individual steps:

```bash
python main.py fetch-data       # pull NBA game logs and team stats
python main.py build-features   # engineer all features
python main.py train            # walk-forward cross-validation + model selection
python main.py evaluate         # holdout metrics, charts, SHAP analysis
python main.py predict-next     # predict next game points
```

---

## Data Sources
## Prediction Formula

### NBA Stats (nba_api)
Uses the open-source [`nba_api`](https://github.com/jasonroman/nba-api) package to pull:
- **PlayerGameLog** — per-game box score for LeBron across all seasons
- **LeagueDashPlayerStats** — league-wide player stats (Base + Advanced)
- **LeagueDashTeamStats** — team stats (Base + Advanced + Opponent)
- **BoxScoreTraditionalV2 / BoxScoreAdvancedV2** — game-level box scores
- **LeagueGameFinder** — Lakers schedule and game IDs
- **CommonPlayerInfo** — player metadata
The model predicts:

### Vegas / Odds Data (sports-api.net)
Uses the Sports API to pull:
- Moneyline odds for each game
- Point spread
- Total (over/under)
- Player points props (LeBron scoring line, over/under price)
```
ŷ = f(X_rolling, X_efficiency, X_opponent, X_fatigue, X_vegas, X_pt, X_pressure)
```

All endpoint paths, parameter names, and league IDs are configurable in `config.yaml` under the `sports_api:` section.
Where `f` is the best-performing model selected by walk-forward cross-validation across:
Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, XGBoost, LightGBM.

---

## Setup
## Feature Groups

### 1. Clone the repository
```bash
git clone <repo-url>
cd lebron-points-prediction
```
### 1. Rolling Window Statistics

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
For each stat `s ∈ {PTS, MIN, FGA, FG3A, FTA, FG_PCT, TS_PCT, USG_PROXY, ...}` and window `w ∈ {3, 5, 10, 20}`:

Optional but recommended for tree-model explainability:
```bash
pip install shap
```
ROLL{w}_{s}[t] = mean( s[t-w : t-1] )   # shifted by 1 — no data leakage
```

### 3. Configure your API key

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
Volatility (coefficient of variation):
```
ROLL{w}_PTS_VOLATILITY[t] = ROLL{w}_PTS_STD[t] / ROLL{w}_PTS[t]
```

Edit `.env` and add your Sports API key:
Z-score within window:
```
SPORTS_API_KEY=your_actual_key_here
SPORTS_API_BASE_URL=https://sports-api.net
NBA_SEASON_START=2018
NBA_SEASON_END=2026
ROLL{w}_PTS_ZSCORE[t] = (PTS[t-1] - μ_w) / σ_w
```

**The key is never hardcoded.** It is loaded at runtime from the `.env` file via `python-dotenv`.
### 2. Exponentially-Weighted Moving Average

### 4. (Optional) Adjust config.yaml
For span `α ∈ {3, 5, 10}` and `col ∈ {PTS, MIN, USG_PROXY, FGA, TS_PCT}`:

If `sports-api.net` uses different endpoint paths or parameter names, edit `config.yaml`:
```yaml
sports_api:
  events_endpoint: "/v1/events"         # change if your API differs
  odds_endpoint: "/v1/odds"
  player_props_endpoint: "/v1/player-props"
  params:
    api_key_param: "api_key"            # the query param name for your key
    sport_param: "sport"
    league_param: "league"
```
EWMA{α}_{col}[t] = (1-λ) · col[t-1] + λ · EWMA{α}_{col}[t-1]
  where λ = 1 - 2/(α+1)
```

---
### 3. Efficiency Stats

## Running the Pipeline
True Shooting Percentage:
```
TS% = PTS / (2 × (FGA + 0.44 × FTA))
```

### Full pipeline (recommended for first run)
```bash
python main.py run-all
Effective Field Goal Percentage:
```
eFG% = (FGM + 0.5 × FG3M) / FGA
```

### Step by step
```bash
# 1. Fetch NBA + odds data
python main.py fetch-data
Usage Rate Proxy:
```
USG_PROXY = (FGA + 0.44×FTA + TOV) × 100 / (MIN × 5)
```

# 2. Build feature set
python main.py build-features
### 4. Vegas / Market Features

# 3. Train all models, select best
python main.py train
Implied team total from spread and over/under:
```
IMPLIED_TEAM_TOTAL = (O/U / 2) ± (|spread| / 2)
  + if Lakers are favored (spread < 0), − if underdogs
```

# 4. Evaluate model performance, generate charts
python main.py evaluate
Implied win probability from American moneyline:
```
              100 / (ML + 100)        if ML > 0  (underdog)
IMPLIED_P =
              |ML| / (|ML| + 100)     if ML < 0  (favorite)
```

### 5. Interaction Terms

# 5. Predict LeBron's next game
python main.py predict-next
```
OPP_DEF_X_USG   = OPP_DEF_RATING × ROLL5_USG_PROXY
PACE_X_MIN      = OPP_PACE × LAST_GAME_MIN
SPREAD_X_MIN    = |LAKERS_SPREAD| × LAST_GAME_MIN
TEAM_TOTAL_X_USG= IMPLIED_TEAM_TOTAL × ROLL5_USG_PROXY
HOME_X_OPP_DEF  = HOME_GAME × OPP_DEF_RATING
FATIGUE_X_B2B   = BACK_TO_BACK × AVG_MIN_L5
```

---

## Prediction Output
## Prospect Theory Index

Running `python main.py predict-next` outputs:
Based on Kahneman & Tversky (1979): decision-makers evaluate outcomes relative to a reference point and weight losses more heavily than equivalent gains. Applied here, LeBron's scoring effort may shift based on how recent performance compares to expectations.

```

  LEBRON JAMES POINTS PREDICTION
  2025-12-26
============================================================
  Predicted Points:     27.4
  80% Confidence Range: 19.2 – 35.6
  Vegas Prop Line:      26.5
  Model Edge vs Vegas:  +0.9
  Recommendation:       NO CLEAR EDGE — model and market are close
### Reference Points

  Top Drivers:
    - ROLL5_PTS
    - IMPLIED_TEAM_TOTAL
    - OPP_DEF_RATING
    - PROSPECT_THEORY_INDEX
    - PRESSURE_INDEX

  DISCLAIMER: This is a statistical/analytical model only.
  It does not constitute financial or betting advice.
============================================================
```
SEASON_AVG_PTS[t]  = expanding mean of PTS[1 : t-1] within season
ROLL10_PTS_REF[t]  = mean( PTS[t-10 : t-1] )
PREV_PROP_LINE[t]  = Vegas player prop from game t-1 (if available)
```

---

## Feature Categories
### Gains and Losses

### Player Performance
Box score stats: points, minutes, FGA, FG%, 3PA, 3P%, FTA, FT%, eFG%, true shooting %, usage rate proxy, assists, rebounds, turnovers, plus/minus.
```
PREV_GAME_PTS_GAP[t]     = PTS[t-1] − ROLL10_PTS_REF[t]

### Recent Form (Rolling Windows: 3, 5, 10, 20 games)
Rolling averages, standard deviation, volatility, and z-scores for all key stats. **All rolling features are computed on lagged data (shifted by 1 game) to prevent data leakage.**
SCORING_DISAPPOINTMENT[t] = max(0,  ROLL10_PTS_REF[t] − PTS[t-1])   # below expectation
SCORING_OVERPERFORMANCE[t]= max(0,  PTS[t-1] − ROLL10_PTS_REF[t])   # above expectation
```

### Exponentially Weighted Moving Averages (EWMA)
Spans: 3, 5, 10. Applied to points, minutes, usage, FGA, and true shooting %. More recent games receive higher weight.
Z-scored across all games:
```
SCORING_DISAPPOINTMENT_Z[t] = (SCORING_DISAPPOINTMENT[t] − μ_D) / σ_D
```

### Opportunity Features
Expected minutes, shot volume trend, usage trend, back-to-back status, games missed before the current game.
Loss aversion amplification (λ = 2.25, per Kahneman & Tversky):
```
LOSS_AVERSION_SCORE[t] = SCORING_DISAPPOINTMENT_Z[t] × 2.25
```

### Opponent Defense
Defensive rating, pace, FG% allowed, 3P% allowed, points allowed per game. Pulled from `LeagueDashTeamStats` with the `Opponent` measure type.
### Shot Aggressiveness After Disappointment

### Team Context
Lakers offensive rating, pace, net rating, and rolling 5-game offensive context.
```
SHOT_AGGRESSIVENESS_CHANGE[t] = ROLL3_FGA[t] − ROLL10_FGA[t]
USG_INCREASE_AFTER_LOSS[t]    = COMING_OFF_LOSS[t] × (ROLL5_USG − ROLL20_USG)[t]
```

### Rest & Fatigue
Days of rest, back-to-back flag, third game in four nights, fourth game in six nights, cumulative minutes over 3/5/7/14 days.
### Composite Prospect Theory Index

### Home / Away
Binary flags. Travel distance and time zone change are noted as future improvements.
```
PROSPECT_THEORY_INDEX =
    0.30 × SCORING_DISAPPOINTMENT_Z
  + 0.25 × COMING_OFF_LOSS
  + 0.20 × USG_INCREASE_AFTER_LOSS
  + 0.15 × SHOT_AGG_CHANGE_Z
  + 0.10 × CLOSE_LOSS_FLAG
```

### Vegas Features
- `implied_team_total` — derived from spread + total
- `implied_win_probability` — from moneyline
- `favorite_flag` / `underdog_flag`
- `blowout_risk` — spread > 10 points
- `player_points_line` — from player prop market
- `market_expectation_gap` — model prediction minus Vegas line
Weights are configurable in `config.yaml` under `prospect_theory.weights`.

---

## Prospect Theory Features
## Pressure Index

Inspired by **Kahneman and Tversky's Prospect Theory (1979)**: people evaluate outcomes relative to a reference point and are disproportionately sensitive to losses compared to equivalent gains.
Quantifies structural game-context pressure independent of prior performance.

Applied here: LeBron's scoring behavior may shift based on how recent performance compares to his own expectations and historical baseline.
### Composite Pressure Index

**Reference points used:**
- Season average points (up to current game)
- Rolling 10-game average
- Previous Vegas player prop line
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

**Features created:**
- `scoring_disappointment` — how far below expectation last game was
- `scoring_overperformance` — how far above expectation
- `loss_aversion_score` — weighted disappointment metric
- `coming_off_loss`, `coming_off_two_losses`, `blowout_loss`
- `close_loss_flag` — lost by 5 or fewer points
- `shot_aggressiveness_change` — FGA trend after poor scoring game
- `three_point_aggressiveness_change`, `ft_aggressiveness_change`
### Cross-Index Interactions

**Composite index:**
```
prospect_theory_index =
  0.30 × scoring_disappointment_z
+ 0.25 × coming_off_loss
+ 0.20 × usage_increase_after_loss
+ 0.15 × shot_aggressiveness_change_z
+ 0.10 × close_loss_flag
PRESSURE_X_PROSPECT = PRESSURE_INDEX × PROSPECT_THEORY_INDEX
PRESSURE_X_OPP_DEF  = PRESSURE_INDEX × OPP_DEF_RATING
FATIGUE_X_PRESSURE  = BACK_TO_BACK × PRESSURE_INDEX
HOME_X_PRESSURE     = HOME_GAME × PRESSURE_INDEX
```
Weights are configurable in `config.yaml`.

---

## Pressure Index
## Model Training

Quantifies game-context pressure that may concentrate or disrupt performance.
### Walk-Forward Cross-Validation

**Components:**
- `playoff_flag` — playoff game
- `close_spread_flag` — spread ≤ 4 points
- `strong_opponent_flag` — opponent offensive rating above threshold
- `late_season_flag` — March/April regular season games
- `losing_streak_flag` — 2+ consecutive losses
- `national_tv_flag` — proxy based on marquee matchups (see note below)
- `rivalry_flag` — vs. GSW, BOS, MIA, CLE, DAL
- `high_total_flag` — over/under above 225
Time-series safe — never trains on future games.

**Composite index:**
```
pressure_index =
  0.20 × playoff_flag
+ 0.15 × close_spread_flag
+ 0.15 × strong_opponent_flag
+ 0.10 × late_season_flag
+ 0.10 × losing_streak_flag
+ 0.10 × national_tv_flag
+ 0.10 × rivalry_flag
+ 0.10 × high_total_flag
For fold k in [0, n_splits):
  test_start = N - n_splits×test_size + k×test_size
  train  = games[0 : test_start]          # all history up to fold
  test   = games[test_start : test_start + test_size]   # next 20 games
  MAE_k  = mean_absolute_error(y_test, model.predict(X_test))
```

> **Note on national TV:** Exact broadcast schedule data is not available from `nba_api`. The `NATIONAL_TV_FLAG` uses a team-market proxy (historically high-TV-market opponents). Replace with actual broadcast data for improved accuracy.
Default: `n_splits = 5`, `test_size = 20 games` (100-game holdout across 5 non-overlapping folds).

---

## Model Methodology

### Models Trained
1. Rolling average baseline
2. Linear Regression
3. Ridge Regression
4. Lasso Regression
5. Elastic Net
6. Random Forest (200 trees)
7. Gradient Boosting (200 estimators)
8. XGBoost (300 estimators) — if installed
9. LightGBM (300 estimators) — if installed
### Model Selection

### Validation Strategy
**Walk-forward cross-validation** is used throughout. The dataset is split chronologically: the model is always trained on older games and tested on newer games. Random splitting is never used — that would leak future information into the training set.
The model with the lowest mean walk-forward MAE is selected, saved, and used for prediction. All models use `SimpleImputer(strategy="median")` to handle missing odds columns.

### Feature Pipeline
All features using rolling/lagged data are shifted by 1 game before computing averages. This ensures the model never sees the current game's stats when making a prediction for that game.
### Candidate Models

### Model Selection
After walk-forward CV, the model with the lowest mean absolute error (MAE) is selected as the best model and saved to `outputs/model_artifacts/best_model.joblib`.
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

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error in points |
| RMSE | Root Mean Square Error |
| R² | Variance explained |
| Baseline MAE | Rolling 5-game average benchmark |
```
MAE  = (1/N) Σ |y_i − ŷ_i|
RMSE = √( (1/N) Σ (y_i − ŷ_i)² )
R²   = 1 − Σ(y_i − ŷ_i)² / Σ(y_i − ȳ)²
```

Segmented metrics are also computed for: home vs. away, back-to-back vs. rested, playoff vs. regular season, tough vs. weak defense opponents.
Compared against a rolling-average baseline:
```
baseline_pred[t] = mean( PTS[t-5 : t-1] )
```

Charts saved to `outputs/charts/`:
- Actual vs Predicted
- Residual distribution
- Rolling MAE
- Feature importance (top 30)
- Model comparison
- SHAP summary (if SHAP installed)
Segment metrics are computed for: home vs. away, back-to-back vs. rested, playoff vs. regular season, tough vs. weak opponent defense.

---

## Limitations
## Project Structure

1. **nba_api rate limits** — the API enforces rate limits. The pipeline includes retry logic and local caching to mitigate this.
2. **Sports API endpoint variability** — `sports-api.net` endpoint structure may differ from the defaults in `config.yaml`. Edit the `sports_api:` section accordingly.
3. **National TV data** — no free programmatic source. Using team-market proxy.
4. **Lineup data** — teammate injury/absence data is not included (labeled as placeholder). This is a meaningful missing feature.
5. **Real-time updates** — this pipeline is batch-based. For live game-day predictions, you'd run `fetch-data` and `predict-next` on game day.
6. **No guarantee of accuracy** — sports are inherently noisy. LeBron's variance in any given game is high. Use this for analytical insight, not betting decisions.
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

## Future Improvements
## Configuration

- Integrate real-time lineup/injury data (e.g., official NBA injury reports)
- Add national TV broadcast schedule as a proper feature
- Include altitude and time zone travel data for road games
- Bayesian hyperparameter optimization (Optuna)
- Stacking ensemble of all trained models
- Live game-day pipeline with automated daily refresh
- Confidence intervals from quantile regression
- Historical odds backfill for full training coverage
Key settings in `config.yaml`:

---
```yaml
odds:
  historical_enabled: false   # true = fetch historical odds (most APIs don't support this)
  use_for_prediction: true    # try odds API on predict-next

## Project Structure
model:
  n_splits: 5
  test_size: 20               # games per fold

```
lebron-points-prediction/
├── README.md
├── requirements.txt
├── .env.example
├── config.yaml
├── main.py
├── data/
│   ├── raw/
│   │   ├── nba/        # Cached nba_api JSON responses
│   │   └── odds/       # Cached Sports API JSON responses
│   ├── processed/      # Merged, feature-engineered Parquet files
│   └── predictions/    # Per-game prediction JSON files
├── notebooks/
│   └── 01_eda.ipynb    # Exploratory data analysis
├── src/
│   ├── config.py           # Config loader (YAML + .env)
│   ├── utils.py            # Logging, retry, caching, normalization
│   ├── nba_client.py       # nba_api wrapper
│   ├── odds_client.py      # Sports API wrapper
│   ├── data_pipeline.py    # Data orchestration
│   ├── feature_engineering.py  # All feature groups
│   ├── prospect_theory.py  # Behavioral features
│   ├── pressure_index.py   # Pressure/context features
│   ├── modeling.py         # Model training + walk-forward CV
│   ├── evaluate.py         # Metrics, charts, SHAP
│   └── predict_next_game.py    # Next-game prediction
└── outputs/
    ├── charts/             # PNG charts
    ├── model_artifacts/    # Saved models + feature lists
    └── reports/            # CSV metrics, logs
rolling_windows: [3, 5, 10, 20]
ewma_spans: [3, 5, 10]
```

---

## Disclaimer

This project is for educational and analytical purposes only. It does not constitute financial advice. Sports prediction involves significant uncertainty. Past performance patterns do not guarantee future results.
This is a statistical/research project. Predictions are for analytical purposes only and do not constitute financial or betting advice.
