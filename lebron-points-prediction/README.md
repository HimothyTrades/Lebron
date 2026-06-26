# LeBron James Points Prediction Model

A machine learning pipeline that predicts LeBron James's points scored in his next NBA game. The project integrates traditional box-score modeling, market-implied expectations from Vegas odds, opponent-adjusted defensive context, schedule fatigue analysis, and behavioral variables inspired by Prospect Theory.

---

## Project Goal

Can we build a statistically rigorous model that predicts how many points LeBron James will score on a given night — combining what the box score tells us with what the betting market implies?

This project answers that question by combining:
- **NBA historical performance data** (game logs, advanced stats, box scores)
- **Opponent defensive context** (defensive rating, pace, paint/perimeter defense)
- **Team context** (Lakers offensive rating, lineup, fatigue)
- **Vegas market data** (spread, moneyline, over/under, player points props)
- **Behavioral variables** (Prospect Theory, pressure index, momentum)

The result is a multi-model ensemble pipeline with walk-forward validation, explainability via feature importances and SHAP values, and a clean next-game prediction interface.

---

## Data Sources

### NBA Stats (nba_api)
Uses the open-source [`nba_api`](https://github.com/jasonroman/nba-api) package to pull:
- **PlayerGameLog** — per-game box score for LeBron across all seasons
- **LeagueDashPlayerStats** — league-wide player stats (Base + Advanced)
- **LeagueDashTeamStats** — team stats (Base + Advanced + Opponent)
- **BoxScoreTraditionalV2 / BoxScoreAdvancedV2** — game-level box scores
- **LeagueGameFinder** — Lakers schedule and game IDs
- **CommonPlayerInfo** — player metadata

### Vegas / Odds Data (sports-api.net)
Uses the Sports API to pull:
- Moneyline odds for each game
- Point spread
- Total (over/under)
- Player points props (LeBron scoring line, over/under price)

All endpoint paths, parameter names, and league IDs are configurable in `config.yaml` under the `sports_api:` section.

---

## Setup

### 1. Clone the repository
```bash
git clone <repo-url>
cd lebron-points-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

Optional but recommended for tree-model explainability:
```bash
pip install shap
```

### 3. Configure your API key

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Edit `.env` and add your Sports API key:
```
SPORTS_API_KEY=your_actual_key_here
SPORTS_API_BASE_URL=https://sports-api.net
NBA_SEASON_START=2018
NBA_SEASON_END=2026
```

**The key is never hardcoded.** It is loaded at runtime from the `.env` file via `python-dotenv`.

### 4. (Optional) Adjust config.yaml

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

---

## Running the Pipeline

### Full pipeline (recommended for first run)
```bash
python main.py run-all
```

### Step by step
```bash
# 1. Fetch NBA + odds data
python main.py fetch-data

# 2. Build feature set
python main.py build-features

# 3. Train all models, select best
python main.py train

# 4. Evaluate model performance, generate charts
python main.py evaluate

# 5. Predict LeBron's next game
python main.py predict-next
```

---

## Prediction Output

Running `python main.py predict-next` outputs:

```
============================================================
  LEBRON JAMES POINTS PREDICTION
  2025-12-26
============================================================
  Predicted Points:     27.4
  80% Confidence Range: 19.2 – 35.6
  Vegas Prop Line:      26.5
  Model Edge vs Vegas:  +0.9
  Recommendation:       NO CLEAR EDGE — model and market are close

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

---

## Feature Categories

### Player Performance
Box score stats: points, minutes, FGA, FG%, 3PA, 3P%, FTA, FT%, eFG%, true shooting %, usage rate proxy, assists, rebounds, turnovers, plus/minus.

### Recent Form (Rolling Windows: 3, 5, 10, 20 games)
Rolling averages, standard deviation, volatility, and z-scores for all key stats. **All rolling features are computed on lagged data (shifted by 1 game) to prevent data leakage.**

### Exponentially Weighted Moving Averages (EWMA)
Spans: 3, 5, 10. Applied to points, minutes, usage, FGA, and true shooting %. More recent games receive higher weight.

### Opportunity Features
Expected minutes, shot volume trend, usage trend, back-to-back status, games missed before the current game.

### Opponent Defense
Defensive rating, pace, FG% allowed, 3P% allowed, points allowed per game. Pulled from `LeagueDashTeamStats` with the `Opponent` measure type.

### Team Context
Lakers offensive rating, pace, net rating, and rolling 5-game offensive context.

### Rest & Fatigue
Days of rest, back-to-back flag, third game in four nights, fourth game in six nights, cumulative minutes over 3/5/7/14 days.

### Home / Away
Binary flags. Travel distance and time zone change are noted as future improvements.

### Vegas Features
- `implied_team_total` — derived from spread + total
- `implied_win_probability` — from moneyline
- `favorite_flag` / `underdog_flag`
- `blowout_risk` — spread > 10 points
- `player_points_line` — from player prop market
- `market_expectation_gap` — model prediction minus Vegas line

---

## Prospect Theory Features

Inspired by **Kahneman and Tversky's Prospect Theory (1979)**: people evaluate outcomes relative to a reference point and are disproportionately sensitive to losses compared to equivalent gains.

Applied here: LeBron's scoring behavior may shift based on how recent performance compares to his own expectations and historical baseline.

**Reference points used:**
- Season average points (up to current game)
- Rolling 10-game average
- Previous Vegas player prop line

**Features created:**
- `scoring_disappointment` — how far below expectation last game was
- `scoring_overperformance` — how far above expectation
- `loss_aversion_score` — weighted disappointment metric
- `coming_off_loss`, `coming_off_two_losses`, `blowout_loss`
- `close_loss_flag` — lost by 5 or fewer points
- `shot_aggressiveness_change` — FGA trend after poor scoring game
- `three_point_aggressiveness_change`, `ft_aggressiveness_change`

**Composite index:**
```
prospect_theory_index =
  0.30 × scoring_disappointment_z
+ 0.25 × coming_off_loss
+ 0.20 × usage_increase_after_loss
+ 0.15 × shot_aggressiveness_change_z
+ 0.10 × close_loss_flag
```
Weights are configurable in `config.yaml`.

---

## Pressure Index

Quantifies game-context pressure that may concentrate or disrupt performance.

**Components:**
- `playoff_flag` — playoff game
- `close_spread_flag` — spread ≤ 4 points
- `strong_opponent_flag` — opponent offensive rating above threshold
- `late_season_flag` — March/April regular season games
- `losing_streak_flag` — 2+ consecutive losses
- `national_tv_flag` — proxy based on marquee matchups (see note below)
- `rivalry_flag` — vs. GSW, BOS, MIA, CLE, DAL
- `high_total_flag` — over/under above 225

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
```

> **Note on national TV:** Exact broadcast schedule data is not available from `nba_api`. The `NATIONAL_TV_FLAG` uses a team-market proxy (historically high-TV-market opponents). Replace with actual broadcast data for improved accuracy.

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

### Validation Strategy
**Walk-forward cross-validation** is used throughout. The dataset is split chronologically: the model is always trained on older games and tested on newer games. Random splitting is never used — that would leak future information into the training set.

### Feature Pipeline
All features using rolling/lagged data are shifted by 1 game before computing averages. This ensures the model never sees the current game's stats when making a prediction for that game.

### Model Selection
After walk-forward CV, the model with the lowest mean absolute error (MAE) is selected as the best model and saved to `outputs/model_artifacts/best_model.joblib`.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error in points |
| RMSE | Root Mean Square Error |
| R² | Variance explained |
| Baseline MAE | Rolling 5-game average benchmark |

Segmented metrics are also computed for: home vs. away, back-to-back vs. rested, playoff vs. regular season, tough vs. weak defense opponents.

Charts saved to `outputs/charts/`:
- Actual vs Predicted
- Residual distribution
- Rolling MAE
- Feature importance (top 30)
- Model comparison
- SHAP summary (if SHAP installed)

---

## Limitations

1. **nba_api rate limits** — the API enforces rate limits. The pipeline includes retry logic and local caching to mitigate this.
2. **Sports API endpoint variability** — `sports-api.net` endpoint structure may differ from the defaults in `config.yaml`. Edit the `sports_api:` section accordingly.
3. **National TV data** — no free programmatic source. Using team-market proxy.
4. **Lineup data** — teammate injury/absence data is not included (labeled as placeholder). This is a meaningful missing feature.
5. **Real-time updates** — this pipeline is batch-based. For live game-day predictions, you'd run `fetch-data` and `predict-next` on game day.
6. **No guarantee of accuracy** — sports are inherently noisy. LeBron's variance in any given game is high. Use this for analytical insight, not betting decisions.

---

## Future Improvements

- Integrate real-time lineup/injury data (e.g., official NBA injury reports)
- Add national TV broadcast schedule as a proper feature
- Include altitude and time zone travel data for road games
- Bayesian hyperparameter optimization (Optuna)
- Stacking ensemble of all trained models
- Live game-day pipeline with automated daily refresh
- Confidence intervals from quantile regression
- Historical odds backfill for full training coverage

---

## Project Structure

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
```

---

## Disclaimer

This project is for educational and analytical purposes only. It does not constitute financial advice. Sports prediction involves significant uncertainty. Past performance patterns do not guarantee future results.
