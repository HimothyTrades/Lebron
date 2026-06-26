"""
Next-game prediction script.
Pulls latest NBA + odds data, builds features, loads best model, outputs prediction.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import get_config
from src.data_pipeline import DataPipeline
from src.feature_engineering import FeatureEngineer
from src.modeling import ModelTrainer
from src.odds_client import OddsClient
from src.prospect_theory import ProspectTheoryFeatures
from src.pressure_index import PressureIndexFeatures

logger = logging.getLogger("lebron.predict")


def _confidence_interval(model, X: pd.DataFrame, feature_cols, n_bootstrap: int = 200) -> Tuple[float, float]:
    """
    Bootstrap confidence interval via prediction variance across trees (RF/GBM)
    or simple residual-based interval for linear models.
    """
    try:
        inner = model.named_steps["model"]
        if hasattr(inner, "estimators_"):
            preds = np.array([est.predict(X[feature_cols])[0] for est in inner.estimators_])
            return float(np.percentile(preds, 10)), float(np.percentile(preds, 90))
    except Exception:
        pass
    # Fallback: ±1 SD from historical MAE (~6 points typical for LeBron models)
    pred = float(model.predict(X[feature_cols])[0])
    return pred - 6.5, pred + 6.5


def _american_odds_to_pct(ml) -> Optional[float]:
    try:
        ml = float(ml)
        if ml > 0:
            return 100 / (ml + 100)
        return abs(ml) / (abs(ml) + 100)
    except Exception:
        return None


class NextGamePredictor:
    """Load saved model and generate a next-game prediction report."""

    def __init__(self):
        self.cfg = get_config()
        self.trainer = ModelTrainer()
        self.model = None
        self.feature_cols = None

    def _load_model(self):
        self.model, self.feature_cols = self.trainer.load_best_model()

    def predict(self) -> Dict:
        """Full prediction pipeline. Returns a results dict."""
        self._load_model()

        # Load master dataset (must be built first)
        pipeline = DataPipeline()
        df = pipeline.load_master()
        if df is None:
            logger.info("No processed data found — running full pipeline...")
            df = pipeline.run()

        # Rebuild features
        fe = FeatureEngineer()
        df = fe.run(df)
        df = ProspectTheoryFeatures().run(df)
        df = PressureIndexFeatures().run(df)

        df = df.sort_values("GAME_DATE").reset_index(drop=True)

        # Use latest game's feature row as "next game" feature vector
        # In a live deployment, you'd fetch the actual next game's context here
        last_row = df.iloc[[-1]].copy()
        last_game_date = last_row["GAME_DATE"].values[0]
        logger.info(f"Using features from last known game: {last_game_date}")

        # Fetch odds for next game (tomorrow's date as proxy)
        next_date = (pd.Timestamp(last_game_date) + timedelta(days=1)).strftime("%Y-%m-%d")
        odds_client = OddsClient()
        odds_df = odds_client.match_odds_to_nba_games([next_date])

        # Update odds columns in last_row if available
        if not odds_df.empty and len(odds_df) > 0:
            for col in ["LAKERS_SPREAD", "over_under", "LAKERS_MONEYLINE",
                        "PLAYER_POINTS_LINE", "PLAYER_POINTS_OVER_PRICE",
                        "PLAYER_POINTS_UNDER_PRICE"]:
                if col in odds_df.columns and col in last_row.columns:
                    last_row[col] = odds_df.iloc[0].get(col, np.nan)

        # Predict
        X = last_row[[c for c in self.feature_cols if c in last_row.columns]]
        # Fill missing feature cols with NaN
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = np.nan
        X = X[self.feature_cols]

        pred_pts = float(self.model.predict(X)[0])
        ci_low, ci_high = _confidence_interval(self.model, last_row, self.feature_cols)

        # Feature importance for this prediction
        fi_df = self.trainer.get_feature_importance()
        top_factors = fi_df.head(5)["feature"].tolist() if fi_df is not None else []

        # Vegas comparison
        prop_line = float(last_row.get("PLAYER_POINTS_LINE", pd.Series([np.nan])).values[0])
        model_edge = pred_pts - prop_line if not np.isnan(prop_line) else None

        if model_edge is not None:
            if model_edge > 2.5:
                recommendation = "LEAN OVER (analytical edge only — not financial advice)"
            elif model_edge < -2.5:
                recommendation = "LEAN UNDER (analytical edge only — not financial advice)"
            else:
                recommendation = "NO CLEAR EDGE — model and market are close"
        else:
            recommendation = "No player prop line available for comparison"

        result = {
            "prediction_date": next_date,
            "player": self.cfg.player_name,
            "predicted_points": round(pred_pts, 1),
            "confidence_interval_80pct": (round(ci_low, 1), round(ci_high, 1)),
            "top_factors": top_factors,
            "vegas_player_prop_line": prop_line if not np.isnan(prop_line) else "N/A",
            "model_edge_vs_vegas": round(model_edge, 2) if model_edge is not None else "N/A",
            "recommendation": recommendation,
            "model_used": self.trainer.best_model_name if self.trainer.best_model_name else "loaded",
        }

        return result

    def print_report(self, result: Dict):
        print("\n" + "=" * 60)
        print(f"  LEBRON JAMES POINTS PREDICTION")
        print(f"  {result['prediction_date']}")
        print("=" * 60)
        print(f"  Predicted Points:     {result['predicted_points']}")
        print(f"  80% Confidence Range: {result['confidence_interval_80pct'][0]} – {result['confidence_interval_80pct'][1]}")
        print(f"  Vegas Prop Line:      {result['vegas_player_prop_line']}")
        print(f"  Model Edge vs Vegas:  {result['model_edge_vs_vegas']}")
        print(f"  Recommendation:       {result['recommendation']}")
        print()
        print(f"  Top Drivers:")
        for f in result["top_factors"]:
            print(f"    - {f}")
        print()
        print("  DISCLAIMER: This is a statistical/analytical model only.")
        print("  It does not constitute financial or betting advice.")
        print("=" * 60 + "\n")


def predict_next_game() -> Dict:
    predictor = NextGamePredictor()
    result = predictor.predict()
    predictor.print_report(result)

    # Save prediction
    cfg = get_config()
    out = cfg.predictions_path / f"prediction_{result['prediction_date']}.json"
    import json
    serializable = {k: (str(v) if not isinstance(v, (int, float, str, list, dict, type(None))) else v)
                    for k, v in result.items()}
    with open(out, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Prediction saved: {out}")
    return result
