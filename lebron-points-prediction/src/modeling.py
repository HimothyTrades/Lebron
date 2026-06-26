"""
Model training, selection, and walk-forward validation.

Models: baseline rolling avg, Linear, Ridge, Lasso, ElasticNet,
        RandomForest, GradientBoosting, XGBoost, LightGBM.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import get_config

logger = logging.getLogger("lebron.modeling")

# Columns to always drop from features
DROP_ALWAYS = [
    "GAME_DATE", "GAME_ID", "Game_ID", "MATCHUP", "WL", "WIN",
    "SEASON", "TEAM_ID", "PLAYER_ID", "OPP_ABBREV",
    "VIDEO_AVAILABLE", "TEAM_NAME", "TEAM_ABBREVIATION",
]


def _try_import_xgboost():
    try:
        from xgboost import XGBRegressor
        return XGBRegressor
    except ImportError:
        return None


def _try_import_lightgbm():
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor
    except ImportError:
        return None


def _build_models(cfg) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "linear": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "lasso": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.5, max_iter=5000)),
        ]),
        "elastic_net": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ]),
        "gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)),
        ]),
    }

    XGB = _try_import_xgboost()
    if XGB is not None:
        models["xgboost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGB(n_estimators=300, learning_rate=0.05, max_depth=5,
                          random_state=42, verbosity=0)),
        ])

    LGB = _try_import_lightgbm()
    if LGB is not None:
        models["lightgbm"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGB(n_estimators=300, learning_rate=0.05, num_leaves=31,
                          random_state=42, verbose=-1)),
        ])

    return models


def _get_feature_cols(df: pd.DataFrame, target: str) -> List[str]:
    """Select numeric feature columns, excluding target and metadata."""
    drop = set(DROP_ALWAYS + [target])
    cols = [c for c in df.columns if c not in drop and df[c].dtype in [np.float64, np.int64, np.float32, np.int32, bool]]
    return cols


def walk_forward_validate(
    df: pd.DataFrame,
    models: Dict[str, Any],
    target: str,
    feature_cols: List[str],
    n_splits: int = 5,
    test_size: int = 20,
) -> pd.DataFrame:
    """
    Walk-forward (time-series) cross-validation.
    For each fold, train on all data before the test window; test on the next `test_size` games.
    Returns a DataFrame with MAE, RMSE, R2 per model per fold.
    """
    n = len(df)
    min_train = n - n_splits * test_size
    if min_train < 50:
        logger.warning(f"Very small training set for walk-forward CV ({min_train} rows). Results may be unreliable.")

    records = []
    for fold in range(n_splits):
        test_start = min_train + fold * test_size
        test_end = test_start + test_size
        if test_end > n:
            break

        train_df = df.iloc[:test_start]
        test_df = df.iloc[test_start:test_end]

        X_train = train_df[feature_cols]
        y_train = train_df[target]
        X_test = test_df[feature_cols]
        y_test = test_df[target]

        # Drop rows with NaN target in training
        mask = y_train.notna()
        X_train, y_train = X_train[mask], y_train[mask]

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                valid_mask = y_test.notna()
                mae = mean_absolute_error(y_test[valid_mask], preds[valid_mask])
                rmse = np.sqrt(mean_squared_error(y_test[valid_mask], preds[valid_mask]))
                r2 = r2_score(y_test[valid_mask], preds[valid_mask]) if valid_mask.sum() > 1 else np.nan
                records.append({"model": name, "fold": fold, "MAE": mae, "RMSE": rmse, "R2": r2})
            except Exception as e:
                logger.warning(f"Model {name} failed on fold {fold}: {e}")

    return pd.DataFrame(records)


def baseline_rolling_avg(df: pd.DataFrame, target: str, window: int = 5) -> pd.Series:
    """Simple rolling average baseline prediction."""
    return df[target].shift(1).rolling(window, min_periods=1).mean()


class ModelTrainer:
    """Train all models, select the best, and save artifacts."""

    def __init__(self):
        self.cfg = get_config()
        self.target = self.cfg.target
        self.models = _build_models(self.cfg)
        self.best_model = None
        self.best_model_name = None
        self.feature_cols: List[str] = []
        self.cv_results: Optional[pd.DataFrame] = None

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Full training pipeline: walk-forward CV → select best → final fit → save."""
        df = df.sort_values("GAME_DATE").reset_index(drop=True)

        self.feature_cols = _get_feature_cols(df, self.target)
        logger.info(f"Training with {len(self.feature_cols)} features, {len(df)} games.")

        # Walk-forward validation
        logger.info("Running walk-forward cross-validation...")
        self.cv_results = walk_forward_validate(
            df,
            self.models,
            self.target,
            self.feature_cols,
            n_splits=self.cfg.n_splits,
            test_size=self.cfg.test_size,
        )

        # Select best model by mean MAE
        summary = self.cv_results.groupby("model")["MAE"].mean()
        self.best_model_name = summary.idxmin()
        logger.info(f"Best model: {self.best_model_name} (mean MAE={summary.min():.2f})")

        # Final fit on full dataset
        self.best_model = self.models[self.best_model_name]
        X = df[self.feature_cols]
        y = df[self.target]
        mask = y.notna()
        self.best_model.fit(X[mask], y[mask])

        # Save artifacts
        self._save_artifacts()

        return {
            "best_model_name": self.best_model_name,
            "cv_results": self.cv_results,
            "feature_cols": self.feature_cols,
            "cv_summary": summary.to_dict(),
        }

    def _save_artifacts(self):
        artifacts_dir = self.cfg.model_artifacts_path
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        model_path = artifacts_dir / "best_model.joblib"
        joblib.dump(self.best_model, model_path)
        logger.info(f"Best model saved: {model_path}")

        features_path = artifacts_dir / "features.json"
        with open(features_path, "w") as f:
            json.dump(self.feature_cols, f, indent=2)
        logger.info(f"Feature list saved: {features_path}")

        meta_path = artifacts_dir / "model_meta.json"
        with open(meta_path, "w") as f:
            json.dump({"model_name": self.best_model_name, "n_features": len(self.feature_cols)}, f, indent=2)

    def load_best_model(self) -> Tuple[Any, List[str]]:
        """Load saved best model and feature list."""
        model_path = self.cfg.model_artifacts_path / "best_model.joblib"
        features_path = self.cfg.model_artifacts_path / "features.json"
        if not model_path.exists():
            raise FileNotFoundError(f"No saved model at {model_path}. Run training first.")
        model = joblib.load(model_path)
        with open(features_path) as f:
            features = json.load(f)
        return model, features

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.best_model is None:
            self.best_model, self.feature_cols = self.load_best_model()
        X = df[self.feature_cols]
        return self.best_model.predict(X)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Extract feature importances or coefficients from best model."""
        if self.best_model is None:
            return None
        try:
            inner = self.best_model.named_steps["model"]
        except AttributeError:
            inner = self.best_model

        if hasattr(inner, "feature_importances_"):
            imp = inner.feature_importances_
        elif hasattr(inner, "coef_"):
            imp = np.abs(inner.coef_)
        else:
            return None

        # LightGBM/XGBoost store feature names on the booster
        feature_names = self.feature_cols
        if hasattr(inner, "feature_name_"):
            feature_names = inner.feature_name_
        elif hasattr(inner, "feature_names_in_"):
            feature_names = list(inner.feature_names_in_)

        n = min(len(feature_names), len(imp))
        return pd.DataFrame(
            {"feature": feature_names[:n], "importance": imp[:n]}
        ).sort_values("importance", ascending=False).reset_index(drop=True)
