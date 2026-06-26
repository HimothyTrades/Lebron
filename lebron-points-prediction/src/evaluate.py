"""
Model evaluation: metrics, charts, residual analysis, feature importance, SHAP.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import get_config
from src.modeling import ModelTrainer, baseline_rolling_avg, _get_feature_cols

logger = logging.getLogger("lebron.evaluate")


def _safe_shap(model, X: pd.DataFrame) -> Optional[np.ndarray]:
    try:
        import shap
        # XGBoost/LightGBM/RF use TreeExplainer; linear models use LinearExplainer
        try:
            inner = model.named_steps["model"]
        except AttributeError:
            inner = model
        if hasattr(inner, "feature_importances_"):
            explainer = shap.TreeExplainer(inner)
        elif hasattr(inner, "coef_"):
            imputer = model.named_steps.get("imputer")
            scaler = model.named_steps.get("scaler")
            X_t = X.copy()
            if imputer:
                X_t = pd.DataFrame(imputer.transform(X_t), columns=X_t.columns)
            if scaler:
                X_t = pd.DataFrame(scaler.transform(X_t), columns=X_t.columns)
            explainer = shap.LinearExplainer(inner, X_t)
            return explainer.shap_values(X_t)
        else:
            return None
        return explainer.shap_values(X)
    except Exception as e:
        logger.info(f"SHAP not available or failed: {e}")
        return None


class Evaluator:
    def __init__(self):
        self.cfg = get_config()
        self.target = self.cfg.target
        self.charts_dir = self.cfg.charts_path
        self.reports_dir = self.cfg.reports_path

    def run(self, df: pd.DataFrame, trainer: ModelTrainer) -> Dict:
        df = df.sort_values("GAME_DATE").reset_index(drop=True)
        feature_cols = trainer.feature_cols
        # Only use features that exist in this DataFrame
        feature_cols = [c for c in feature_cols if c in df.columns]

        # Walk-forward predictions on holdout portion
        test_start = len(df) - self.cfg.n_splits * self.cfg.test_size
        test_df = df.iloc[test_start:].copy()

        # Build X_test with exactly the saved feature columns (fill missing with NaN)
        X_test = pd.concat(
            {col: test_df[col] if col in test_df.columns else pd.Series(np.nan, index=test_df.index)
             for col in feature_cols},
            axis=1,
        )
        y_test = test_df[self.target]

        preds = trainer.best_model.predict(X_test)
        test_df["PREDICTED"] = preds
        test_df["RESIDUAL"] = y_test - preds
        test_df["ABS_ERROR"] = np.abs(test_df["RESIDUAL"])

        valid = y_test.notna()
        mae = mean_absolute_error(y_test[valid], preds[valid])
        rmse = np.sqrt(mean_squared_error(y_test[valid], preds[valid]))
        r2 = r2_score(y_test[valid], preds[valid]) if valid.sum() > 1 else np.nan

        logger.info(f"Holdout MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}")

        # Baseline comparison
        baseline_preds = baseline_rolling_avg(df, self.target)[test_start:]
        base_valid = y_test.notna() & baseline_preds.notna()
        baseline_mae = mean_absolute_error(y_test[base_valid], baseline_preds[base_valid]) if base_valid.sum() > 0 else np.nan

        metrics = {
            "MAE": mae, "RMSE": rmse, "R2": r2,
            "baseline_MAE": baseline_mae,
            "model": trainer.best_model_name,
        }

        # Segment metrics
        seg_metrics = self._segment_metrics(test_df, y_test, preds)
        metrics.update(seg_metrics)

        # Save metrics
        self._save_metrics(metrics, trainer.cv_results)

        # Charts
        self._plot_actual_vs_predicted(y_test, preds, test_df["GAME_DATE"] if "GAME_DATE" in test_df.columns else None)
        self._plot_residuals(test_df)
        self._plot_rolling_error(test_df)
        self._plot_feature_importance(trainer)
        self._plot_model_comparison(trainer.cv_results)
        self._shap_analysis(trainer.best_model, X_test, feature_cols)

        return metrics

    # ------------------------------------------------------------------
    # Segment metrics
    # ------------------------------------------------------------------

    def _segment_metrics(self, test_df: pd.DataFrame, y_true: pd.Series, preds: np.ndarray) -> Dict:
        results = {}

        def seg_mae(mask):
            m = mask & y_true.notna()
            if m.sum() == 0:
                return np.nan
            return mean_absolute_error(y_true[m], preds[m.values])

        # Home/Away
        if "HOME_GAME" in test_df.columns:
            results["MAE_home"] = seg_mae(test_df["HOME_GAME"] == 1)
            results["MAE_away"] = seg_mae(test_df["HOME_GAME"] == 0)

        # Rest
        if "BACK_TO_BACK" in test_df.columns:
            results["MAE_b2b"] = seg_mae(test_df["BACK_TO_BACK"] == 1)
            results["MAE_rested"] = seg_mae(test_df["BACK_TO_BACK"] == 0)

        # Playoff vs regular
        if "PLAYOFF_FLAG" in test_df.columns:
            results["MAE_playoff"] = seg_mae(test_df["PLAYOFF_FLAG"] == 1)
            results["MAE_regular"] = seg_mae(test_df["PLAYOFF_FLAG"] == 0)

        # Opponent defense tier
        if "OPP_DEF_RATING" in test_df.columns:
            med = test_df["OPP_DEF_RATING"].median()
            results["MAE_tough_defense"] = seg_mae(test_df["OPP_DEF_RATING"] > med)
            results["MAE_weak_defense"] = seg_mae(test_df["OPP_DEF_RATING"] <= med)

        return results

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------

    def _plot_actual_vs_predicted(self, y_true: pd.Series, preds: np.ndarray, dates=None):
        fig, ax = plt.subplots(figsize=(14, 5))
        x = range(len(y_true))
        ax.plot(x, y_true.values, label="Actual", linewidth=1.5, color="royalblue")
        ax.plot(x, preds, label="Predicted", linewidth=1.5, linestyle="--", color="tomato")
        ax.set_title(f"LeBron Points: Actual vs Predicted")
        ax.set_xlabel("Game")
        ax.set_ylabel("Points")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = self.charts_dir / "actual_vs_predicted.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        logger.info(f"Chart saved: {out}")

    def _plot_residuals(self, test_df: pd.DataFrame):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].scatter(test_df["PREDICTED"], test_df["RESIDUAL"], alpha=0.5, color="steelblue")
        axes[0].axhline(0, color="red", linestyle="--")
        axes[0].set_xlabel("Predicted Points")
        axes[0].set_ylabel("Residual")
        axes[0].set_title("Residual vs Predicted")

        axes[1].hist(test_df["RESIDUAL"].dropna(), bins=30, color="steelblue", edgecolor="white")
        axes[1].axvline(0, color="red", linestyle="--")
        axes[1].set_xlabel("Residual")
        axes[1].set_title("Residual Distribution")

        fig.tight_layout()
        out = self.charts_dir / "residuals.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        logger.info(f"Chart saved: {out}")

    def _plot_rolling_error(self, test_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(14, 4))
        roll_err = test_df["ABS_ERROR"].rolling(10, min_periods=1).mean()
        ax.plot(roll_err.values, color="darkorange")
        ax.set_title("Rolling 10-Game Mean Absolute Error")
        ax.set_xlabel("Game (holdout)")
        ax.set_ylabel("MAE")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = self.charts_dir / "rolling_error.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)

    def _plot_feature_importance(self, trainer: ModelTrainer):
        fi = trainer.get_feature_importance()
        if fi is None:
            return
        top = fi.head(30)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.barh(top["feature"][::-1], top["importance"][::-1], color="steelblue")
        ax.set_title("Top 30 Feature Importances")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        out = self.charts_dir / "feature_importance.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        logger.info(f"Chart saved: {out}")

    def _plot_model_comparison(self, cv_results: Optional[pd.DataFrame]):
        if cv_results is None or cv_results.empty:
            return
        summary = cv_results.groupby("model")["MAE"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(summary.index, summary.values, color="steelblue")
        ax.set_title("Model Comparison — Mean Walk-Forward MAE")
        ax.set_xlabel("MAE (points)")
        fig.tight_layout()
        out = self.charts_dir / "model_comparison.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        logger.info(f"Chart saved: {out}")

    def _shap_analysis(self, model, X: pd.DataFrame, feature_cols: List[str]):
        shap_vals = _safe_shap(model, X)
        if shap_vals is None:
            logger.info("Skipping SHAP analysis (not available).")
            return
        try:
            import shap
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_vals, X, feature_names=feature_cols, show=False, max_display=20)
            out = self.charts_dir / "shap_summary.png"
            plt.savefig(out, dpi=120, bbox_inches="tight")
            plt.close()
            logger.info(f"SHAP chart saved: {out}")
        except Exception as e:
            logger.warning(f"SHAP plot failed: {e}")

    # ------------------------------------------------------------------
    # Save metrics CSV
    # ------------------------------------------------------------------

    def _save_metrics(self, metrics: Dict, cv_results: Optional[pd.DataFrame]):
        rows = [{"metric": k, "value": v} for k, v in metrics.items() if not isinstance(v, pd.DataFrame)]
        pd.DataFrame(rows).to_csv(self.reports_dir / "model_metrics.csv", index=False)
        logger.info(f"Metrics saved: {self.reports_dir / 'model_metrics.csv'}")

        if cv_results is not None:
            cv_results.to_csv(self.reports_dir / "cv_results.csv", index=False)
