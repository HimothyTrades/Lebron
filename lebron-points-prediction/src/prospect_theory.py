"""
Prospect Theory-inspired behavioral features.

Prospect Theory (Kahneman & Tversky 1979) suggests decision-makers evaluate
outcomes relative to a reference point and are more sensitive to losses than
equivalent gains (loss aversion). Applied here: LeBron's scoring behavior may
shift based on recent performance relative to expectations.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

from src.config import get_config
from src.utils import z_score_series

logger = logging.getLogger("lebron.prospect_theory")


class ProspectTheoryFeatures:
    """Compute behavioral/psychological features inspired by Prospect Theory."""

    def __init__(self):
        self.cfg = get_config()
        self.weights: Dict[str, float] = self.cfg.prospect_weights

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all Prospect Theory features to DataFrame."""
        logger.info("Building Prospect Theory features...")
        df = df.sort_values("GAME_DATE").reset_index(drop=True)

        df = self._reference_points(df)
        df = self._gains_losses(df)
        df = self._win_loss_context(df)
        df = self._aggressiveness_changes(df)
        df = self._composite_index(df)

        return df

    # ------------------------------------------------------------------
    # Reference points
    # ------------------------------------------------------------------

    def _reference_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute multiple reference points for scoring expectations."""
        pts = df["PTS"]

        # Season average up to (but not including) the current game
        df["SEASON_AVG_PTS"] = (
            pts.shift(1)
            .groupby(df["SEASON"])
            .transform(lambda x: x.expanding().mean())
        )

        # Rolling 10-game average (lagged)
        df["ROLL10_PTS_REF"] = pts.shift(1).rolling(10, min_periods=3).mean()

        # Previous game's Vegas player prop line as reference (if available)
        if "PLAYER_POINTS_LINE" in df.columns:
            df["PREV_PROP_LINE"] = df["PLAYER_POINTS_LINE"].shift(1)
        else:
            df["PREV_PROP_LINE"] = np.nan

        return df

    # ------------------------------------------------------------------
    # Gains and losses relative to reference
    # ------------------------------------------------------------------

    def _gains_losses(self, df: pd.DataFrame) -> pd.DataFrame:
        prev_pts = df["PTS"].shift(1)
        ref = df["ROLL10_PTS_REF"]

        # Performance gap: how much previous game deviated from rolling avg
        df["PREV_GAME_PTS_GAP"] = prev_pts - ref

        df["PREV_ABOVE_AVG"] = (df["PREV_GAME_PTS_GAP"] > 0).astype(int)
        df["PREV_BELOW_AVG"] = (df["PREV_GAME_PTS_GAP"] < 0).astype(int)

        # Loss aversion score: weighted by how far below expectation
        df["SCORING_DISAPPOINTMENT"] = np.maximum(0, ref - prev_pts)
        df["SCORING_OVERPERFORMANCE"] = np.maximum(0, prev_pts - ref)

        # Z-scores for use in composite
        df["SCORING_DISAPPOINTMENT_Z"] = z_score_series(df["SCORING_DISAPPOINTMENT"].fillna(0))
        df["SCORING_OVERPERFORMANCE_Z"] = z_score_series(df["SCORING_OVERPERFORMANCE"].fillna(0))

        # Expectation gap vs Vegas prop
        if "PREV_PROP_LINE" in df.columns:
            df["BELOW_PROP_LAST_GAME"] = (prev_pts < df["PREV_PROP_LINE"]).astype(int)
            df["ABOVE_PROP_LAST_GAME"] = (prev_pts > df["PREV_PROP_LINE"]).astype(int)
        else:
            df["BELOW_PROP_LAST_GAME"] = 0
            df["ABOVE_PROP_LAST_GAME"] = 0

        # Loss aversion composite: scoring disappointment creates higher urgency
        df["LOSS_AVERSION_SCORE"] = df["SCORING_DISAPPOINTMENT_Z"] * 2.25

        return df

    # ------------------------------------------------------------------
    # Win/loss context
    # ------------------------------------------------------------------

    def _win_loss_context(self, df: pd.DataFrame) -> pd.DataFrame:
        if "WIN" not in df.columns:
            df["WIN"] = np.nan
            df["COMING_OFF_LOSS"] = 0
            df["COMING_OFF_WIN"] = 0
            df["COMING_OFF_TWO_LOSSES"] = 0
            df["COMING_OFF_TWO_WINS"] = 0
            df["BLOWOUT_LOSS"] = 0
            df["BLOWOUT_WIN"] = 0
            df["CLOSE_LOSS_FLAG"] = 0
            return df

        win = df["WIN"]
        df["COMING_OFF_LOSS"] = (win.shift(1) == 0).astype(int)
        df["COMING_OFF_WIN"] = (win.shift(1) == 1).astype(int)

        df["COMING_OFF_TWO_LOSSES"] = (
            (win.shift(1) == 0) & (win.shift(2) == 0)
        ).astype(int)
        df["COMING_OFF_TWO_WINS"] = (
            (win.shift(1) == 1) & (win.shift(2) == 1)
        ).astype(int)

        # Blowout: plus_minus > 15 or < -15
        pm = df["PLUS_MINUS"].shift(1) if "PLUS_MINUS" in df.columns else pd.Series(np.nan, index=df.index)
        df["BLOWOUT_LOSS"] = ((win.shift(1) == 0) & (pm < -15)).astype(int)
        df["BLOWOUT_WIN"] = ((win.shift(1) == 1) & (pm > 15)).astype(int)

        # Close loss: lost by <=5
        df["CLOSE_LOSS_FLAG"] = ((win.shift(1) == 0) & (pm > -6) & (pm < 0)).astype(int)

        # Streak features
        df["LOSING_STREAK"] = self._streak(win, streak_type=0)
        df["WINNING_STREAK"] = self._streak(win, streak_type=1)

        return df

    def _streak(self, series: pd.Series, streak_type: int) -> pd.Series:
        """Count consecutive occurrences of streak_type (0=loss, 1=win) up to previous game."""
        streak = pd.Series(0, index=series.index)
        count = 0
        for i in range(len(series)):
            if i == 0:
                streak.iloc[i] = 0
                count = 0
            else:
                prev = series.iloc[i - 1]
                if prev == streak_type:
                    count += 1
                else:
                    count = 0
                streak.iloc[i] = count
        return streak

    # ------------------------------------------------------------------
    # Shot aggressiveness changes
    # ------------------------------------------------------------------

    def _aggressiveness_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        After a poor scoring game, does LeBron take more shots?
        Compare rolling 3-game FGA/3PA/FTA trend to rolling 10-game baseline.
        """
        fga_roll3 = df["FGA"].shift(1).rolling(3, min_periods=1).mean()
        fga_roll10 = df["FGA"].shift(1).rolling(10, min_periods=3).mean()
        df["SHOT_AGGRESSIVENESS_CHANGE"] = fga_roll3 - fga_roll10

        fga3_roll3 = df["FG3A"].shift(1).rolling(3, min_periods=1).mean()
        fga3_roll10 = df["FG3A"].shift(1).rolling(10, min_periods=3).mean()
        df["THREE_POINT_AGGRESSIVENESS_CHANGE"] = fga3_roll3 - fga3_roll10

        fta_roll3 = df["FTA"].shift(1).rolling(3, min_periods=1).mean()
        fta_roll10 = df["FTA"].shift(1).rolling(10, min_periods=3).mean()
        df["FT_AGGRESSIVENESS_CHANGE"] = fta_roll3 - fta_roll10

        df["SHOT_AGG_CHANGE_Z"] = z_score_series(df["SHOT_AGGRESSIVENESS_CHANGE"].fillna(0))

        # Usage increase after loss
        if "COMING_OFF_LOSS" in df.columns:
            usg_diff = df.get("ROLL5_USG_PROXY", pd.Series(np.nan, index=df.index)) - \
                       df.get("ROLL20_USG_PROXY", pd.Series(np.nan, index=df.index))
            df["USG_INCREASE_AFTER_LOSS"] = df["COMING_OFF_LOSS"] * usg_diff.fillna(0)
        else:
            df["USG_INCREASE_AFTER_LOSS"] = 0.0

        return df

    # ------------------------------------------------------------------
    # Composite Prospect Theory Index
    # ------------------------------------------------------------------

    def _composite_index(self, df: pd.DataFrame) -> pd.DataFrame:
        w = self.weights
        df["PROSPECT_THEORY_INDEX"] = (
            w.get("scoring_disappointment_z", 0.30) * df["SCORING_DISAPPOINTMENT_Z"].fillna(0)
            + w.get("coming_off_loss", 0.25) * df["COMING_OFF_LOSS"].fillna(0)
            + w.get("usage_increase_after_loss", 0.20) * df["USG_INCREASE_AFTER_LOSS"].fillna(0)
            + w.get("shot_aggressiveness_change_z", 0.15) * df["SHOT_AGG_CHANGE_Z"].fillna(0)
            + w.get("close_loss_flag", 0.10) * df["CLOSE_LOSS_FLAG"].fillna(0)
        )
        logger.info("Prospect Theory Index computed.")
        return df


def add_prospect_theory_features(df: pd.DataFrame) -> pd.DataFrame:
    return ProspectTheoryFeatures().run(df)
