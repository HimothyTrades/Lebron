"""
Pressure Index: quantifies game-context pressure that may affect performance.

Combines playoff context, spread tightness, opponent strength, schedule position,
national TV games, rivalry matchups, and recent losing streaks into a single index.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

from src.config import get_config

logger = logging.getLogger("lebron.pressure_index")

# Christmas game dates (used across seasons; month-day format)
CHRISTMAS_MONTH_DAY = "12-25"

# National TV placeholder teams — games vs. historically top-market teams
# are treated as likely national TV games until a proper data source is available
NATIONAL_TV_PROXY_TEAMS = {"GSW", "BOS", "MIA", "NYK", "CHI", "DAL", "LAC"}


class PressureIndexFeatures:
    """Compute pressure-index features for each game."""

    def __init__(self):
        self.cfg = get_config()
        self.weights: Dict[str, float] = self.cfg.pressure_weights
        self.rivalry_teams = set(self.cfg.rivalry_teams)
        self.strong_opp_threshold = self.cfg.strong_opp_threshold
        self.close_spread_threshold = self.cfg.close_spread_threshold
        self.high_total_threshold = self.cfg.high_total_threshold

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Building Pressure Index features...")
        df = df.sort_values("GAME_DATE").reset_index(drop=True)

        df = self._playoff_flags(df)
        df = self._game_context_flags(df)
        df = self._opponent_strength_flags(df)
        df = self._schedule_pressure(df)
        df = self._national_tv_proxy(df)
        df = self._composite_index(df)

        return df

    # ------------------------------------------------------------------
    # Playoff & elimination flags
    # ------------------------------------------------------------------

    def _playoff_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        df["PLAYOFF_FLAG"] = df.get("PLAYOFF", pd.Series(False, index=df.index)).astype(int)
        # Elimination game flag: requires external data; placeholder = 0
        df["ELIMINATION_GAME"] = 0
        logger.debug("NOTE: ELIMINATION_GAME is a placeholder — enrich with playoff series data.")
        return df

    # ------------------------------------------------------------------
    # Game context flags
    # ------------------------------------------------------------------

    def _game_context_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        # Christmas game
        if "GAME_DATE" in df.columns:
            df["CHRISTMAS_GAME"] = (
                df["GAME_DATE"].dt.strftime("%m-%d") == CHRISTMAS_MONTH_DAY
            ).astype(int)
        else:
            df["CHRISTMAS_GAME"] = 0

        # Rivalry game
        if "OPP_ABBREV" in df.columns:
            df["RIVALRY_FLAG"] = df["OPP_ABBREV"].isin(self.rivalry_teams).astype(int)
        else:
            df["RIVALRY_FLAG"] = 0

        # Late season (games 60+ of regular season, roughly March-April)
        if "GAME_DATE" in df.columns:
            df["LATE_SEASON_FLAG"] = (
                (df["GAME_DATE"].dt.month >= 3) & (~df["PLAYOFF_FLAG"].astype(bool))
            ).astype(int)
        else:
            df["LATE_SEASON_FLAG"] = 0

        return df

    # ------------------------------------------------------------------
    # Opponent strength
    # ------------------------------------------------------------------

    def _opponent_strength_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        # Use opponent offensive rating as strength proxy
        # (strong offense = harder game for LeBron's team = more pressure)
        opp_off = df.get("OPP_OFF_RATING", df.get("OPP_DEF_RATING", pd.Series(np.nan, index=df.index)))
        df["STRONG_OPP_FLAG"] = (opp_off > self.strong_opp_threshold).astype(int)

        # Close spread = competitive game
        spread = df.get("LAKERS_SPREAD", pd.Series(np.nan, index=df.index))
        df["CLOSE_SPREAD_FLAG"] = (spread.abs() <= self.close_spread_threshold).astype(int)

        # High total = fast-paced, high-scoring expected game
        total = df.get("over_under", pd.Series(np.nan, index=df.index))
        df["HIGH_TOTAL_FLAG"] = (total > self.high_total_threshold).astype(int)

        # Must-win proxy: large losing streak
        losing_streak = df.get("LOSING_STREAK", pd.Series(0, index=df.index))
        df["MUST_WIN_PROXY"] = (losing_streak >= 3).astype(int)

        return df

    # ------------------------------------------------------------------
    # Schedule pressure
    # ------------------------------------------------------------------

    def _schedule_pressure(self, df: pd.DataFrame) -> pd.DataFrame:
        # Losing streak flag
        losing_streak = df.get("LOSING_STREAK", pd.Series(0, index=df.index))
        df["LOSING_STREAK_FLAG"] = (losing_streak >= 2).astype(int)

        # Close score environment from closing spread
        spread = df.get("LAKERS_SPREAD", pd.Series(np.nan, index=df.index))
        df["CLOSE_SCORE_ENV"] = (spread.abs() < 3).astype(int)

        # High leverage: playoff + close spread
        df["HIGH_LEVERAGE_FLAG"] = (
            (df["PLAYOFF_FLAG"] == 1) & (df.get("CLOSE_SPREAD_FLAG", 0) == 1)
        ).astype(int)

        return df

    # ------------------------------------------------------------------
    # National TV proxy
    # ------------------------------------------------------------------

    def _national_tv_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        PLACEHOLDER: Real national TV data requires an external schedule source.
        Proxy: games vs. historically high-TV-market teams are flagged.
        Replace this column with actual broadcast data when available.
        """
        if "OPP_ABBREV" in df.columns:
            df["NATIONAL_TV_FLAG"] = df["OPP_ABBREV"].isin(NATIONAL_TV_PROXY_TEAMS).astype(int)
        else:
            df["NATIONAL_TV_FLAG"] = 0

        logger.debug(
            "NOTE: NATIONAL_TV_FLAG uses team-market proxy. "
            "Enrich with actual broadcast schedule for better accuracy."
        )
        return df

    # ------------------------------------------------------------------
    # Composite Pressure Index
    # ------------------------------------------------------------------

    def _composite_index(self, df: pd.DataFrame) -> pd.DataFrame:
        w = self.weights
        df["PRESSURE_INDEX"] = (
            w.get("playoff_flag", 0.20) * df["PLAYOFF_FLAG"].fillna(0)
            + w.get("close_spread_flag", 0.15) * df["CLOSE_SPREAD_FLAG"].fillna(0)
            + w.get("strong_opponent_flag", 0.15) * df["STRONG_OPP_FLAG"].fillna(0)
            + w.get("late_season_flag", 0.10) * df["LATE_SEASON_FLAG"].fillna(0)
            + w.get("losing_streak_flag", 0.10) * df["LOSING_STREAK_FLAG"].fillna(0)
            + w.get("national_tv_flag", 0.10) * df["NATIONAL_TV_FLAG"].fillna(0)
            + w.get("rivalry_flag", 0.10) * df["RIVALRY_FLAG"].fillna(0)
            + w.get("high_total_flag", 0.10) * df["HIGH_TOTAL_FLAG"].fillna(0)
        )

        # Cross interactions with other indices
        pt_index = df.get("PROSPECT_THEORY_INDEX", pd.Series(0.0, index=df.index))
        df["PRESSURE_X_PROSPECT"] = df["PRESSURE_INDEX"] * pt_index.fillna(0)
        df["PRESSURE_X_OPP_DEF"] = df["PRESSURE_INDEX"] * df.get(
            "OPP_DEF_RATING", pd.Series(0.0, index=df.index)
        ).fillna(0)
        df["FATIGUE_X_PRESSURE"] = df.get("BACK_TO_BACK", pd.Series(0, index=df.index)) * df["PRESSURE_INDEX"]
        df["HOME_X_PRESSURE"] = df.get("HOME_GAME", pd.Series(0, index=df.index)) * df["PRESSURE_INDEX"]

        logger.info("Pressure Index computed.")
        return df


def add_pressure_index_features(df: pd.DataFrame) -> pd.DataFrame:
    return PressureIndexFeatures().run(df)
