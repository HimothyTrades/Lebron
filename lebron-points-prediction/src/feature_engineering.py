"""
Feature engineering: player performance, rolling stats, EWMA, opponent context,
team context, rest/fatigue, home/away, interaction features, Vegas features.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from src.config import get_config
from src.utils import safe_divide, z_score_series

logger = logging.getLogger("lebron.features")


class FeatureEngineer:
    """Build all model features from the master dataset."""

    def __init__(self):
        self.cfg = get_config()
        self.windows = self.cfg.rolling_windows
        self.ewma_spans = self.cfg.ewma_spans

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature groups in order. Returns enriched DataFrame."""
        logger.info(f"Engineering features on {len(df)} rows...")
        df = df.sort_values("GAME_DATE").reset_index(drop=True)

        df = self._player_base_features(df)
        df = self._rolling_features(df)
        df = self._ewma_features(df)
        df = self._opportunity_features(df)
        df = self._opponent_features(df)
        df = self._team_context_features(df)
        df = self._vegas_features(df)
        df = self._interaction_features(df)

        out_path = self.cfg.processed_path / "features.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"Features saved: {out_path} ({df.shape[1]} columns)")
        return df

    # ------------------------------------------------------------------
    # Player base features
    # ------------------------------------------------------------------

    def _player_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all base player stat columns exist with clean types."""
        float_cols = [
            "PTS", "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
            "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "STL",
            "BLK", "TOV", "PLUS_MINUS", "TS_PCT", "EFG_PCT",
        ]
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = np.nan

        # Usage rate proxy: USG = (FGA + 0.44*FTA + TOV) / MIN * 100 / 5
        df["USG_PROXY"] = df.apply(
            lambda r: safe_divide(
                (r["FGA"] + 0.44 * r["FTA"] + r["TOV"]) * 100,
                r["MIN"] * 5 if r["MIN"] > 0 else 1,
            ),
            axis=1,
        )

        # Win/loss numeric
        if "WL" in df.columns:
            df["WIN"] = (df["WL"] == "W").astype(int)
        else:
            df["WIN"] = np.nan

        return df

    # ------------------------------------------------------------------
    # Rolling features (shift by 1 to prevent data leakage)
    # ------------------------------------------------------------------

    def _rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        stat_cols = [
            "PTS", "MIN", "FGA", "FG3A", "FTA", "FG_PCT", "FG3_PCT", "FT_PCT",
            "USG_PROXY", "TS_PCT", "AST", "REB", "TOV", "PLUS_MINUS", "EFG_PCT",
        ]
        for window in self.windows:
            for col in stat_cols:
                if col not in df.columns:
                    continue
                # Shift by 1 — use only PREVIOUS games, never current game
                shifted = df[col].shift(1)
                df[f"ROLL{window}_{col}"] = shifted.rolling(window, min_periods=1).mean()

            # Rolling std for points
            shifted_pts = df["PTS"].shift(1)
            df[f"ROLL{window}_PTS_STD"] = shifted_pts.rolling(window, min_periods=2).std()
            df[f"ROLL{window}_PTS_VOLATILITY"] = df[f"ROLL{window}_PTS_STD"] / (
                df[f"ROLL{window}_PTS"].replace(0, np.nan)
            )

            # Z-score within rolling window
            def rolling_zscore(series: pd.Series, w: int) -> pd.Series:
                return (series - series.rolling(w, min_periods=2).mean()) / (
                    series.rolling(w, min_periods=2).std().replace(0, np.nan)
                )

            df[f"ROLL{window}_PTS_ZSCORE"] = rolling_zscore(df["PTS"].shift(1), window)
            df[f"ROLL{window}_USG_ZSCORE"] = rolling_zscore(df["USG_PROXY"].shift(1), window)

        return df

    # ------------------------------------------------------------------
    # EWMA features
    # ------------------------------------------------------------------

    def _ewma_features(self, df: pd.DataFrame) -> pd.DataFrame:
        ewma_cols = ["PTS", "MIN", "USG_PROXY", "FGA", "TS_PCT"]
        for span in self.ewma_spans:
            for col in ewma_cols:
                if col not in df.columns:
                    continue
                # Shift by 1 to avoid leakage
                df[f"EWMA{span}_{col}"] = (
                    df[col].shift(1).ewm(span=span, adjust=False).mean()
                )
        return df

    # ------------------------------------------------------------------
    # Opportunity features
    # ------------------------------------------------------------------

    def _opportunity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Last game minutes
        df["LAST_GAME_MIN"] = df["MIN"].shift(1)

        # Average minutes over recent windows
        for w in [5, 10]:
            df[f"AVG_MIN_L{w}"] = df["MIN"].shift(1).rolling(w, min_periods=1).mean()

        # Usage trend: rolling 5 minus rolling 20
        if "ROLL5_USG_PROXY" in df.columns and "ROLL20_USG_PROXY" in df.columns:
            df["USG_TREND"] = df["ROLL5_USG_PROXY"] - df["ROLL20_USG_PROXY"]

        # Shot volume trend
        if "ROLL5_FGA" in df.columns and "ROLL20_FGA" in df.columns:
            df["SHOT_VOLUME_TREND"] = df["ROLL5_FGA"] - df["ROLL20_FGA"]

        # Games missed before this game (gap in calendar days > 2 means likely missed)
        df["GAMES_MISSED_BEFORE"] = (df["DAYS_REST"].fillna(1) > 2).astype(int)

        return df

    # ------------------------------------------------------------------
    # Opponent defense features
    # ------------------------------------------------------------------

    def _opponent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map opponent stats from merged league data (prefixed OPP_)."""
        opp_map = {
            "OPP_DEF_RATING": ["OPP_DEF_RATING", "OPP_E_DEF_RATING"],
            "OPP_PACE": ["OPP_PACE", "OPP_E_PACE"],
            "OPP_FG_PCT": ["OPP_OPP_FG_PCT", "OPP_FG_PCT_ALLOWED"],
            "OPP_FG3_PCT": ["OPP_OPP_FG3_PCT", "OPP_FG3_PCT_ALLOWED"],
            "OPP_PTS": ["OPP_OPP_PTS", "OPP_POINTS_ALLOWED"],
        }
        for target, candidates in opp_map.items():
            if target not in df.columns:
                for c in candidates:
                    if c in df.columns:
                        df[target] = df[c]
                        break
                else:
                    df[target] = np.nan

        # Pace-adjusted opportunity
        if "OPP_PACE" in df.columns:
            df["PACE_OPP_PRODUCT"] = df.get("OPP_PACE", pd.Series(np.nan, index=df.index))

        return df

    # ------------------------------------------------------------------
    # Team context features
    # ------------------------------------------------------------------

    def _team_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Try to map columns from merged team data
        team_map = {
            "LAL_OFF_RATING": ["OFF_RATING", "E_OFF_RATING", "ADV_OFF_RATING"],
            "LAL_PACE": ["PACE", "E_PACE", "ADV_PACE"],
            "LAL_NET_RATING": ["NET_RATING", "E_NET_RATING", "ADV_NET_RATING"],
        }
        for target, candidates in team_map.items():
            if target not in df.columns:
                for c in candidates:
                    if c in df.columns:
                        df[target] = df[c]
                        break
                else:
                    df[target] = np.nan

        # Rolling team context (last 5 games)
        for col in ["LAL_OFF_RATING", "LAL_PACE"]:
            if col in df.columns:
                df[f"ROLL5_{col}"] = df[col].shift(1).rolling(5, min_periods=1).mean()

        return df

    # ------------------------------------------------------------------
    # Vegas features
    # ------------------------------------------------------------------

    def _vegas_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Ensure source columns exist
        for col in ["LAKERS_SPREAD", "over_under", "LAKERS_MONEYLINE", "OPP_MONEYLINE",
                    "PLAYER_POINTS_LINE"]:
            if col not in df.columns:
                df[col] = np.nan

        spread = pd.to_numeric(df["LAKERS_SPREAD"], errors="coerce")
        total = pd.to_numeric(df["over_under"], errors="coerce")
        ml_lal = pd.to_numeric(df["LAKERS_MONEYLINE"], errors="coerce")
        ml_opp = pd.to_numeric(df["OPP_MONEYLINE"], errors="coerce")

        # Implied team totals
        lakers_fav = spread < 0
        df["IMPLIED_TEAM_TOTAL"] = np.where(
            lakers_fav,
            (total / 2) + (spread.abs() / 2),
            (total / 2) - (spread.abs() / 2),
        )
        df["IMPLIED_OPP_TOTAL"] = np.where(
            lakers_fav,
            (total / 2) - (spread.abs() / 2),
            (total / 2) + (spread.abs() / 2),
        )

        # Implied win probability from American moneyline
        df["IMPLIED_WIN_PROB"] = ml_lal.apply(_american_to_implied_prob)
        df["FAVORITE_FLAG"] = (spread < 0).astype(int)
        df["UNDERDOG_FLAG"] = (spread > 0).astype(int)

        # Blowout risk: large spread indicates likely blowout
        df["BLOWOUT_RISK"] = (spread.abs() > 10).astype(int)

        # Player prop
        df["PLAYER_POINTS_LINE"] = pd.to_numeric(df["PLAYER_POINTS_LINE"], errors="coerce")

        return df

    # ------------------------------------------------------------------
    # Interaction features
    # ------------------------------------------------------------------

    def _interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        _get = lambda col: df.get(col, pd.Series(np.nan, index=df.index))

        df["PACE_X_MIN"] = _get("OPP_PACE") * _get("LAST_GAME_MIN")
        df["PACE_X_USG"] = _get("OPP_PACE") * _get("ROLL5_USG_PROXY")
        df["OPP_DEF_X_USG"] = _get("OPP_DEF_RATING") * _get("ROLL5_USG_PROXY")
        df["REST_X_AGE"] = _get("DAYS_REST") * 39  # LeBron age proxy
        df["SPREAD_X_MIN"] = _get("LAKERS_SPREAD").abs() * _get("LAST_GAME_MIN")
        df["BLOWOUT_X_MIN"] = _get("BLOWOUT_RISK") * _get("AVG_MIN_L5")
        df["TEAM_TOTAL_X_USG"] = _get("IMPLIED_TEAM_TOTAL") * _get("ROLL5_USG_PROXY")
        df["HOME_X_OPP_DEF"] = _get("HOME_GAME") * _get("OPP_DEF_RATING")
        df["FATIGUE_X_BACK2BACK"] = _get("BACK_TO_BACK") * _get("AVG_MIN_L5")

        return df


# ------------------------------------------------------------------
# Module-level convenience
# ------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return FeatureEngineer().run(df)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _american_to_implied_prob(ml) -> float:
    """Convert American moneyline odds to implied probability."""
    try:
        ml = float(ml)
        if ml > 0:
            return 100 / (ml + 100)
        else:
            return abs(ml) / (abs(ml) + 100)
    except Exception:
        return float("nan")
