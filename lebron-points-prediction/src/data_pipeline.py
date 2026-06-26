"""
Data pipeline: orchestrates NBA + odds data fetching, merging, and saving.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np

from src.config import get_config
from src.nba_client import NBAClient
from src.odds_client import OddsClient
from src.utils import normalize_team_name

logger = logging.getLogger("lebron.data_pipeline")


class DataPipeline:
    """Fetch, merge, and persist all raw data needed for feature engineering."""

    def __init__(self):
        self.cfg = get_config()
        self.nba = NBAClient()
        self.odds = OddsClient()

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Full pipeline: fetch all data, merge, save, return master DataFrame."""
        logger.info("Starting data pipeline...")
        game_logs = self._build_player_game_logs()
        team_stats = self._build_team_context(game_logs)
        opp_stats = self._build_opponent_context(game_logs)
        merged = self._merge_all(game_logs, team_stats, opp_stats)
        merged = self._add_odds(merged)
        merged = self._add_rest_features(merged)
        merged = self._add_home_away(merged)
        out_path = self.cfg.processed_path / "master_dataset.parquet"
        merged.to_parquet(out_path, index=False)
        logger.info(f"Master dataset saved: {out_path} ({len(merged)} rows)")
        return merged

    def load_master(self) -> Optional[pd.DataFrame]:
        """Load previously processed master dataset."""
        path = self.cfg.processed_path / "master_dataset.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None

    # ------------------------------------------------------------------
    # NBA data builders
    # ------------------------------------------------------------------

    def _build_player_game_logs(self) -> pd.DataFrame:
        logger.info("Fetching player game logs...")
        df = self.nba.get_player_game_logs()

        # Standardize column names
        rename = {
            "Game_ID": "GAME_ID",
            "MATCHUP": "MATCHUP",
            "WL": "WL",
            "MIN": "MIN",
            "FGM": "FGM",
            "FGA": "FGA",
            "FG_PCT": "FG_PCT",
            "FG3M": "FG3M",
            "FG3A": "FG3A",
            "FG3_PCT": "FG3_PCT",
            "FTM": "FTM",
            "FTA": "FTA",
            "FT_PCT": "FT_PCT",
            "OREB": "OREB",
            "DREB": "DREB",
            "REB": "REB",
            "AST": "AST",
            "STL": "STL",
            "BLK": "BLK",
            "TOV": "TOV",
            "PF": "PF",
            "PTS": "PTS",
            "PLUS_MINUS": "PLUS_MINUS",
        }
        existing = {k: v for k, v in rename.items() if k in df.columns}
        df = df.rename(columns=existing)

        # Parse minutes to float
        if "MIN" in df.columns:
            df["MIN"] = df["MIN"].apply(_parse_minutes)

        # True shooting percentage
        df["TS_PCT"] = df.apply(
            lambda r: _true_shooting(r.get("PTS", 0), r.get("FGA", 0), r.get("FTA", 0)),
            axis=1,
        )

        # eFG%
        df["EFG_PCT"] = df.apply(
            lambda r: _efg(r.get("FGM", 0), r.get("FG3M", 0), r.get("FGA", 1)),
            axis=1,
        )

        # Extract opponent abbreviation from MATCHUP (e.g. "LAL vs. GSW" or "LAL @ BOS")
        if "MATCHUP" in df.columns:
            df["OPP_ABBREV"] = df["MATCHUP"].apply(_parse_opponent)
            df["HOME_GAME"] = df["MATCHUP"].apply(lambda x: 0 if "@" in str(x) else 1)

        df = df.sort_values("GAME_DATE").reset_index(drop=True)
        logger.info(f"Player game logs: {len(df)} games")
        return df

    def _build_team_context(self, game_logs: pd.DataFrame) -> pd.DataFrame:
        """Build a per-season Lakers team stats lookup."""
        logger.info("Fetching Lakers team stats...")
        base = self.nba.get_league_team_stats()
        adv = self.nba.get_league_team_stats(measure="Advanced")

        base_lal = base[base["TEAM_ABBREVIATION"] == self.cfg.team_abbreviation].copy()
        adv_lal = adv[adv["TEAM_ABBREVIATION"] == self.cfg.team_abbreviation].copy()

        merge_keys = ["SEASON", "TEAM_ABBREVIATION"]
        # Rename advanced cols to avoid clashes
        adv_cols = [c for c in adv_lal.columns if c not in merge_keys + ["TEAM_ID", "TEAM_NAME", "GP", "W", "L"]]
        adv_lal = adv_lal[merge_keys + adv_cols].rename(
            columns={c: f"ADV_{c}" for c in adv_cols}
        )

        merged = base_lal.merge(adv_lal, on=merge_keys, how="left")
        return merged

    def _build_opponent_context(self, game_logs: pd.DataFrame) -> pd.DataFrame:
        """Build opponent defensive stats per season."""
        logger.info("Fetching opponent team stats...")
        defense = self.nba.get_league_team_defensive_stats()
        # Rename to OPP_ prefix
        id_cols = ["SEASON", "TEAM_ABBREVIATION", "TEAM_ID", "TEAM_NAME"]
        stat_cols = [c for c in defense.columns if c not in id_cols]
        defense = defense.rename(columns={c: f"OPP_{c}" for c in stat_cols})
        return defense

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    def _merge_all(
        self,
        game_logs: pd.DataFrame,
        team_stats: pd.DataFrame,
        opp_stats: pd.DataFrame,
    ) -> pd.DataFrame:
        logger.info("Merging datasets...")
        df = game_logs.copy()

        # Merge Lakers team stats by season
        team_merge = team_stats[["SEASON"] + [c for c in team_stats.columns if c != "SEASON" and c not in df.columns]].copy()
        df = df.merge(team_merge, on="SEASON", how="left")

        # Merge opponent stats by season + OPP_ABBREV
        if "OPP_ABBREV" in df.columns and "TEAM_ABBREVIATION" in opp_stats.columns:
            opp_merge = opp_stats.rename(columns={"TEAM_ABBREVIATION": "OPP_ABBREV"}).copy()
            opp_merge = opp_merge[["SEASON", "OPP_ABBREV"] + [c for c in opp_merge.columns if c not in ["SEASON", "OPP_ABBREV"] and c not in df.columns]]
            df = df.merge(opp_merge, on=["SEASON", "OPP_ABBREV"], how="left")

        return df.sort_values("GAME_DATE").reset_index(drop=True)

    def _add_odds(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Fetching odds data...")
        if not self.odds._has_key():
            logger.warning("No odds API key — adding NaN odds columns.")
            odds_cols = [
                "LAKERS_SPREAD", "LAKERS_MONEYLINE", "OPP_MONEYLINE",
                "over_under", "PLAYER_POINTS_LINE",
                "PLAYER_POINTS_OVER_PRICE", "PLAYER_POINTS_UNDER_PRICE",
            ]
            for col in odds_cols:
                df[col] = float("nan")
            return df

        dates = df["GAME_DATE"].dt.strftime("%Y-%m-%d").unique().tolist()
        odds_df = self.odds.match_odds_to_nba_games(dates)
        odds_df["GAME_DATE"] = pd.to_datetime(odds_df["GAME_DATE"])

        # Select only odds columns to merge
        keep = [c for c in odds_df.columns if c not in df.columns or c == "GAME_DATE"]
        df = df.merge(odds_df[keep], on="GAME_DATE", how="left")
        return df

    def _add_rest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("GAME_DATE").reset_index(drop=True)
        df["DAYS_REST"] = df["GAME_DATE"].diff().dt.days.fillna(3).clip(upper=14)
        df["BACK_TO_BACK"] = (df["DAYS_REST"] == 1).astype(int)

        # Third game in 4 nights
        df["THIRD_IN_FOUR"] = 0
        for i in range(2, len(df)):
            span = (df.loc[i, "GAME_DATE"] - df.loc[i - 2, "GAME_DATE"]).days
            if span <= 3:
                df.loc[i, "THIRD_IN_FOUR"] = 1

        # Fourth game in 6 nights
        df["FOURTH_IN_SIX"] = 0
        for i in range(3, len(df)):
            span = (df.loc[i, "GAME_DATE"] - df.loc[i - 3, "GAME_DATE"]).days
            if span <= 5:
                df.loc[i, "FOURTH_IN_SIX"] = 1

        return df

    def _add_home_away(self, df: pd.DataFrame) -> pd.DataFrame:
        if "HOME_GAME" not in df.columns:
            df["HOME_GAME"] = 0
        df["AWAY_GAME"] = 1 - df["HOME_GAME"]
        return df


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_minutes(val) -> float:
    try:
        if isinstance(val, (int, float)):
            return float(val)
        parts = str(val).split(":")
        return float(parts[0]) + float(parts[1]) / 60 if len(parts) == 2 else float(parts[0])
    except Exception:
        return float("nan")


def _true_shooting(pts: float, fga: float, fta: float) -> float:
    denom = 2 * (fga + 0.44 * fta)
    return pts / denom if denom > 0 else 0.0


def _efg(fgm: float, fg3m: float, fga: float) -> float:
    return (fgm + 0.5 * fg3m) / fga if fga > 0 else 0.0


def _parse_opponent(matchup: str) -> str:
    """Extract opponent abbreviation from matchup string like 'LAL vs. GSW' or 'LAL @ BOS'."""
    try:
        parts = str(matchup).split()
        return normalize_team_name(parts[-1])
    except Exception:
        return "UNK"
