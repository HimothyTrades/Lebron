"""
NBA data client wrapping nba_api endpoints.
Handles rate limiting, retries, and local JSON caching.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from nba_api.stats.endpoints import (
    boxscoreadvancedv2,
    boxscoretraditionalv2,
    commonplayerinfo,
    leaguedashplayerstats,
    leaguedashteamstats,
    leaguegamefinder,
    playergamelog,
    teamgamelog,
)
from nba_api.stats.static import players, teams

from src.config import get_config
from src.utils import cache_path, load_cache, retry, save_cache, setup_logging

logger = logging.getLogger("lebron.nba_client")

# nba_api throttles hard — wait at least this many seconds between calls
_API_SLEEP: float = 0.8


def _sleep():
    time.sleep(_API_SLEEP)


class NBAClient:
    """Pulls NBA data via nba_api with caching and retry logic."""

    def __init__(self):
        self.cfg = get_config()
        self.cache_dir = self.cfg.raw_nba_path
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cached(self, key: str) -> Optional[Any]:
        p = cache_path(self.cache_dir, key)
        return load_cache(p, self.cfg.cache_ttl_hours) if self.cfg.cache_enabled else None

    def _store(self, key: str, data: Any) -> None:
        if self.cfg.cache_enabled:
            save_cache(cache_path(self.cache_dir, key), data)

    @retry(max_attempts=5, initial_delay=3.0, backoff=2.0, exceptions=(Exception,))
    def _fetch(self, endpoint_cls, cache_key: str, **kwargs) -> pd.DataFrame:
        cached = self._cached(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit: {cache_key}")
            return pd.DataFrame(cached)
        logger.info(f"Fetching {endpoint_cls.__name__} (key={cache_key})")
        _sleep()
        result = endpoint_cls(**kwargs)
        df = result.get_data_frames()[0]
        self._store(cache_key, df.to_dict(orient="records"))
        return df

    # ------------------------------------------------------------------
    # Player info
    # ------------------------------------------------------------------

    def get_player_id(self, player_name: str = None) -> int:
        """Return nba_api player_id for the configured player."""
        name = player_name or self.cfg.player_name
        matches = players.find_players_by_full_name(name)
        if not matches:
            raise ValueError(f"Player not found: {name}")
        return matches[0]["id"]

    def get_player_info(self, player_id: int = None) -> pd.DataFrame:
        pid = player_id or self.cfg.player_id
        key = f"player_info_{pid}"
        return self._fetch(commonplayerinfo.CommonPlayerInfo, key, player_id=pid)

    # ------------------------------------------------------------------
    # Game logs
    # ------------------------------------------------------------------

    def get_player_game_logs(
        self, player_id: int = None, season: str = None
    ) -> pd.DataFrame:
        """Fetch per-game stats for a player in a season."""
        pid = player_id or self.cfg.player_id
        seasons = [season] if season else self.cfg.seasons
        frames: List[pd.DataFrame] = []
        for s in seasons:
            key = f"player_gamelog_{pid}_{s}"
            df = self._fetch(
                playergamelog.PlayerGameLog,
                key,
                player_id=pid,
                season=s,
                season_type_all_star="Regular Season",
            )
            df["SEASON"] = s
            frames.append(df)
            # Also pull playoffs
            key_po = f"player_gamelog_{pid}_{s}_playoffs"
            try:
                df_po = self._fetch(
                    playergamelog.PlayerGameLog,
                    key_po,
                    player_id=pid,
                    season=s,
                    season_type_all_star="Playoffs",
                )
                df_po["SEASON"] = s
                df_po["PLAYOFF"] = True
                frames.append(df_po)
            except Exception as e:
                logger.warning(f"No playoff logs for {s}: {e}")
        combined = pd.concat(frames, ignore_index=True)
        combined["GAME_DATE"] = pd.to_datetime(combined["GAME_DATE"], format="%b %d, %Y", errors="coerce")
        combined = combined.sort_values("GAME_DATE").reset_index(drop=True)
        combined = combined.drop_duplicates(subset=["Game_ID"]).reset_index(drop=True)
        if "PLAYOFF" not in combined.columns:
            combined["PLAYOFF"] = False
        else:
            combined["PLAYOFF"] = combined["PLAYOFF"].fillna(False)
        return combined

    def get_team_game_logs(
        self, team_id: int = None, season: str = None
    ) -> pd.DataFrame:
        """Fetch per-game stats for a team in a season."""
        tid = team_id or self.cfg.team_id
        seasons = [season] if season else self.cfg.seasons
        frames: List[pd.DataFrame] = []
        for s in seasons:
            key = f"team_gamelog_{tid}_{s}"
            df = self._fetch(
                teamgamelog.TeamGameLog,
                key,
                team_id=tid,
                season=s,
                season_type_all_star="Regular Season",
            )
            df["SEASON"] = s
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True)
        combined["GAME_DATE"] = pd.to_datetime(combined["GAME_DATE"], format="%b %d, %Y", errors="coerce")
        return combined.sort_values("GAME_DATE").reset_index(drop=True)

    # ------------------------------------------------------------------
    # League-wide stats
    # ------------------------------------------------------------------

    def get_league_player_stats(
        self, season: str = None, per_mode: str = "PerGame"
    ) -> pd.DataFrame:
        seasons = [season] if season else self.cfg.seasons
        frames = []
        for s in seasons:
            key = f"league_player_stats_{s}_{per_mode}"
            df = self._fetch(
                leaguedashplayerstats.LeagueDashPlayerStats,
                key,
                season=s,
                per_mode_simple=per_mode,
                measure_type_simple_display="Base",
            )
            df["SEASON"] = s
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def get_league_player_advanced(self, season: str = None) -> pd.DataFrame:
        seasons = [season] if season else self.cfg.seasons
        frames = []
        for s in seasons:
            key = f"league_player_adv_{s}"
            df = self._fetch(
                leaguedashplayerstats.LeagueDashPlayerStats,
                key,
                season=s,
                per_mode_simple="PerGame",
                measure_type_simple_display="Advanced",
            )
            df["SEASON"] = s
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def get_league_team_stats(
        self, season: str = None, measure: str = "Base"
    ) -> pd.DataFrame:
        seasons = [season] if season else self.cfg.seasons
        frames = []
        for s in seasons:
            key = f"league_team_stats_{s}_{measure}"
            df = self._fetch(
                leaguedashteamstats.LeagueDashTeamStats,
                key,
                season=s,
                per_mode_simple="PerGame",
                measure_type_simple_display=measure,
            )
            df["SEASON"] = s
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def get_league_team_defensive_stats(self, season: str = None) -> pd.DataFrame:
        """Opponent stats = defensive stats proxy."""
        seasons = [season] if season else self.cfg.seasons
        frames = []
        for s in seasons:
            key = f"league_team_opp_{s}"
            df = self._fetch(
                leaguedashteamstats.LeagueDashTeamStats,
                key,
                season=s,
                per_mode_simple="PerGame",
                measure_type_simple_display="Opponent",
            )
            df["SEASON"] = s
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Box scores
    # ------------------------------------------------------------------

    def get_boxscore_traditional(self, game_id: str) -> Dict[str, pd.DataFrame]:
        key = f"boxscore_trad_{game_id}"
        cached = self._cached(key)
        if cached:
            return {k: pd.DataFrame(v) for k, v in cached.items()}
        _sleep()
        result = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        frames = result.get_data_frames()
        data = {"players": frames[0], "starters": frames[1], "team": frames[2]}
        self._store(key, {k: v.to_dict(orient="records") for k, v in data.items()})
        return data

    def get_boxscore_advanced(self, game_id: str) -> Dict[str, pd.DataFrame]:
        key = f"boxscore_adv_{game_id}"
        cached = self._cached(key)
        if cached:
            return {k: pd.DataFrame(v) for k, v in cached.items()}
        _sleep()
        result = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
        frames = result.get_data_frames()
        data = {"players": frames[0], "team": frames[1]}
        self._store(key, {k: v.to_dict(orient="records") for k, v in data.items()})
        return data

    # ------------------------------------------------------------------
    # Schedule / game finder
    # ------------------------------------------------------------------

    def get_lakers_schedule(self, season: str = None) -> pd.DataFrame:
        """Use LeagueGameFinder to get Lakers schedule."""
        seasons = [season] if season else self.cfg.seasons
        frames = []
        for s in seasons:
            key = f"lakers_schedule_{s}"
            df = self._fetch(
                leaguegamefinder.LeagueGameFinder,
                key,
                team_id_nullable=self.cfg.team_id,
                season_nullable=s,
                season_type_nullable="Regular Season",
            )
            df["SEASON"] = s
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True)
        if "GAME_DATE" in combined.columns:
            combined["GAME_DATE"] = pd.to_datetime(combined["GAME_DATE"], errors="coerce")
        return combined.sort_values("GAME_DATE").reset_index(drop=True)

    def get_all_data(self) -> Dict[str, pd.DataFrame]:
        """Convenience: pull all key datasets and return as a dict."""
        logger.info("Fetching all NBA data...")
        return {
            "player_game_logs": self.get_player_game_logs(),
            "team_game_logs": self.get_team_game_logs(),
            "league_player_stats": self.get_league_player_stats(),
            "league_player_advanced": self.get_league_player_advanced(),
            "league_team_stats": self.get_league_team_stats(),
            "league_team_defense": self.get_league_team_defensive_stats(),
            "lakers_schedule": self.get_lakers_schedule(),
        }
