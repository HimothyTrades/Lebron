"""
Configuration loader.
Reads config.yaml and .env, provides a single Config object used across the project.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class Config:
    """Central config object. Access all settings through this."""

    def __init__(self, config_path: Optional[Path] = None):
        root = Path(__file__).parent.parent
        if config_path is None:
            config_path = root / "config.yaml"
        self._cfg = _load_yaml(config_path)

        # Env vars
        self.sports_api_key: str = os.getenv("SPORTS_API_KEY", "")
        self.sports_api_base_url: str = os.getenv(
            "SPORTS_API_BASE_URL", "https://sports-api.net"
        )
        self.nba_season_start: int = int(os.getenv("NBA_SEASON_START", "2018"))
        self.nba_season_end: int = int(os.getenv("NBA_SEASON_END", "2026"))

        # Player
        self.player_name: str = self._cfg["player"]["name"]
        self.player_id: int = self._cfg["player"]["player_id"]
        self.team_abbreviation: str = self._cfg["player"]["team_abbreviation"]
        self.team_id: int = self._cfg["player"]["team_id"]

        # Seasons
        self.seasons: List[str] = self._cfg["seasons"]

        # Target
        self.target: str = self._cfg["target"]

        # Rolling windows
        self.rolling_windows: List[int] = self._cfg["rolling_windows"]
        self.ewma_spans: List[int] = self._cfg["ewma_spans"]

        # Model
        self.model_type: str = self._cfg["model"]["type"]
        self.test_method: str = self._cfg["model"]["test_method"]
        self.n_splits: int = self._cfg["model"]["n_splits"]
        self.test_size: int = self._cfg["model"]["test_size"]

        # Odds / Sports API
        odds_cfg = self._cfg.get("odds", {})
        self.odds_enabled: bool = odds_cfg.get("enabled", True)
        self.odds_historical_enabled: bool = odds_cfg.get("historical_enabled", False)
        self.odds_use_for_training: bool = odds_cfg.get("use_for_training", False)
        self.odds_use_for_prediction: bool = odds_cfg.get("use_for_prediction", True)
        self.odds_fail_gracefully: bool = odds_cfg.get("fail_gracefully", True)
        self.odds_markets: List[str] = odds_cfg.get("markets", [])
        self.bookmakers: List[str] = odds_cfg.get("bookmakers", [])
        self.sports_api_cfg: Dict[str, Any] = self._cfg.get("sports_api", {})

        # Prospect theory weights
        self.prospect_weights: Dict[str, float] = self._cfg["prospect_theory"]["weights"]

        # Pressure index weights & settings
        self.pressure_weights: Dict[str, float] = self._cfg["pressure_index"]["weights"]
        self.rivalry_teams: List[str] = self._cfg["pressure_index"]["rivalry_teams"]
        self.strong_opp_threshold: float = self._cfg["pressure_index"][
            "strong_opponent_threshold"
        ]
        self.close_spread_threshold: float = self._cfg["pressure_index"][
            "close_spread_threshold"
        ]
        self.high_total_threshold: float = self._cfg["pressure_index"][
            "high_total_threshold"
        ]

        # Paths (relative to project root)
        self.root: Path = root
        p = self._cfg["paths"]
        self.raw_nba_path: Path = root / p["raw_nba"]
        self.raw_odds_path: Path = root / p["raw_odds"]
        self.processed_path: Path = root / p["processed"]
        self.predictions_path: Path = root / p["predictions"]
        self.model_artifacts_path: Path = root / p["model_artifacts"]
        self.charts_path: Path = root / p["charts"]
        self.reports_path: Path = root / p["reports"]

        # Cache
        self.cache_enabled: bool = self._cfg["cache"]["enabled"]
        self.cache_ttl_hours: int = self._cfg["cache"]["ttl_hours"]

        # Logging
        self.log_level: str = self._cfg["logging"]["level"]
        self.log_file: Path = root / self._cfg["logging"]["file"]

        self._ensure_dirs()

    def _ensure_dirs(self):
        for d in [
            self.raw_nba_path,
            self.raw_odds_path,
            self.processed_path,
            self.predictions_path,
            self.model_artifacts_path,
            self.charts_path,
            self.reports_path,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        return self._cfg.get(key, default)


# Module-level singleton
_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
