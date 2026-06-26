"""
Shared utilities: logging setup, retry logic, caching helpers, team name normalization.
"""

import hashlib
import json
import logging
import os
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

import pandas as pd

F = TypeVar("F", bound=Callable[..., Any])

# NBA team abbreviation → full name variants
TEAM_NAME_MAP: Dict[str, List[str]] = {
    "LAL": ["Los Angeles Lakers", "LA Lakers", "Lakers"],
    "GSW": ["Golden State Warriors", "Golden State", "Warriors"],
    "BOS": ["Boston Celtics", "Boston", "Celtics"],
    "MIA": ["Miami Heat", "Miami", "Heat"],
    "CLE": ["Cleveland Cavaliers", "Cleveland", "Cavaliers", "Cavs"],
    "DAL": ["Dallas Mavericks", "Dallas", "Mavericks", "Mavs"],
    "DEN": ["Denver Nuggets", "Denver", "Nuggets"],
    "PHX": ["Phoenix Suns", "Phoenix", "Suns"],
    "MIL": ["Milwaukee Bucks", "Milwaukee", "Bucks"],
    "BKN": ["Brooklyn Nets", "Brooklyn", "Nets"],
    "PHI": ["Philadelphia 76ers", "Philadelphia", "76ers", "Sixers"],
    "NYK": ["New York Knicks", "New York", "Knicks"],
    "CHI": ["Chicago Bulls", "Chicago", "Bulls"],
    "TOR": ["Toronto Raptors", "Toronto", "Raptors"],
    "MEM": ["Memphis Grizzlies", "Memphis", "Grizzlies"],
    "SAS": ["San Antonio Spurs", "San Antonio", "Spurs"],
    "OKC": ["Oklahoma City Thunder", "Oklahoma City", "Thunder"],
    "POR": ["Portland Trail Blazers", "Portland", "Trail Blazers", "Blazers"],
    "UTA": ["Utah Jazz", "Utah", "Jazz"],
    "MIN": ["Minnesota Timberwolves", "Minnesota", "Timberwolves", "Wolves"],
    "NOP": ["New Orleans Pelicans", "New Orleans", "Pelicans"],
    "SAC": ["Sacramento Kings", "Sacramento", "Kings"],
    "ATL": ["Atlanta Hawks", "Atlanta", "Hawks"],
    "CHA": ["Charlotte Hornets", "Charlotte", "Hornets"],
    "WAS": ["Washington Wizards", "Washington", "Wizards"],
    "DET": ["Detroit Pistons", "Detroit", "Pistons"],
    "IND": ["Indiana Pacers", "Indiana", "Pacers"],
    "ORL": ["Orlando Magic", "Orlando", "Magic"],
    "HOU": ["Houston Rockets", "Houston", "Rockets"],
    "LAC": ["Los Angeles Clippers", "LA Clippers", "Clippers"],
    "OKC": ["Oklahoma City Thunder", "Oklahoma City", "Thunder"],
}

_FULL_TO_ABBREV: Dict[str, str] = {}
for abbrev, names in TEAM_NAME_MAP.items():
    for name in names:
        _FULL_TO_ABBREV[name.lower()] = abbrev


def normalize_team_name(name: str) -> str:
    """Convert any team name variant to its 3-letter abbreviation."""
    if name.upper() in TEAM_NAME_MAP:
        return name.upper()
    return _FULL_TO_ABBREV.get(name.lower().strip(), name.upper())


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> logging.Logger:
    """Configure root logger and return it."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=fmt, handlers=handlers)
    return logging.getLogger("lebron")


def retry(
    max_attempts: int = 5,
    initial_delay: float = 2.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """Decorator: retry with exponential backoff."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        raise
                    logging.getLogger("lebron.retry").warning(
                        f"{func.__name__} failed (attempt {attempt}/{max_attempts}): {exc}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= backoff

        return wrapper  # type: ignore

    return decorator


def cache_path(cache_dir: Path, key: str) -> Path:
    """Return a deterministic cache file path for a string key."""
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return cache_dir / f"{h}.json"


def load_cache(path: Path, ttl_hours: int = 24) -> Optional[Any]:
    """Load cached JSON if it exists and is fresh enough."""
    if not path.exists():
        return None
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    if age_hours > ttl_hours:
        return None
    with open(path) as f:
        return json.load(f)


def save_cache(path: Path, data: Any) -> None:
    """Save data as JSON to cache path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def z_score_series(s: pd.Series) -> pd.Series:
    """Return z-scores; returns zeros if std is 0."""
    std = s.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def season_year_to_str(start_year: int) -> str:
    """Convert 2022 → '2022-23'."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def str_to_season_years(season_str: str):
    """Convert '2022-23' → (2022, 2023)."""
    parts = season_str.split("-")
    start = int(parts[0])
    end = int(parts[0][:2] + parts[1]) if len(parts[1]) == 2 else int(parts[1])
    return start, end
