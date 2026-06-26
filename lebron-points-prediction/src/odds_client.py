"""
Sports API odds client.

Endpoint paths and parameter names are configured in config.yaml under sports_api:.
If sports-api.net uses different URL structures, edit config.yaml — no code changes needed.

HOW TO ADAPT:
  1. Open config.yaml
  2. Under sports_api:, update events_endpoint, odds_endpoint, player_props_endpoint
  3. Update the params: block to match the actual query parameter names your API uses
  4. Run: python main.py fetch-data
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from src.config import get_config
from src.utils import (
    cache_path,
    load_cache,
    normalize_team_name,
    retry,
    save_cache,
    setup_logging,
)

logger = logging.getLogger("lebron.odds_client")


class OddsClient:
    """
    Configurable Sports API wrapper.
    All endpoint paths and query param names come from config.yaml.
    """

    def __init__(self):
        self.cfg = get_config()
        self.api_key = self.cfg.sports_api_key
        self.base_url = self.cfg.sports_api_base_url.rstrip("/")
        self.api_cfg = self.cfg.sports_api_cfg
        self.cache_dir = self.cfg.raw_odds_path
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.api_key or self.api_key == "your_key_here":
            logger.warning(
                "SPORTS_API_KEY is not set. Odds features will use NaN placeholders. "
                "Set your key in .env to enable real odds data."
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _has_key(self) -> bool:
        return bool(self.api_key) and self.api_key != "your_key_here"

    def _cached(self, key: str) -> Optional[Any]:
        p = cache_path(self.cache_dir, key)
        return load_cache(p, self.cfg.cache_ttl_hours) if self.cfg.cache_enabled else None

    def _store(self, key: str, data: Any) -> None:
        if self.cfg.cache_enabled:
            save_cache(cache_path(self.cache_dir, key), data)

    @retry(max_attempts=4, initial_delay=2.0, backoff=2.0, exceptions=(requests.RequestException,))
    def _get(self, endpoint: str, params: Dict[str, Any]) -> Any:
        """
        Generic GET request.
        endpoint: path from config, e.g. '/v1/events'
        params: query params dict — keys come from config.yaml params block
        """
        if not self._has_key():
            return None

        p = self.api_cfg.get("params", {})
        # Inject API key using the configured parameter name
        api_key_param = p.get("api_key_param", "api_key")
        params[api_key_param] = self.api_key

        url = f"{self.base_url}{endpoint}"
        logger.info(f"GET {url} params={list(params.keys())}")
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------

    def get_events(self, date: Optional[str] = None) -> Optional[List[Dict]]:
        """
        Fetch NBA game events/schedule from Sports API.

        ADAPT: If your API uses a different endpoint path, update
               sports_api.events_endpoint in config.yaml.
        Returns list of raw event dicts, or None if no API key.
        """
        cache_key = f"events_{date or 'all'}"
        cached = self._cached(cache_key)
        if cached is not None:
            return cached

        p = self.api_cfg.get("params", {})
        endpoint = self.api_cfg.get("events_endpoint", "/v1/events")
        params: Dict[str, Any] = {
            # ADAPT: edit the param name mapping in config.yaml if needed
            p.get("sport_param", "sport"): self.api_cfg.get("sport_key", "basketball_nba"),
            p.get("league_param", "league"): self.api_cfg.get("league_id", "NBA"),
        }
        if date:
            params[p.get("date_param", "date")] = date

        data = self._get(endpoint, params)
        if data is not None:
            # ADAPT: your API may nest the list under a key like 'data', 'events', 'results'
            # Unwrap here if needed:
            events = data if isinstance(data, list) else data.get("data", data.get("events", []))
            self._store(cache_key, events)
            return events
        return None

    def get_game_odds(
        self, event_id: str = None, date: str = None
    ) -> Optional[Dict]:
        """
        Fetch moneyline, spread, and total for a specific game.

        ADAPT: Update sports_api.odds_endpoint in config.yaml.
        """
        cache_key = f"odds_{event_id or 'all'}_{date or ''}"
        cached = self._cached(cache_key)
        if cached is not None:
            return cached

        p = self.api_cfg.get("params", {})
        endpoint = self.api_cfg.get("odds_endpoint", "/v1/odds")
        params: Dict[str, Any] = {
            p.get("sport_param", "sport"): self.api_cfg.get("sport_key", "basketball_nba"),
            p.get("league_param", "league"): self.api_cfg.get("league_id", "NBA"),
        }
        if event_id:
            params[p.get("event_id_param", "event_id")] = event_id
        if date:
            params[p.get("date_param", "date")] = date

        # Pull all configured markets in one call if API supports comma-separated
        markets = ",".join(self.cfg.odds_markets[:3])  # moneyline, spread, total
        params[p.get("market_param", "market")] = markets

        data = self._get(endpoint, params)
        if data is not None:
            result = data if isinstance(data, dict) else {"raw": data}
            self._store(cache_key, result)
            return result
        return None

    def get_player_props(
        self, event_id: str = None, player_name: str = None, date: str = None
    ) -> Optional[List[Dict]]:
        """
        Fetch player points props.

        ADAPT: Update sports_api.player_props_endpoint in config.yaml.
        Some APIs include props in the main odds endpoint with a different market key.
        """
        cache_key = f"props_{event_id or 'all'}_{player_name or 'all'}_{date or ''}"
        cached = self._cached(cache_key)
        if cached is not None:
            return cached

        p = self.api_cfg.get("params", {})
        endpoint = self.api_cfg.get("player_props_endpoint", "/v1/player-props")
        params: Dict[str, Any] = {
            p.get("sport_param", "sport"): self.api_cfg.get("sport_key", "basketball_nba"),
            p.get("league_param", "league"): self.api_cfg.get("league_id", "NBA"),
            p.get("market_param", "market"): "player_points_prop",
        }
        if event_id:
            params[p.get("event_id_param", "event_id")] = event_id
        if date:
            params[p.get("date_param", "date")] = date

        data = self._get(endpoint, params)
        if data is not None:
            props = data if isinstance(data, list) else data.get("data", data.get("props", []))
            # Filter to target player if name provided
            if player_name and isinstance(props, list):
                props = [
                    p for p in props
                    if player_name.lower() in str(p.get("player_name", p.get("player", ""))).lower()
                ]
            self._store(cache_key, props)
            return props
        return None

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize_odds_response(self, raw: Dict) -> Dict[str, Any]:
        """
        Flatten a raw odds API response into a standard dict with keys:
          home_team, away_team, moneyline_home, moneyline_away,
          spread, spread_home, over_under, bookmaker

        ADAPT: The field name mapping below mirrors a common API structure.
        If your API uses different field names, edit the mapping here.
        """
        normalized: Dict[str, Any] = {
            "home_team": None,
            "away_team": None,
            "moneyline_home": None,
            "moneyline_away": None,
            "spread": None,
            "spread_home": None,
            "over_under": None,
            "bookmaker": None,
        }

        if not raw:
            return normalized

        # ADAPT: update these field paths for your API's response shape
        # Common patterns: raw['home_team'], raw['teams']['home'], raw['home']
        normalized["home_team"] = normalize_team_name(
            str(raw.get("home_team", raw.get("home", raw.get("homeTeam", ""))))
        )
        normalized["away_team"] = normalize_team_name(
            str(raw.get("away_team", raw.get("away", raw.get("awayTeam", ""))))
        )

        # Odds are often nested under bookmakers list
        bookmakers = raw.get("bookmakers", raw.get("books", [raw]))
        if bookmakers:
            book = bookmakers[0]
            normalized["bookmaker"] = book.get("key", book.get("name", "unknown"))
            markets = book.get("markets", [book])
            for mkt in markets:
                mkt_key = mkt.get("key", mkt.get("market", ""))
                outcomes = mkt.get("outcomes", mkt.get("selections", []))
                if "h2h" in mkt_key or "moneyline" in mkt_key.lower():
                    for o in outcomes:
                        team = normalize_team_name(str(o.get("name", "")))
                        price = o.get("price", o.get("odds", o.get("american", None)))
                        if team == normalized["home_team"]:
                            normalized["moneyline_home"] = price
                        else:
                            normalized["moneyline_away"] = price
                elif "spread" in mkt_key.lower() or "handicap" in mkt_key.lower():
                    for o in outcomes:
                        team = normalize_team_name(str(o.get("name", "")))
                        point = o.get("point", o.get("handicap", None))
                        if team == normalized["home_team"]:
                            normalized["spread_home"] = point
                            normalized["spread"] = point
                elif "total" in mkt_key.lower() or "over_under" in mkt_key.lower():
                    for o in outcomes:
                        if o.get("name", "").lower() == "over":
                            normalized["over_under"] = o.get("point", o.get("total", None))

        return normalized

    def match_odds_to_nba_games(
        self, game_dates: List[str]
    ) -> pd.DataFrame:
        """
        For a list of game dates (YYYY-MM-DD strings), fetch odds and return
        a DataFrame indexed by date with Lakers odds columns.
        """
        records = []
        for date in game_dates:
            events = self.get_events(date=date)
            if not events:
                records.append({"GAME_DATE": date})
                continue

            # Find Lakers game
            lakers_event = None
            for ev in events:
                home = normalize_team_name(
                    str(ev.get("home_team", ev.get("home", ev.get("homeTeam", ""))))
                )
                away = normalize_team_name(
                    str(ev.get("away_team", ev.get("away", ev.get("awayTeam", ""))))
                )
                if home == self.cfg.team_abbreviation or away == self.cfg.team_abbreviation:
                    lakers_event = ev
                    break

            if not lakers_event:
                records.append({"GAME_DATE": date})
                continue

            event_id = str(
                lakers_event.get("id", lakers_event.get("event_id", lakers_event.get("gameId", "")))
            )
            raw_odds = self.get_game_odds(event_id=event_id, date=date)
            norm = self.normalize_odds_response(raw_odds or {})
            norm["GAME_DATE"] = date

            # Determine if Lakers are home or away
            is_home = norm.get("home_team") == self.cfg.team_abbreviation
            norm["HOME_FLAG"] = int(is_home)
            norm["LAKERS_MONEYLINE"] = norm["moneyline_home"] if is_home else norm["moneyline_away"]
            norm["OPP_MONEYLINE"] = norm["moneyline_away"] if is_home else norm["moneyline_home"]
            # Spread from Lakers perspective (negative = Lakers favored)
            norm["LAKERS_SPREAD"] = norm["spread_home"] if is_home else (
                -norm["spread_home"] if norm["spread_home"] is not None else None
            )

            # Player props
            props = self.get_player_props(
                event_id=event_id,
                player_name=self.cfg.player_name,
                date=date,
            )
            if props:
                prop = props[0]
                norm["PLAYER_POINTS_LINE"] = prop.get(
                    "point", prop.get("line", prop.get("points_line", None))
                )
                norm["PLAYER_POINTS_OVER_PRICE"] = prop.get(
                    "over_price", prop.get("over", None)
                )
                norm["PLAYER_POINTS_UNDER_PRICE"] = prop.get(
                    "under_price", prop.get("under", None)
                )
            else:
                norm["PLAYER_POINTS_LINE"] = None
                norm["PLAYER_POINTS_OVER_PRICE"] = None
                norm["PLAYER_POINTS_UNDER_PRICE"] = None

            records.append(norm)

        df = pd.DataFrame(records)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        return df

    def get_empty_odds_df(self) -> pd.DataFrame:
        """Return an empty odds DataFrame with all expected columns (for use when API key is missing)."""
        cols = [
            "GAME_DATE", "home_team", "away_team", "moneyline_home", "moneyline_away",
            "spread", "spread_home", "over_under", "bookmaker", "HOME_FLAG",
            "LAKERS_MONEYLINE", "OPP_MONEYLINE", "LAKERS_SPREAD",
            "PLAYER_POINTS_LINE", "PLAYER_POINTS_OVER_PRICE", "PLAYER_POINTS_UNDER_PRICE",
        ]
        return pd.DataFrame(columns=cols)
