"""
Sample data generator for LeBron James game logs.

Used as fallback when nba_api is blocked by network policy (e.g. in sandboxed
environments). Generates statistically realistic data based on LeBron's known
career averages. Clearly labeled as SAMPLE data throughout.

This data is NOT real — it is synthetic and used only to exercise the pipeline.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger("lebron.sample_data")

# Season-level averages based on LeBron's actual career stats (publicly known)
SEASON_PROFILES = {
    "2018-19": dict(pts=27.4, min=35.2, fga=19.3, fg_pct=0.510, fg3a=5.9, fg3_pct=0.339,
                    fta=6.5, ft_pct=0.665, ast=8.3, reb=8.5, tov=3.6, pm=5.5, games=55),
    "2019-20": dict(pts=25.3, min=34.6, fga=18.8, fg_pct=0.493, fg3a=5.3, fg3_pct=0.348,
                    fta=7.1, ft_pct=0.693, ast=10.2, reb=7.8, tov=3.9, pm=8.0, games=67),
    "2020-21": dict(pts=25.0, min=33.4, fga=18.4, fg_pct=0.513, fg3a=5.5, fg3_pct=0.365,
                    fta=6.4, ft_pct=0.698, ast=7.8, reb=7.7, tov=3.7, pm=5.5, games=45),
    "2021-22": dict(pts=30.3, min=37.2, fga=21.7, fg_pct=0.524, fg3a=5.9, fg3_pct=0.359,
                    fta=7.4, ft_pct=0.756, ast=6.2, reb=8.2, tov=3.5, pm=3.5, games=56),
    "2022-23": dict(pts=28.9, min=35.5, fga=20.0, fg_pct=0.500, fg3a=5.1, fg3_pct=0.324,
                    fta=8.1, ft_pct=0.761, ast=6.8, reb=8.3, tov=3.2, pm=4.0, games=55),
    "2023-24": dict(pts=25.7, min=35.3, fga=18.3, fg_pct=0.540, fg3a=4.8, fg3_pct=0.410,
                    fta=6.7, ft_pct=0.737, ast=8.3, reb=7.3, tov=3.5, pm=5.5, games=71),
    "2024-25": dict(pts=23.7, min=34.1, fga=17.2, fg_pct=0.487, fg3a=4.5, fg3_pct=0.377,
                    fta=5.9, ft_pct=0.726, ast=8.5, reb=7.2, tov=3.3, pm=4.0, games=60),
    "2025-26": dict(pts=22.0, min=32.0, fga=16.5, fg_pct=0.480, fg3a=4.2, fg3_pct=0.360,
                    fta=5.5, ft_pct=0.720, ast=8.0, reb=7.0, tov=3.1, pm=3.5, games=40),
}

OPPONENTS = [
    "GSW", "BOS", "MIA", "DAL", "DEN", "PHX", "MIL", "BKN", "PHI",
    "NYK", "CHI", "TOR", "MEM", "SAS", "OKC", "POR", "UTA", "MIN",
    "NOP", "SAC", "ATL", "CHA", "WAS", "DET", "IND", "ORL", "HOU",
    "LAC", "CLE",
]


def _season_start(season_str: str) -> pd.Timestamp:
    year = int(season_str.split("-")[0])
    return pd.Timestamp(f"{year}-10-19")


def generate_player_game_logs(seasons: List[str] = None, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic LeBron game-log data.
    Returns a DataFrame with the same schema as nba_api PlayerGameLog.
    """
    if seasons is None:
        seasons = list(SEASON_PROFILES.keys())

    rng = np.random.default_rng(seed)
    all_rows = []
    game_id_counter = 200000001

    for season in seasons:
        if season not in SEASON_PROFILES:
            continue
        p = SEASON_PROFILES[season]
        n = p["games"]
        start = _season_start(season)

        for i in range(n):
            # Space games ~2.5 days apart with some variance
            date = start + pd.Timedelta(days=int(i * 2.5 + rng.integers(0, 2)))
            is_home = rng.random() > 0.5
            opp = rng.choice(OPPONENTS)
            matchup = f"LAL {'vs.' if is_home else '@'} {opp}"

            # Generate stats with natural correlation and variance
            pts = max(0, round(rng.normal(p["pts"], p["pts"] * 0.22)))
            minutes = max(10, min(48, round(rng.normal(p["min"], 3.5), 1)))
            fga = max(1, round(rng.normal(p["fga"], p["fga"] * 0.20)))
            fg3a = max(0, round(rng.normal(p["fg3a"], p["fg3a"] * 0.25)))
            fta = max(0, round(rng.normal(p["fta"], p["fta"] * 0.25)))

            fg_pct = max(0.20, min(1.0, rng.normal(p["fg_pct"], 0.08)))
            fg3_pct = max(0.0, min(1.0, rng.normal(p["fg3_pct"], 0.10)))
            ft_pct = max(0.40, min(1.0, rng.normal(p["ft_pct"], 0.08)))

            fgm = round(fga * fg_pct)
            fg3m = round(fg3a * fg3_pct)
            ftm = round(fta * ft_pct)
            # Recalculate pts from made shots for consistency
            pts_calc = fgm * 2 + fg3m + ftm
            # Blend: use either direct pts or calculated (50/50 for realism)
            pts_final = round((pts + pts_calc) / 2)

            ast = max(0, round(rng.normal(p["ast"], 2.5)))
            reb = max(0, round(rng.normal(p["reb"], 2.2)))
            oreb = max(0, round(reb * 0.15))
            dreb = reb - oreb
            tov = max(0, round(rng.normal(p["tov"], 1.2)))
            stl = max(0, round(rng.normal(1.2, 0.7)))
            blk = max(0, round(rng.normal(0.6, 0.5)))
            pm = round(rng.normal(p["pm"], 12.0))
            win = rng.random() > 0.45  # ~55% win rate

            row = {
                "SEASON_ID": f"2{season.split('-')[0]}",
                "Player_ID": 2544,
                "Game_ID": f"{game_id_counter:010d}",
                "GAME_DATE": date.strftime("%b %d, %Y"),
                "MATCHUP": matchup,
                "WL": "W" if win else "L",
                "MIN": f"{int(minutes)}:{int((minutes % 1) * 60):02d}",
                "FGM": fgm,
                "FGA": fga,
                "FG_PCT": round(fga and fgm / fga or 0, 3),
                "FG3M": fg3m,
                "FG3A": fg3a,
                "FG3_PCT": round(fg3a and fg3m / fg3a or 0, 3),
                "FTM": ftm,
                "FTA": fta,
                "FT_PCT": round(fta and ftm / fta or 0, 3),
                "OREB": oreb,
                "DREB": dreb,
                "REB": reb,
                "AST": ast,
                "STL": stl,
                "BLK": blk,
                "TOV": tov,
                "PF": max(0, round(rng.normal(1.5, 0.8))),
                "PTS": pts_final,
                "PLUS_MINUS": pm,
                "VIDEO_AVAILABLE": 1,
                "SEASON": season,
                "PLAYOFF": False,
            }
            all_rows.append(row)
            game_id_counter += 1

    df = pd.DataFrame(all_rows)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y", errors="coerce")
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    logger.warning(
        f"SAMPLE DATA: Generated {len(df)} synthetic LeBron game logs. "
        "This is NOT real NBA data. Run with real nba_api access for actual predictions."
    )
    return df


def generate_team_stats(seasons: List[str] = None, seed: int = 99) -> pd.DataFrame:
    """Generate synthetic Lakers and opponent team stats."""
    if seasons is None:
        seasons = list(SEASON_PROFILES.keys())
    rng = np.random.default_rng(seed)
    rows = []
    all_teams = ["LAL"] + OPPONENTS
    for season in seasons:
        for team in all_teams:
            is_lal = team == "LAL"
            rows.append({
                "TEAM_ID": hash(team) % 1000000,
                "TEAM_ABBREVIATION": team,
                "TEAM_NAME": team,
                "SEASON": season,
                "GP": 82,
                "W": rng.integers(25, 65),
                "L": 82 - rng.integers(25, 65),
                "OFF_RATING": round(rng.normal(112.5 if is_lal else 111.0, 4.0), 1),
                "DEF_RATING": round(rng.normal(111.0 if is_lal else 112.5, 4.0), 1),
                "NET_RATING": round(rng.normal(1.5 if is_lal else 0.0, 6.0), 1),
                "PACE": round(rng.normal(99.5, 3.0), 1),
                "AST_PCT": round(rng.normal(0.60, 0.05), 3),
                "AST_TO": round(rng.normal(1.8, 0.3), 2),
                "OPP_FG_PCT": round(rng.normal(0.455, 0.015), 3),
                "OPP_FG3_PCT": round(rng.normal(0.352, 0.025), 3),
                "OPP_PTS": round(rng.normal(112.0, 5.0), 1),
                "E_OFF_RATING": round(rng.normal(112.5 if is_lal else 111.0, 4.0), 1),
                "E_DEF_RATING": round(rng.normal(111.0 if is_lal else 112.5, 4.0), 1),
                "E_NET_RATING": round(rng.normal(1.5 if is_lal else 0.0, 6.0), 1),
                "E_PACE": round(rng.normal(99.5, 3.0), 1),
            })
    return pd.DataFrame(rows)


def save_sample_cache(raw_nba_path, seasons: List[str] = None):
    """Save synthetic data to cache so the pipeline reads it on startup."""
    import json
    from src.utils import cache_path

    if seasons is None:
        seasons = list(SEASON_PROFILES.keys())

    gl = generate_player_game_logs(seasons)
    ts = generate_team_stats(seasons)

    for season in seasons:
        # Player game log cache
        season_gl = gl[gl["SEASON"] == season].copy()
        # Convert dates back to string for JSON
        season_gl["GAME_DATE"] = season_gl["GAME_DATE"].dt.strftime("%b %d, %Y")
        key = f"player_gamelog_2544_{season}"
        p = cache_path(raw_nba_path, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(season_gl.to_dict(orient="records"), f)

        # Playoff placeholder (empty)
        key_po = f"player_gamelog_2544_{season}_playoffs"
        p_po = cache_path(raw_nba_path, key_po)
        with open(p_po, "w") as f:
            json.dump([], f)

        # Team stats caches
        lal_ts = ts[(ts["TEAM_ABBREVIATION"] == "LAL") & (ts["SEASON"] == season)]
        for measure, data in [
            (f"league_team_stats_{season}_Base", ts[ts["SEASON"] == season]),
            (f"league_team_stats_{season}_Advanced", ts[ts["SEASON"] == season]),
            (f"league_team_opp_{season}", ts[ts["SEASON"] == season]),
            (f"league_player_stats_{season}_PerGame", pd.DataFrame()),
            (f"league_player_adv_{season}", pd.DataFrame()),
        ]:
            cp = cache_path(raw_nba_path, measure)
            with open(cp, "w") as f:
                json.dump(data.to_dict(orient="records"), f)

        # Lakers schedule
        sched_rows = season_gl.rename(columns={"Game_ID": "GAME_ID"})[
            ["GAME_ID", "GAME_DATE", "MATCHUP", "WL"]
        ].copy()
        sched_rows["GAME_DATE"] = sched_rows["GAME_DATE"]
        key_s = f"lakers_schedule_{season}"
        cp_s = cache_path(raw_nba_path, key_s)
        with open(cp_s, "w") as f:
            json.dump(sched_rows.to_dict(orient="records"), f)

    logger.info(f"Sample data cached to {raw_nba_path}")
