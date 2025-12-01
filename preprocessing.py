import nfl_data_py as nfl
import pandas as pd

# Load weekly player data

years = [2020, 2021, 2022, 2023, 2024]
columns = [
    'player_id', 'player_name', 'position', 'position_group', 'recent_team',
    'season', 'week', 'season_type', 'opponent_team',
    'completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions',
    'sacks', 'sack_yards',
    'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles',
    'receptions', 'targets', 'receiving_yards', 'receiving_tds', 'receiving_fumbles',
    'fantasy_points', 'fantasy_points_ppr'
]

downcast = True
table = nfl.import_weekly_data(years, columns, downcast)


# Basic cleaning / filtering

# Keep only regular season games
table = table[table["season_type"] == "REG"].copy()

# Drop rows without fantasy points
table = table.dropna(subset=["fantasy_points_ppr"]).copy()

# Only keep main offensive positions
table = table[table['position_group'].isin(['QB', 'RB', 'WR', 'TE'])].copy()

# Sort chronologically for time-based features
table = table.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


# Build team-level schedule context (no leakage)

schedule = nfl.import_schedules(years)

# Home side of each game
home = schedule[['season', 'week', 'home_team', 'away_team', 'spread_line', 'total_line']].copy()
home['team'] = home['home_team']
home['is_home'] = 1

# Away side of each game
away = schedule[['season', 'week', 'home_team', 'away_team', 'spread_line', 'total_line']].copy()
away['team'] = away['away_team']
away['spread_line'] = -away['spread_line']      #
away['is_home'] = 0

# Combine to get one row per team per game
team_schedule = pd.concat([home, away], ignore_index=True)
team_schedule = team_schedule[["season", "week", "team", "spread_line", "total_line", "is_home"]]

# Sort and shift schedule features so they represent *previous game* context
team_schedule = team_schedule.sort_values(["team", "season", "week"]).reset_index(drop=True)

schedule_cols = ["spread_line", "total_line", "is_home"]
for col in schedule_cols:
    team_schedule[col] = team_schedule.groupby("team")[col].shift(1)

# Merge previous-game team context into player-level table
table = table.merge(
    team_schedule,
    left_on=["season", "week", "recent_team"],
    right_on=["season", "week", "team"],
    how="left"
).drop(columns=["team"])

# Optional: create is_away from is_home (previous game)
table["is_away"] = table["is_home"].map({1: 0, 0: 1})


# One-hot encoding for position group

table = pd.get_dummies(table, columns=["position_group"], drop_first=False)


# Player performance rolling features

rolling_cols = [
    "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
    "carries", "rushing_yards", "rushing_tds",
    "receptions", "targets", "receiving_yards", "receiving_tds"
]

# Shift base stats so they are from previous games
for col in rolling_cols:
    table[col] = table.groupby("player_id")[col].shift(1)

# Rolling averages over last 3 and 5 games (using already-shifted stats)
for col in rolling_cols:
    for w in [3, 5]:
        table[f"{col}_roll{w}"] = (
            table.groupby("player_id")[col]
                  .rolling(w, min_periods=1)
                  .mean()
                  .reset_index(level=0, drop=True)
        )


# Target and columns to drop

target_col = "fantasy_points_ppr"

drop_cols = [
    "player_id", "player_name", "recent_team",
    "season_type", "opponent_team", "position",
    "fantasy_points", "season", "week",
    "home_team", "away_team",  
    target_col
]
