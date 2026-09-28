"""Feature engineering for match-outcome prediction.

Every feature must be knowable before kick-off. Categorical codes are learned from training rows
only, and rolling averages use strictly earlier matches (closed="left"), so no information from
the match being predicted, or from the test period, leaks into training.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULT_CODES = {"D": 0, "L": 1, "W": 2}
RESULT_LABELS = {code: label for label, code in RESULT_CODES.items()}

BASE_PREDICTORS = ["venue_code", "opp_code", "hour", "day_code"]
ROLLING_STATS = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt", "poss", "xg", "xga"]
ROLLING_PREDICTORS = [f"{c}_rolling" for c in ROLLING_STATS]
ROLLING_WINDOW = 3

# fbref names a club one way in the "team" column and another in "opponent". Every club in the
# data must be listed, or its matches silently drop out of the paired analysis; Nottingham
# Forest, Sheffield United and West Brom were missing.
TEAM_TO_OPPONENT_NAME = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Nottingham Forest": "Nott'ham Forest",
    "Sheffield United": "Sheffield Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Bromwich Albion": "West Brom",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
}


def load_matches(path: Path) -> pd.DataFrame:
    matches = pd.read_csv(path, index_col=0).reset_index(drop=True)
    matches["date"] = pd.to_datetime(matches["date"])
    return matches


def _codes_from_training(values: pd.Series, train_mask: pd.Series) -> pd.Series:
    """Integer codes learned from training rows; categories first seen later map to -1."""
    known = values[train_mask].dropna().unique()
    return values.map({v: i for i, v in enumerate(known)}).fillna(-1).astype(int)


def add_base_features(matches: pd.DataFrame, split_date: pd.Timestamp) -> pd.DataFrame:
    matches = matches.copy()
    train_mask = matches["date"] < split_date
    matches["venue_code"] = _codes_from_training(matches["venue"], train_mask)
    matches["opp_code"] = _codes_from_training(matches["opponent"], train_mask)
    matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype(int)
    matches["day_code"] = matches["date"].dt.dayofweek
    matches["result_code"] = matches["result"].map(RESULT_CODES)
    return matches


def rolling_averages(group: pd.DataFrame, cols: list[str], new_cols: list[str]) -> pd.DataFrame:
    """Each row gets the mean of the team's previous matches, never including itself."""
    group = group.sort_values("date")
    group[new_cols] = group[cols].rolling(ROLLING_WINDOW, closed="left").mean()
    return group.dropna(subset=new_cols)


def add_rolling_features(matches: pd.DataFrame) -> pd.DataFrame:
    with_form = pd.concat(
        rolling_averages(group, ROLLING_STATS, ROLLING_PREDICTORS)
        for _, group in matches.groupby("team")
    )
    return with_form.reset_index(drop=True)


def pair_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Joins each match's prediction for the home side with the one for its opponent.

    `predictions` needs date, team, opponent, actual and predicted columns. Each match appears
    once per team in the data, so pairing both views lets us check whether the model's two
    predictions for one fixture agree.
    """
    predictions = predictions.assign(
        opponent_name=predictions["team"].map(lambda t: TEAM_TO_OPPONENT_NAME.get(t, t))
    )
    return predictions.merge(
        predictions,
        left_on=["date", "opponent_name"],
        right_on=["date", "opponent"],
        suffixes=("_x", "_y"),
    )
