from pathlib import Path

import pandas as pd

from features import (
    TEAM_TO_OPPONENT_NAME,
    add_base_features,
    load_matches,
    pair_predictions,
    rolling_averages,
)

DATA = Path(__file__).parents[1] / "data/matches-2019-2024.csv"


def test_rolling_features_exclude_current_match_and_future_rows():
    games = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=5), "poss": [10, 20, 30, 90, 50]}
    )
    first = rolling_averages(games.copy(), ["poss"], ["poss_rolling"])
    games.loc[3:, "poss"] = 999
    changed = rolling_averages(games.copy(), ["poss"], ["poss_rolling"])

    assert first.loc[3, "poss_rolling"] == 20
    assert changed.loc[3, "poss_rolling"] == 20
    assert first.loc[4, "poss_rolling"] != changed.loc[4, "poss_rolling"]


def test_opponent_codes_are_learned_from_training_rows_only():
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01", "2023-01-08", "2023-09-01"]),
            "venue": ["Home", "Away", "Home"],
            "opponent": ["Arsenal", "Chelsea", "Luton Town"],
            "time": ["15:00", "17:30", "20:00"],
            "result": ["W", "D", "L"],
        }
    )

    coded = add_base_features(matches, pd.Timestamp("2023-08-01"))

    assert list(coded["opp_code"]) == [0, 1, -1]  # a club first seen in the test period
    assert list(coded["hour"]) == [15, 17, 20]
    assert list(coded["result_code"]) == [2, 0, 1]


def test_every_club_in_the_dataset_can_be_paired_with_its_opponent_row():
    matches = load_matches(DATA)
    opponent_names = set(matches["opponent"])

    unmatched = {
        team
        for team in matches["team"].unique()
        if TEAM_TO_OPPONENT_NAME.get(team, team) not in opponent_names
    }

    assert unmatched == set()


def test_both_views_of_a_fixture_are_joined():
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "team": ["Nottingham Forest", "Arsenal"],
            "opponent": ["Arsenal", "Nott'ham Forest"],
            "actual": [1, 2],
            "predicted": [1, 2],
        }
    )

    pairs = pair_predictions(predictions)

    assert len(pairs) == 2
    row = pairs[pairs["team_x"] == "Nottingham Forest"].iloc[0]
    assert row["team_y"] == "Arsenal"
