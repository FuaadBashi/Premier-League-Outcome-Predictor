"""Train and evaluate the Premier League result predictor on a historical time split."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

from features import (
    BASE_PREDICTORS,
    RESULT_CODES,
    RESULT_LABELS,
    ROLLING_PREDICTORS,
    add_base_features,
    add_rolling_features,
    load_matches,
    pair_predictions,
)

DEFAULT_DATA = Path(__file__).parent / "data/matches-2019-2024.csv"


@dataclass
class Evaluation:
    name: str
    train_rows: int
    test_rows: int
    accuracy: float
    precision: float
    predictions: pd.DataFrame


def new_model() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=1000, min_samples_split=100, random_state=1)


def evaluate(
    name: str, data: pd.DataFrame, predictors: list[str], split_date: pd.Timestamp
) -> Evaluation:
    train = data[data["date"] < split_date]
    test = data[data["date"] >= split_date]
    if train.empty or test.empty:
        raise ValueError("The split date must leave both training and test rows")
    model = new_model().fit(train[predictors], train["result_code"])
    predicted = model.predict(test[predictors])
    predictions = test[["date", "team", "opponent"]].assign(
        actual=test["result_code"], predicted=predicted
    )
    return Evaluation(
        name,
        len(train),
        len(test),
        accuracy_score(test["result_code"], predicted),
        precision_score(test["result_code"], predicted, average="weighted", zero_division=0),
        predictions,
    )


def baseline_accuracy(data: pd.DataFrame, split_date: pd.Timestamp) -> float:
    """Accuracy of always predicting the training set's most common result."""
    most_common = data.loc[data["date"] < split_date, "result_code"].mode()[0]
    return float((data.loc[data["date"] >= split_date, "result_code"] == most_common).mean())


def confident_pair_accuracy(predictions: pd.DataFrame) -> tuple[int, int]:
    """Where the model says team X wins *and* its opponent loses, how often X really won."""
    pairs = pair_predictions(predictions)
    confident = pairs[
        (pairs["predicted_x"] == RESULT_CODES["W"]) & (pairs["predicted_y"] == RESULT_CODES["L"])
    ]
    return int((confident["actual_x"] == RESULT_CODES["W"]).sum()), len(confident)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--split-date",
        default="2023-08-10",
        help="matches on or after this date are the test set (default: the 2023-24 season)",
    )
    args = parser.parse_args(argv)
    split_date = pd.Timestamp(args.split_date)

    matches = add_base_features(load_matches(args.data), split_date)
    try:
        runs = [
            evaluate("Fixture only", matches, BASE_PREDICTORS, split_date),
            evaluate(
                "Fixture + recent form",
                add_rolling_features(matches),
                BASE_PREDICTORS + ROLLING_PREDICTORS,
                split_date,
            ),
        ]
    except ValueError as e:
        parser.error(str(e))

    print(f"Train: matches before {split_date.date()}   Test: matches from {split_date.date()}")
    baseline = baseline_accuracy(matches, split_date)
    print(f"Baseline (always predict the most common result): {baseline:.1%}")
    print()
    print(f"{'Model':<24}{'Train':>7}{'Test':>7}{'Accuracy':>11}{'Precision':>11}")
    for r in runs:
        print(
            f"{r.name:<24}{r.train_rows:>7}{r.test_rows:>7}{r.accuracy:>11.1%}{r.precision:>11.1%}"
        )

    best = runs[-1]
    print(f"\nConfusion matrix ({best.name}):")
    table = pd.crosstab(
        best.predictions["actual"].map(RESULT_LABELS),
        best.predictions["predicted"].map(RESULT_LABELS),
        rownames=["actual"],
        colnames=["predicted"],
    )
    print(table.to_string())

    right, total = confident_pair_accuracy(best.predictions)
    if total:
        print(
            f"\nWhen the model predicts one side wins and the other loses: "
            f"{right}/{total} correct ({right / total:.1%})"
        )


if __name__ == "__main__":
    main()
