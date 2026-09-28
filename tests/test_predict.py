import pandas as pd
import pytest

from features import BASE_PREDICTORS, add_base_features, load_matches
from predict import DEFAULT_DATA, baseline_accuracy, evaluate


@pytest.fixture(scope="module")
def matches():
    return add_base_features(load_matches(DEFAULT_DATA), pd.Timestamp("2023-08-10"))


def test_the_split_keeps_every_test_match_after_every_training_match(matches):
    split = pd.Timestamp("2023-08-10")
    result = evaluate("fixture", matches, BASE_PREDICTORS, split)

    assert result.train_rows + result.test_rows == len(matches)
    assert result.predictions["date"].min() >= split


def test_the_model_beats_always_predicting_the_most_common_result(matches):
    split = pd.Timestamp("2023-08-10")

    result = evaluate("fixture", matches, BASE_PREDICTORS, split)

    assert result.accuracy > baseline_accuracy(matches, split)


def test_a_split_with_no_test_rows_is_rejected(matches):
    with pytest.raises(ValueError, match="both training and test rows"):
        evaluate("fixture", matches, BASE_PREDICTORS, pd.Timestamp("2030-01-01"))
