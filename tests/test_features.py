import importlib.util
from pathlib import Path
import pandas as pd

spec = importlib.util.spec_from_file_location('predictor', Path(__file__).parents[1] / 'PredicativeModel.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def test_rolling_features_exclude_current_match_and_future_rows():
    games = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=5), 'poss': [10, 20, 30, 90, 50]})
    first = model.rolling_averages(games.copy(), ['poss'], ['poss_rolling'])
    games.loc[3:, 'poss'] = 999
    changed = model.rolling_averages(games.copy(), ['poss'], ['poss_rolling'])
    assert first.loc[3, 'poss_rolling'] == 20
    assert changed.loc[3, 'poss_rolling'] == 20
    assert first.loc[4, 'poss_rolling'] != changed.loc[4, 'poss_rolling']
