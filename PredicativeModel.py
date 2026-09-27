import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import argparse
from pathlib import Path

def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

def main():
    parser = argparse.ArgumentParser(description="Evaluate historical match predictions")
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("PremMatches2(2024-2019).csv"))
    parser.add_argument("--split-date", default="2023-08-10")
    args = parser.parse_args()
    project_data = args.data
    split_date = pd.Timestamp(args.split_date)

    matches = pd.read_csv(project_data, index_col=0).reset_index(drop=True)

    matches["date"] = pd.to_datetime(matches["date"])

    venue_values = matches.loc[matches["date"] < split_date, "venue"].dropna().unique()
    matches["venue_code"] = matches["venue"].map(dict(zip(venue_values, range(len(venue_values))))).fillna(-1).astype(int)
    matches["result_code"] = matches["result"].map({"D": 0, "L": 1, "W": 2})
    opponent_values = matches.loc[matches["date"] < split_date, "opponent"].dropna().unique()
    matches["opp_code"] = matches["opponent"].map(dict(zip(opponent_values, range(len(opponent_values))))).fillna(-1).astype(int)

    matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
    matches["day_code"] = matches["date"].dt.dayofweek

    rfc = RandomForestClassifier(n_estimators=1000, min_samples_split=100, random_state=1)

    train = matches[matches["date"] < split_date]
    test = matches[matches["date"] >= split_date]
    if train.empty or test.empty:
        parser.error("The split must leave both training and test rows")
    print(f"Split: {split_date.date()}, train={len(train)}, test={len(test)}")
    predictors = ["venue_code", "opp_code", "hour", "day_code"]
    rfc.fit(train[predictors], train["result_code"])
    preds = rfc.predict(test[predictors])

    error = accuracy_score(test["result_code"], preds)

    combined = pd.DataFrame(dict(actual=test["result_code"], predicted=preds))
    print(error)
    print(pd.crosstab(index=combined["actual"], columns=combined["predicted"]))

    precision = precision_score(test["result_code"], preds, average='weighted',zero_division=0)
    print("Weighted Precision:", precision)

    cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt", "poss", "xg", "xga"]
    new_cols = [f"{c}_rolling" for c in cols]

    matches_rolling = pd.concat(rolling_averages(group, cols, new_cols) for _, group in matches.groupby("team"))
    matches_rolling.index = range(matches_rolling.shape[0])

    def make_predictions(data, predictors):
        train = data[data["date"] < split_date]
        test = data[data["date"] >= split_date ]
        rfc.fit(train[predictors], train["result_code"])
        preds = rfc.predict(test[predictors])
        combined = pd.DataFrame(dict(actual=test["result_code"], predicted=preds), index=test.index)
        precision =  precision_score(test["result_code"], preds, average='weighted', zero_division=0)
        return combined, precision

    combined, precision = make_predictions(matches_rolling, predictors + new_cols)
    print("New Weighted Precision:",precision)

    combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)

    class MissingDict(dict):
        __missing__ = lambda self, key: key

    map_values = {"Brighton and Hove Albion": "Brighton", "Manchester United": "Manchester Utd", "Newcastle United": "Newcastle Utd", "Tottenham Hotspur": "Tottenham", "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves"}
    mapping = MissingDict(**map_values)

    combined["new_team"] = combined["team"].map(mapping)
    merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])

    merged_x_wins_y_lose = merged[(merged["predicted_x"] == 2) & (merged["predicted_y"] ==1)]["actual_x"].value_counts()

    print("Predicted team x to beat team y",merged_x_wins_y_lose)

if __name__ == "__main__":
    main()
