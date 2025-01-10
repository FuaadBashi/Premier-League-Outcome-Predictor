import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import seaborn as sns



project_data = "/Users/fuaad/PythonProjects/Premier League Predictive Model/PremMatches2.csv"

matches = pd.read_csv(project_data, index_col=0)

# converting so that col its no londer a object type
matches["date"] = pd.to_datetime(matches["date"])

# convert columns to numeric values
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["result_code"] = matches["result"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["formation_code"] = matches["formation"].str.replace("-", "", regex=True)
matches["formation_code"] = matches["formation_code"].str.replace("◆", "", regex=True).astype("int")


# convert time to hours and time of day
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek

# print(matches.head())
# print(matches.dtypes)

# Random forests is an ensemble learning method for classification, regression and other tasks
"""Random Forest classifier is a machine learning algorithm that uses a collection of decision trees to classify data into different classes. 
Good here because for example opcode doesnt define the diffculty of oppenent its just numerates the different oppenents meaning there is no linear
relationship between opcode -> oppenents, and a random Forest classifier can pick this up while a linear model can not"""

rfc = RandomForestClassifier(n_estimators=1000, min_samples_split=100, random_state=1)
"""n_estimators=1000: The random forest will consist of 1000 trees, ensuring robust predictions due to ensemble averaging.
min_samples_split=100: Each internal node must have at least 100 samples to consider splitting, encouraging simpler trees and potentially reducing overfitting."""

train = matches[matches["date"] < '2022-01-01']
test = matches[matches["date"] > '2022-01-01']
predictors = ["venue_code", "opp_code", "hour", "day_code", "poss", "formation_code"]
rfc.fit(train[predictors], train["result_code"])
preds = rfc.predict(test[predictors])

# calculate accuracy of the model
error = accuracy_score(test["result_code"], preds)

"""error = 0.47851153039832284(47.8% accuracy)"""


combined = pd.DataFrame(dict(actual=test["result_code"], predicted=preds))
print(error)
print(pd.crosstab(index=combined["actual"], columns=combined["predicted"]))
# predicted  0   1   2
# actual              
# 0          1  26  29
# 1          0  86  30
# 2          3  27  74


""" 
average='micro':
Treats all instances as part of a single pool.
Precision is computed globally (considering all true positives and false positives across classes).
Use case: When class imbalance is not a major concern.

average='macro':
Computes precision for each class independently and then takes the average.
Does not consider class imbalance (each class contributes equally).
Use case: When you want to treat all classes equally regardless of size.

average='weighted':
Similar to macro, but weighs each class's precision by its number of instances.
Use case: When you want to account for class imbalance.

average=None:
Returns precision for each class as an array instead of a single value.
Use case: When you want a breakdown of precision for each individual class.

I used "average='weighted'" because it gives an overall precision score accounting for class imbalance, which is often a practical and interpretable metric.
later if further insights are need or wanted into class-wise performance, i can supplement it with average=None to examine which classes need improvement.

"""
precision = precision_score(test["result_code"], preds, average='weighted',zero_division=0)
print("Weighted Precision:", precision)
"""weighted Precision: 0.377082861681235(37.7% precision)"""

# grouped_matches = matches.groupby("team")
# group = grouped_matches.get_group("Arsenal").sort_values("date")


# the function computes rolling averages for each match for each given column
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date") # sorts by date
    rolling_stats = group[cols].rolling(3, closed='left').mean() # closed left takes the rolling average of the matches previous weeks excluding the current week. 
    group[new_cols] = rolling_stats # assign the rolling average as a new column
    group = group.dropna(subset=new_cols) #  drop missing values
    return group

cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt", "poss", "xg", "xga"]
new_cols = [f"{c}_rolling" for c in cols]



"""took our original matches data frame we grouped it by team which creates 
one data frame for each team in our data and then we applied a function to each of those team data 
frames to compute the rolling averages so i have rolling averages for every match"""
matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols)) 
matches_rolling = matches_rolling.droplevel('team') # Drops the extra index level
matches_rolling.index = range(matches_rolling.shape[0]) # makes sure there are only unique values in index

# created a funtion that will just make it easy to continue to iterate on the algorithm
def make_predictions(data, predictors):
    train = data[data["date"] < '2023-08-9']
    test = data[data["date"] > '2023-08-10' ]
    rfc.fit(train[predictors], train["result_code"])
    preds = rfc.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["result_code"], predicted=preds), index=test.index)
    precision =  precision_score(test["result_code"], preds, average='weighted', zero_division=0)
    return combined, precision

# now we can call this function with different data and predictors
combined, precision = make_predictions(matches_rolling, predictors + new_cols)
print("New Weighted Precision:",precision)
"""new precision = 0.41051496669698356(41% precision, improved by 3.3%)"""


combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)


class MissingDict(dict):
    __missing__ = lambda self, key: key

# Make sure the oppenent names column are their full team-name(getting rid of inconsistents)
map_values = {"Brighton and Hove Albion": "Brighton", "Manchester United": "Manchester Utd", "Newcastle United": "Newcastle Utd", "Tottenham Hotspur": "Tottenham", "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves"} 
mapping = MissingDict(**map_values)

combined["new_team"] = combined["team"].map(mapping)
merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])

merged_x_wins_y_lose = merged[(merged["predicted_x"] == 2) & (merged["predicted_y"] ==1)]["actual_x"].value_counts()

print("Predicted team x to beat team y",merged_x_wins_y_lose)
# 138/183 percision = 75%, so big improvement with winning team prediction

"""started out with some data on premier league matches then moved on to cleaning the data and getting it ready for
machine learning then created an initial machine learning model with just a few predictors and a target. trained a random forest model 
to actually operate on that set of predictors and computed a precision score then improved accuracy by generating more predictors 
and training the model again using these rolling averages. improved precision and then improved precision again by looking
at both sides of a match both how team a was predicted to do and how team b was predicted to do and ended up with a 68 percent precision"""