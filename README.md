# Premier League Match Prediction

A Python machine-learning experiment covering match-data collection, categorical features, rolling team statistics, and random-forest predictions.

## Code to explore

- [DataScraper.py](DataScraper.py): match-data collection.
- [PredicativeModel.py](PredicativeModel.py): preprocessing, date-based train/test selection, rolling features, and evaluation.
- [PremMatches2(2024-2019).csv](PremMatches2%282024-2019%29.csv): checked-in historical dataset.

The rolling features use prior matches via `rolling(3, closed='left')`. The final analysis joins predictions for both sides of a match.

## Run locally

```bash
git clone https://github.com/FuaadBashi/Premier-League-Outcome-Predictor.git
cd Premier-League-Outcome-Predictor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pandas numpy requests beautifulsoup4 scikit-learn matplotlib seaborn lxml
```

Before running, change `project_data` in `PredicativeModel.py` from the original absolute path to a local CSV, for example `PremMatches2(2024-2019).csv`, and review the date split and required columns.

```bash
python PredicativeModel.py
```

## Interpreting the results

Weighted precision over all predictions and the win rate within a filtered subset are different quantities. Earlier README figures and source comments disagree, so this overview does not repeat a headline percentage. A reproducible result should record the dataset version, date split, class mapping, selection rule, sample count, and metric together.

This is a historical modeling experiment, not evidence of profitable betting or future match accuracy.
