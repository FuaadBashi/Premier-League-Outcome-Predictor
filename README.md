# Premier-League-Outcome-Predictor
This project is a comprehensive data-driven predictive model for Premier League matches. It includes data scraping, preprocessing, feature engineering, and machine learning modeling to predict match outcomes and analyse team performance trends. Achieves a final precision of 68% on outcome of match
# Premier League Predictive Model
---

## Features
1. **Data Scraping:**
   - Utilizes Python's `requests` and `BeautifulSoup` libraries to scrape match statistics and shooting data from FBRef.
   - Consolidates data into a comprehensive dataset across multiple seasons.

2. **Data Cleaning & Feature Engineering:**
   - Processes scraped data to remove inconsistencies and fill missing values.
   - Converts categorical features into numeric representations for machine learning.
   - Adds rolling averages for key performance metrics such as goals scored, shots on target, and more.

3. **Predictive Model:**
   - Implements a **Random Forest Classifier** using `scikit-learn` to predict match results.
   - Features engineered include venue, opponent code, expected goals (`xg`), and more.

4. **Performance Evaluation:**
   - Evaluates model performance using **accuracy** and **precision** metrics.
   - Achieves improved precision by adding rolling averages and dual team analysis.

---

## Setup
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Required Python packages:
  ```bash
  pip install pandas numpy requests beautifulsoup4 scikit-learn matplotlib seaborn lxml
  ```

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/Premier-League-Predictive-Model.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Premier-League-Predictive-Model
   ```
3. Place the dataset `PremMatches.csv` in the project folder or scrape data using the `DataScraper.py` script.

---

## Files
1. **`DataScraper.py`**
   - Scrapes Premier League match data from FBRef and saves it to a CSV file.
   - Example usage:
     ```bash
     python DataScraper.py
     ```

2. **`PremMatches.csv`**
   - Pre-scraped dataset containing match results and statistics for multiple seasons.

3. **`PredicativeModel.py`**
   - Implements the predictive model and trains a Random Forest Classifier.
   - Includes functions for feature engineering, training, and evaluation.
   - Example usage:
     ```bash
     python PredicativeModel.py
     ```

---

## Usage
### Step 1: Data Scraping
Run the `DataScraper.py` script to scrape Premier League data:
```bash
python DataScraper.py
```
This script will save a CSV file containing match data to the project directory.

### Step 2: Predictive Modeling
Run the `PredicativeModel.py` script to train and evaluate the predictive model:
```bash
python PredicativeModel.py
```
### Key Results:
- Initial precision: **52%**
- Improved precision with rolling averages: **53.1%**
- Final precision using dual-team analysis: **68%**

---

## Technical Highlights
1. **Rolling Averages:**
   - Rolling averages are computed for performance metrics (e.g., goals, shots, distances) to enhance model features.
   - Excludes the current match data to avoid data leakage.

2. **Dual Team Analysis:**
   - Considers both teams' predicted performances to refine match outcome predictions.

3. **Custom Data Mapping:**
   - Maps team names to ensure consistency in predictions.

---

## Future Work
- Add more features such as player statistics, injuries, or weather conditions.
- Experiment with different machine learning algorithms like Gradient Boosting or Neural Networks.
- Expand the analysis to include other leagues or competitions.

---

## Contributors
- **Fuaad Shurie** - Data Scientist

Feel free to open issues or pull requests to contribute to this project!

---

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

