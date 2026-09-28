"""Downloads Premier League match logs and shooting stats from fbref.com, one team at a time.

Writes data/matches-scraped.csv with the same columns as the bundled dataset. fbref rate-limits
aggressive clients, so the script waits between requests; a full five-season run takes a while.
"""

import time
from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (premier-league-outcome-predictor research script)"}
TIMEOUT = 30

years = list(range(2024, 2019, -1))
all_matches = []
url = "https://fbref.com/en/comps/9/2023-2024/2023-2024-Premier-League-Stats"


for year in years:
    data = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    soup = BeautifulSoup(data.text, features="lxml")
    standings_table = soup.select("table.stats_table")[0]

    links = [link.get("href") for link in standings_table.find_all("a")]
    links = [link for link in links if "/squads/" in link]
    team_urls = [f"https://fbref.com{link}" for link in links]

    previous_season = soup.find("a", class_="button2 prev").get("href")
    url = f"https://fbref.com{previous_season}"

    for team_url in team_urls:
        team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
        data = requests.get(team_url, headers=HEADERS, timeout=TIMEOUT)

        matches = pd.read_html(StringIO(data.text), match="Scores & Fixtures")[0]
        soup = BeautifulSoup(data.text, features="lxml")
        links = [link.get("href") for link in soup.find_all("a")]
        links = [link for link in links if link and "/shooting/" in link]
        data = requests.get(f"https://fbref.com{links[0]}", headers=HEADERS, timeout=TIMEOUT)
        try:
            shooting = pd.read_html(StringIO(data.text), match="Shooting")[0]
        except ValueError:
            continue

        shooting.columns = shooting.columns.droplevel()
        try:
            team_data = matches.merge(
                shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date"
            )
        except ValueError:
            continue
        team_data = team_data[team_data["Comp"] == "Premier League"]

        team_data["Season"] = year
        team_data["Team"] = team_name
        all_matches.append(team_data)
        print(team_name)
        # Adding a delay to prevent overwhelming the server with requests
        time.sleep(5)

match_df = pd.concat(all_matches)
match_df.columns = [c.lower() for c in match_df.columns]
match_df.to_csv("data/matches-scraped.csv")
