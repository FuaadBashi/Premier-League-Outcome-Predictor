from io import StringIO
import requests
from bs4 import BeautifulSoup
import time
import pandas as pd


years = list(range(2024, 2019, -1))
all_matches = []
url = "https://fbref.com/en/comps/9/2023-2024/2023-2024-Premier-League-Stats"


for year in years:
    data = requests.get(url)
    soup = BeautifulSoup(data.text,features="lxml")
    standings_table = soup.select('table.stats_table')[0]

    links = [l.get("href") for l in standings_table.find_all('a')]
    links = [l for l in links if '/squads/' in l]
    team_urls = [f"https://fbref.com{l}" for l in links]
    
    previous_season = soup.find('a', class_='button2 prev').get("href")
    url = f"https://fbref.com{previous_season}"


    for team_url in team_urls:
        team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
        data = requests.get(team_url)

        matches = pd.read_html(StringIO(data.text), match="Scores & Fixtures")[0]
        soup = BeautifulSoup(data.text, features="lxml")
        links = [l.get("href") for l in soup.find_all('a')]
        links = [l for l in links if l and '/shooting/' in l]
        data = requests.get(f"https://fbref.com{links[0]}")
        try:
            shooting = pd.read_html(StringIO(data.text), match="Shooting")[0]
        except ValueError:
            continue
        
        shooting.columns = shooting.columns.droplevel()
        try:
            team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
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
match_df
match_df.to_csv("PremMatches2.csv")

 