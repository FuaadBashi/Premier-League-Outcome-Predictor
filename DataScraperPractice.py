import pandas as pd
import requests
from bs4 import BeautifulSoup


url = "https://fbref.com/en/comps/9/2023-2024/2023-2024-Premier-League-Stats"
data = requests.get(url)

soup = BeautifulSoup(data.text,features="lxml")
standings_table = soup.select('table.stats_table')[0]
links = standings_table.find_all('a')
links = [l.get("href") for l in links]
links = [l for l in links if '/squads/' in l]
team_urls = [f"https://fbref.com{l}" for l in links]


data = requests.get(team_urls[0])

soup2 =  BeautifulSoup(data.text,features="lxml")
# select class filter
prem_tag = soup2.find_all("div", {"class": "filter"})[1]  
prem_tag = prem_tag.find_all('a')
prem_tag = [p.get("href") for p in prem_tag]
prem_tag.pop(0)
prem_data_url = [p for p in prem_tag if 'Premier-League' in p]

data = requests.get("https://fbref.com"+prem_data_url[0])

matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
print(matches.head())

soup3 = BeautifulSoup(data.text,features="lxml")
links = soup3.find_all('a')
links = [l.get("href") for l in links]
links = [l for l in links if l and '/shooting/' in l]
data = requests.get(f"https://fbref.com{links[0]}")


shooting = pd.read_html(data.text, match="Shooting")[1]
shooting.columns = shooting.columns.droplevel()
print(shooting.head())

team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
team_data.head()
