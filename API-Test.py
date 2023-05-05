import requests

# Replace {YOUR_API_KEY} with your actual API key
api_key = "f36a541e7645483fabe53ab140e03f7d"
competition_id = 2021  # Premier League ID
url = f"https://api.football-data.org/v2/competitions/{competition_id}/matches?odds=ASIA"
headers = {"X-Auth-Token": api_key}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    matches_data = response.json()["matches"]
    for match in matches_data:
        home_team = match["homeTeam"]["name"]
        away_team = match["awayTeam"]["name"]
        match_date = match["utcDate"]
        if "asianHandicap" in match["odds"]:
            asian_handicap_data = match["odds"]["asianHandicap"]
            asian_handicap_home = asian_handicap_data["home"]
            asian_handicap_away = asian_handicap_data["away"]
            asian_handicap_line = asian_handicap_data["handicap"]
            print(f"{home_team} vs {away_team} on {match_date}")
            print(f"Asian Handicap Line: {asian_handicap_line}")
            print(f"{home_team} Handicap Odds: {asian_handicap_home}")
            print(f"{away_team} Handicap Odds: {asian_handicap_away}")
        else:
            print(f"No Asian Handicap data available for {home_team} vs {away_team} on {match_date}")
else:
    print(f"Error: {response.status_code} - {response.text}")