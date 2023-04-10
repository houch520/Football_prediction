import pandas as pd
import numpy as np

# read in the dataset
data = pd.read_csv('JPN.csv', usecols=['Home', 'Away', 'HG', 'AG'])

# create a set of all unique teams
teams = set(data['Home']) | set(data['Away'])
team_attack = {team: 5 for team in teams}
team_defense = {team: 5 for team in teams}
ratio = 0.2

# define a function to predict the number of goals for each team
def predict_goals(home_team, away_team):
    home_ratio = team_attack[home_team] - team_defense[away_team] + 1
    away_ratio = team_attack[away_team] - team_defense[home_team] + 1
    home_goals = max(0, int(home_ratio))
    away_goals = max(0, int(away_ratio))
    return home_goals, away_goals
  
# iterate through each match in the dataset
for index, row in data.iterrows():
    home_team = row['Home']
    away_team = row['Away']
    actual_home_goals = row['HG']
    actual_away_goals = row['AG']
    
    # predict the number of goals for each team
    predicted_home_goals, predicted_away_goals = predict_goals(home_team, away_team)
    
    # compare actual goals to predicted goals
    if actual_home_goals > predicted_home_goals:
        # increase the attack ratio for the home team
        team_attack[home_team] += ratio
        # decrease the defense ratio for the away team
        team_defense[away_team] -= ratio
    elif actual_home_goals < predicted_home_goals:
        # decrease the attack ratio for the home team
        team_attack[home_team] -= ratio
        # increase the defense ratio for the away team
        team_defense[away_team] += ratio
    
    if actual_away_goals > predicted_away_goals:
        # increase the attack ratio for the away team
        team_attack[away_team] += ratio
        # decrease the defense ratio for the home team
        team_defense[home_team] -= ratio
    elif actual_away_goals < predicted_away_goals:
        # decrease the attack ratio for the away team
        team_attack[away_team] -= ratio
        # increase the defense ratio for the home team
        team_defense[home_team] += ratio

# read in the test dataset
test_data = pd.read_csv('test.csv', usecols=['Home', 'Away', 'PredictHG', 'PredictAG'])

# iterate through each match in the test dataset
for index, row in test_data.iterrows():
    home_team = row['Home']
    away_team = row['Away']
    
    # predict the number of goals for each team
    predicted_home_goals, predicted_away_goals = predict_goals(home_team, away_team)
    
    # output the predicted goals
    test_data.at[index, 'PredictHG'] = predicted_home_goals
    test_data.at[index, 'PredictAG'] = predicted_away_goals

# print the updated attack and defense ratios for each team
for team in teams:
    print(team, 'Attack:', team_attack[team], 'Defense:', team_defense[team])

# print the predicted goals for the matches in the test dataset
print(test_data)