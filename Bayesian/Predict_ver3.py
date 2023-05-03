import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import norm, poisson
import argparse

# # Create an ArgumentParser object
# parser = argparse.ArgumentParser(description='Description of your program')

# # Add an argument for the first string
# parser.add_argument('string1', type=str, help='Description of the first string')

# # Add an argument for the second string
# parser.add_argument('string2', type=str, help='Description of the second string')

# # Parse the command-line arguments
# args = parser.parse_args()

tour="RUS_2020"
Reverse = ""
# Read in the dataset
data = pd.read_csv('Simulation\Combined\\'+tour+Reverse+'.csv', encoding="ISO-8859-1")
output =  'Simulation\Predicted\\'+tour+Reverse+'_res.csv'
# train_data = data# Use first 80% of data as train data
train_data = data.iloc[:int(0.4*len(data)), :]# Use first 80% of data as train data
test_data = data.iloc[int(0.4*len(data)):, :] # Use last 20% of data as test data

# Create a set of all unique teams
teams = set(data['Home']) | set(data['Away'])

# Define the prior distribution for the attack and defense ratings
prior_mean = 1.5 # prior mean for the attack and defense ratings
prior_std = 0.5 # prior standard deviation for the attack and defense ratings
team_attack_mean = {team: prior_mean for team in teams}
team_defense_mean = {team: prior_mean for team in teams}
team_attack_std = {team: prior_std for team in teams}
team_defense_std = {team: prior_std for team in teams}
k = 0.2 # Set the value of k
alpha = 0.8 # Set the value of alpha 
#test set 0.38,0.82
#0.11,0.25
#0.4
# Define a function to update the attack and defense ratings of a team after a match
def update_ratings(home_team, away_team, home_goals, away_goals):
    home_attack = team_attack_mean[home_team]
    home_defense = team_defense_mean[home_team]
    away_attack = team_attack_mean[away_team]
    away_defense = team_defense_mean[away_team]

    home_attack_rating_dist = norm(team_attack_mean[home_team], team_attack_std[home_team])
    away_defense_rating_dist = norm(team_defense_mean[away_team], team_defense_std[away_team])
    home_score_probs = [poisson.pmf(i, home_attack_rating_dist.mean() * away_defense_rating_dist.mean()) for i in range(6)]
    away_attack_rating_dist = norm(team_attack_mean[away_team], team_attack_std[away_team])
    home_defense_rating_dist = norm(team_defense_mean[home_team], team_defense_std[home_team])
    away_score_probs = [poisson.pmf(i, away_attack_rating_dist.mean() * home_defense_rating_dist.mean()) for i in range(6)]
    home_expected_goals = sum(i * home_score_probs[i] for i in range(6))
    away_expected_goals = sum(i * away_score_probs[i] for i in range(6))
    # Calculate the expected goals for each team
    # home_expected_goals = max(0.01, home_attack * away_defense)
    # away_expected_goals = max(0.01, away_attack * home_defense)

    # Calculate the likelihood function for the number of goals scored by each team
    home_goals_likelihood = stats.poisson.pmf(home_goals, home_expected_goals)
    away_goals_likelihood = stats.poisson.pmf(away_goals, away_expected_goals)

    home_diff=min(max(home_goals - home_expected_goals, -2),3)
    away_diff=min(max(away_goals - away_expected_goals, -2),3)
    # Update the attack and defense ratings based on the actual goals scored
    home_attack_new = home_attack + k * ((home_diff)- max(0.2 * ((1+home_diff))*(home_diff)/2,0))
    home_defense_new = home_defense + k * ((away_diff)- max(0.2 * ((1+away_diff))*(away_diff)/2,0))
    away_attack_new = away_attack + k * ((away_diff)- max(0.2 * ((1+away_diff))*(away_diff)/2,0))
    away_defense_new = away_defense + k * ((home_diff)- max(0.2 * ((1+home_diff))*(home_diff)/2,0))

    # Update the team ratings with a weighted average of the old and new ratings
    team_attack_mean[home_team] = alpha * home_attack_new + (1 - alpha) * home_attack
    team_defense_mean[home_team] = alpha * home_defense_new + (1 - alpha) * home_defense
    team_attack_mean[away_team] = alpha * away_attack_new + (1 - alpha) * away_attack
    team_defense_mean[away_team] = alpha * away_defense_new + (1 - alpha) * away_defense

# Define a function to predict the outcome of a match based on the attack and defense ratings of each team
def predict_outcome(home_team, away_team):
    home_attack_rating_dist = norm(team_attack_mean[home_team], team_attack_std[home_team])
    away_defense_rating_dist = norm(team_defense_mean[away_team], team_defense_std[away_team])
    home_score_probs = [poisson.pmf(i, home_attack_rating_dist.mean() * away_defense_rating_dist.mean()) for i in range(6)]
    away_attack_rating_dist = norm(team_attack_mean[away_team], team_attack_std[away_team])
    home_defense_rating_dist = norm(team_defense_mean[home_team], team_defense_std[home_team])
    away_score_probs = [poisson.pmf(i, away_attack_rating_dist.mean() * home_defense_rating_dist.mean()) for i in range(6)]
    outcome_probs = np.zeros((3,))
    for i in range(6):
        for j in range(6):
            if i > j:
                outcome_probs[0] += home_score_probs[i] * away_score_probs[j]
            elif i < j:
                outcome_probs[1] += home_score_probs[i] * away_score_probs[j]
            else:
                outcome_probs[2] += home_score_probs[i] * away_score_probs[j]
    outcome_probs /= np.sum(outcome_probs)
    # print(f"Probability of {home_team} winning: {outcome_probs[0]:.2%}")
    # print(f"Probability of {away_team} winning: {outcome_probs[1]:.2%}")
    # print(f"Probability of a draw: {outcome_probs[2]:.2%}")
    # print(f"\nProbability distribution of scores:")
    home_expected_goals = sum(i * home_score_probs[i] for i in range(6))
    away_expected_goals = sum(i * away_score_probs[i] for i in range(6))
    # print(f"expected goals {home_expected_goals}-{away_expected_goals}")
    # print(f"{home_team}: {[(i, home_score_probs[i]*100) for i in range(6)]}")
    # print(f"{away_team}: {[(i, away_score_probs[i]*100) for i in range(6)]}")
    if outcome_probs[0] > outcome_probs[1] and outcome_probs[2]<outcome_probs[0]:
        return home_team, outcome_probs[0], outcome_probs[1]
    elif outcome_probs[0] < outcome_probs[1] and outcome_probs[2]<outcome_probs[1]:
        return away_team, outcome_probs[0], outcome_probs[1]
    elif outcome_probs[2]>outcome_probs[0] and outcome_probs[2]>outcome_probs[1]:
        return 'Draw', outcome_probs[0], outcome_probs[1]
# Initialize the attack and defense ratings for all teams to 1.0
team_attack_mean = dict.fromkeys(teams, 1.0)
team_defense_mean = dict.fromkeys(teams, 1.0)

# Make predictions for each match in the test data
predictions = []
actual_results = []
for i in range(len(train_data)):
    home_team = train_data.iloc[i]['Home']
    away_team = train_data.iloc[i]['Away']
    home_goals = train_data.iloc[i]['HG']
    away_goals = train_data.iloc[i]['AG']
    
    # Update the ratings for the Home and Away teams
    update_ratings(home_team, away_team, home_goals, away_goals)
# Create empty lists to store the predicted and actual results
predictions = []
actual_results = []

# Open a file for writing
with open(output, 'w', encoding='ISO-8859-1') as file:
    # Write the header row to the file
    file.write('Date,Home,Away,HG,AG,Res,prediction,Hit?,odds,AvgH,AvgD,AvgA,Better,Asian_res,Asian_hit\n')

    for i in range(len(test_data)):
        date = test_data.iloc[i]['Date']
        home_team = test_data.iloc[i]['Home']
        away_team = test_data.iloc[i]['Away']
        home_goals = test_data.iloc[i]['HG']
        away_goals = test_data.iloc[i]['AG']
        result = test_data.iloc[i]['Res']
        AvgH = test_data.iloc[i]['AvgH']
        AvgD = test_data.iloc[i]['AvgD']
        AvgA = test_data.iloc[i]['AvgA']
        
        # Check if AvgH/AvgD/AvgA is empty and replace with 1 if it is
        if AvgH == '' or pd.isna(AvgH):
            AvgH = 1
        if AvgD == '' or pd.isna(AvgD):
            AvgD = 1
        if AvgA == '' or pd.isna(AvgA):
            AvgA = 1

        if result == 'H':
            actual_result = home_team
        elif result == 'A':
            actual_result = away_team
        else:
            actual_result = 'Draw'
        
        if(AvgH<AvgA):
            better=home_team
        else:
            better=away_team
        Asian=1 if better == home_team else -1

        if(home_goals+Asian>away_goals):
            Asian_res=home_team
        elif(home_goals+Asian<away_goals):
            Asian_res=away_team
        else:   
            Asian_res='Draw'
        
        
        prediction, pre_home_score, pre_away_score= predict_outcome(home_team, away_team)
        
        predictions.append(prediction)
        actual_results.append(actual_result)

        Asian_hit = 1 if Asian_res==prediction else 0
        if prediction=='Draw':
            Asian_hit =-1
        #Analyse
        Check_normal = 1 if prediction == actual_result else 0
        
        if Check_normal==1:
            if actual_result==home_team:
                odds = AvgH-1 
            elif actual_result==away_team:
                odds = AvgA-1
            else:
                odds = AvgD-1
        else:
            odds=-1

        # Write the prediction results to the file
        file.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
            date, home_team, away_team, home_goals, away_goals, actual_result,prediction,Check_normal,odds, AvgH, AvgD, AvgA,better,Asian_res,Asian_hit
            ))
        update_ratings(home_team, away_team, home_goals, away_goals)

# Calculate the accuracy of the predictions
correct_predictions = [p == a  for p, a in zip(predictions, actual_results)]
accuracy = sum(correct_predictions) / len(correct_predictions)


# # Read in the test dataset
# test_data = pd.read_csv('TestData\\'+tour+'Test.csv')

# # Open a new file to write the predictions
# with open(output, 'w', encoding='utf-8')  as file:
#     # Write the header row
#     file.write('Date,Home,Away,PredictResult,H,D,A\n')

#     # Make predictions for each match in the test data
#     for i in range(len(test_data)):
#         date = test_data.iloc[i]['Date']
#         home_team = test_data.iloc[i]['Home']
#         away_team = test_data.iloc[i]['Away']

#         # Predict the outcome of the match
#         prediction, HP, AP = predict_outcome(home_team, away_team)

#         # Write the prediction to the output file
#         file.write('{},{},{},{},{:.2%},{:.2%},{:.2%}\n'.format(date, home_team, away_team, prediction,HP,1-HP-AP,AP))
# with open('Result\\Stat\\Stat'+tour+'.csv', 'w', encoding='utf-8')  as file:
#     # Write the header row
#     file.write('Team,Attack,Defense\n')

#     # Make predictions for each match in the test data
#     for i in teams:
#         team = i
#         attack = team_attack_mean[i]
#         defense = team_defense_mean[i]
#         file.write('{},{},{}\n'.format(team,attack,defense))

        
# Print the results
print('Accuracy:', accuracy)
correct_predictions = [p == a or a=='Draw' for p, a in zip(predictions, actual_results)]
accuracy = sum(correct_predictions) / len(correct_predictions)
print('Accuracy +Draw:', accuracy)
