import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import norm, poisson

# Read in the dataset
data = pd.read_csv('JPN.csv')
train_data = data# Use first 80% of data as train data
# train_data = data.iloc[:int(0.8*len(data)), :]# Use first 80% of data as train data
test_data = data.iloc[int(0.8*len(data)):, :] # Use last 20% of data as test data

# Create a set of all unique teams
teams = set(data['Home']) | set(data['Away'])

# Define the prior distribution for the attack and defense ratings
prior_mean = 1.5 # prior mean for the attack and defense ratings
prior_std = 0.5 # prior standard deviation for the attack and defense ratings
team_attack_mean = {team: prior_mean for team in teams}
team_defense_mean = {team: prior_mean for team in teams}
team_attack_std = {team: prior_std for team in teams}
team_defense_std = {team: prior_std for team in teams}
k = 0.11  # Set the value of k
alpha = 0.25  # Set the value of alpha

# Define a function to update the attack and defense ratings of a team after a match
def update_ratings(home_team, away_team, home_goals, away_goals):
    home_attack = team_attack_mean[home_team]
    home_defense = team_defense_mean[home_team]
    away_attack = team_attack_mean[away_team]
    away_defense = team_defense_mean[away_team]

    # Calculate the expected goals for each team
    home_expected_goals = max(0.01, home_attack * away_defense)
    away_expected_goals = max(0.01, away_attack * home_defense)

    # Calculate the likelihood function for the number of goals scored by each team
    home_goals_likelihood = stats.poisson.pmf(home_goals, home_expected_goals)
    away_goals_likelihood = stats.poisson.pmf(away_goals, away_expected_goals)

    # Update the attack and defense ratings based on the actual goals scored
    home_attack_new = home_attack + k * (home_goals - home_expected_goals)
    home_defense_new = home_defense + k * (away_goals - away_expected_goals)
    away_attack_new = away_attack + k * (away_goals - away_expected_goals)
    away_defense_new = away_defense + k * (home_goals - home_expected_goals)

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
    print(f"Probability of {home_team} winning: {outcome_probs[0]:.2%}")
    print(f"Probability of {away_team} winning: {outcome_probs[1]:.2%}")
    print(f"Probability of a draw: {outcome_probs[2]:.2%}")
    print(f"\nProbability distribution of scores:")
    print(f"{home_team}: {[(i, home_score_probs[i]*100) for i in range(6)]}")
    print(f"{away_team}: {[(i, away_score_probs[i]*100) for i in range(6)]}")
    if outcome_probs[0] > outcome_probs[1]:
        return home_team, np.argmax(home_score_probs), np.argmax(away_score_probs)
    elif outcome_probs[0] < outcome_probs[1]:
        return away_team, np.argmax(home_score_probs), np.argmax(away_score_probs)
    else:
        return 'Draw', np.argmax(home_score_probs), np.argmax(away_score_probs)
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
    
    # Update the ratings for the home and away teams
    update_ratings(home_team, away_team, home_goals, away_goals)

# for i in range(len(test_data)):
#     home_team = test_data.iloc[i]['Home']
#     away_team = test_data.iloc[i]['Away']
#     home_goals = test_data.iloc[i]['HG']
#     away_goals = test_data.iloc[i]['AG']
    
#     # Update the ratings for the home and away teams
#     update_ratings(home_team, away_team, home_goals, away_goals)
    
#     # Extract the actual result from the 'Res' column and convert to 'Home', 'Away', or 'Draw'
#     result = test_data.iloc[i]['Res']
#     if result == 'H':
#         actual_result = home_team
#     elif result == 'A':
#         actual_result = away_team
#     else:
#         actual_result = 'Draw'
    
#     # Predict the outcome of the match
#     prediction,pre_home_score,pre_away_score = predict_outcome(home_team, away_team)
    
#     # Append the predicted and actual results to the lists
#     predictions.append(prediction)
#     actual_results.append(actual_result)
    
#     # Print the team ratings and predicted/actual results for the match
#     print('Match {}: {} vs. {}:'.format(i+1, home_team, away_team))
#     print('- Team ratings:')
#     print('  - {}: attack={:.2f}, defense={:.2f}'.format(home_team, team_attack_mean[home_team], team_defense_mean[home_team]))
#     print('  - {}: attack={:.2f}, defense={:.2f}'.format(away_team, team_attack_mean[away_team], team_defense_mean[away_team]))
#     print('- Predicted outcome:{}({}-{})'.format(prediction,pre_home_score,pre_away_score))
#     print('- Actual outcome:{}({}-{})'.format(actual_result,home_goals,away_goals))
#     print()

# # Calculate the accuracy of the predictions
# correct_predictions = [p == a or a=='Draw' for p, a in zip(predictions, actual_results)]
# accuracy = sum(correct_predictions) / len(correct_predictions)


# Read in the test dataset
test_data = pd.read_csv('test.csv')

# Open a new file to write the predictions
with open('predictions.csv', 'w') as file:
    # Write the header row
    file.write('Home,Away,PredictResult\n')

    # Make predictions for each match in the test data
    for i in range(len(test_data)):
        home_team = test_data.iloc[i]['Home']
        away_team = test_data.iloc[i]['Away']

        # Predict the outcome of the match
        prediction, _, _ = predict_outcome(home_team, away_team)

        # Write the prediction to the output file
        file.write('{},{},{}\n'.format(home_team, away_team, prediction))
# # Print the results
# print('Accuracy:', accuracy)