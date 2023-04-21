import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load the data
data = pd.read_csv("Source/J2.csv")

# Preprocess the data
teams = set(data["Home"].unique()) | set(data["Away"].unique())
team_to_idx = {team: i for i, team in enumerate(teams)}
data["Home"] = data["Home"].apply(lambda x: team_to_idx[x])
data["Away"] = data["Away"].apply(lambda x: team_to_idx[x])

# Split the data into training and testing sets
train_data = data.iloc[:-5]
test_data = data.iloc[-5:]

# Define the model
model = Sequential()
model.add(LSTM(32, input_shape=(10, 2)))
model.add(Dense(2, activation="linear"))

# Compile the model
model.compile(loss="mse", optimizer="adam")

# Train the model
X_train = []
y_train = []
for i in range(10, len(train_data)):
    X_train.append(train_data.iloc[i-10:i][["Home", "Away"]].values)
    y_train.append(train_data.iloc[i][["HG", "AG"]].values)
X_train = np.array(X_train).astype("float32")
y_train = np.array(y_train).astype("float32")
model.fit(X_train, y_train, epochs=50, batch_size=32)

X_test = test_data[["Home", "Away"]].rolling(10, min_periods=1, win_type="boxcar").mean().values
n_samples = X_test.shape[0]
if n_samples < 10:
    X_test = np.pad(X_test, ((0, 10 - n_samples), (0, 0)), mode="constant")
X_test = np.array(X_test).astype("float32").reshape(1, 10, 2)
y_pred = model.predict(X_test)
# Print the test data and the predicted outcome
print("Test data:")
print(test_data[["Home", "Away"]])
print("\nPredicted outcome:")
print(y_pred)