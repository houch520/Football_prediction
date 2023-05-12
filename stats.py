import pandas as pd

# 讀取數據
data = pd.read_csv('Simulation\Predicted1\\2019-2020_res.csv')

data.drop(["Div","Date","HomeTeam","AwayTeam"], axis=1, inplace=True)

# 計算相關係數
corr_matrix = data.corr()

# 顯示相關係數矩陣
print(corr_matrix)