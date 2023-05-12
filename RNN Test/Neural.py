import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 讀取數據集
data = pd.read_csv("Simulation\Predicted1\\2019-2020_res.csv")

# 刪除不必要的列
data.drop(["Div", "Hit?", "odds", "B365H", "B365D", "B365A", "B365CAHH", "B365CAHA"], axis=1, inplace=True)

# 處理日期格式
data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)

# 將FTR轉換為三元特徵
data["AHCh_win"] = data.apply(lambda row: 1 if row["AHCh_res"] == row["HomeTeam"] else 2 if row["AHCh_res"] == row["AwayTeam"] else 0, axis=1)
data["prediction_win"] = data.apply(lambda row: 1 if row["prediction"] == row["HomeTeam"] else 2 if row["prediction"] == row["AwayTeam"] else 0, axis=1)

# 刪除原FTR列和HomeTeam、AwayTeam列
data.drop(["FTR","AHCh_res", "HomeTeam", "AwayTeam","prediction"], axis=1, inplace=True)

# 將AHCh、PredictH、PredictD和PredictA添加到特徵矩陣中
X = data[["AHCh","prediction_win", "PredictH", "PredictD", "PredictA"]]
y = data["AHCh_win"]

# 拆分數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特徵轉換
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 建立模型
model = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=500, random_state=42)

# 訓練模型
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 將預測結果轉換為H、D、A
y_pred = ["H" if x == 1 else "A" if x == 2 else "D" for x in y_pred]
y_test = ["H" if x == 1 else "A" if x == 2 else "D" for x in y_test]

# 打印預測結果和實際結果
for i in range(len(y_pred)):
    print(f"Predicted: {y_pred[i]}, Actual: {y_test[i]}")

# 評估模型性能
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)