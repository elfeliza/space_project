import joblib
import numpy as np
import pandas as pd
import requests
from catboost import CatBoostClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

try:
    open("dataset.csv")
except FileNotFoundError:
    file_id = "1HJTDfP6njwgsToIrSSPBNNRXvh9QmX_R"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = requests.get(url)
    while r.status_code != 200:
        r = requests.get(url)
    with open("dataset.csv", "wb") as f:
        f.write(r.content)

df = pd.read_csv("dataset.csv")
target = "Dx:Cancer"
df = df.drop_duplicates()

for name in df.columns:
    mean = np.array(df[f"{name}"] != "?").mean()
    df[name] = df[name].replace({"?": mean})

X = df.drop(columns=target)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y
)


def score(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return recall_score(y_test, y_pred, average="macro"), y_pred


scaler = joblib.load("scaler")

X_test_scal = scaler.transform(X_test)

model = CatBoostClassifier()
model.load_model("model")

Score, y_pred = score(model, X_test, y_test)
print("RECALL SCORE: ", Score)

ans = pd.DataFrame({"predict": y_pred})
ans.to_csv("predict.csv")
