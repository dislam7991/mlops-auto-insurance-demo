from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def evaluate_and_report(p):
    Path("reports").mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(p["paths"]["test"])
    y = df[p["target"]]
    X = df.drop(columns=[p["target"]])

    model = joblib.load("models/model.pkl")
    y_pred = model.predict(X)

    acc = float(accuracy_score(y, y_pred))
    f1  = float(f1_score(y, y_pred, average="binary"))

    with open("reports/metrics.json", "w") as f:
        json.dump({"accuracy": acc, "f1": f1}, f, indent=2)

    cm = confusion_matrix(y, y_pred)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png")
    plt.close()

def drift_report(p):
    Path("reports").mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(p["paths"]["train"])
    test  = pd.read_parquet(p["paths"]["test"])
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=train, current_data=test)
    report.save_html("reports/evidently_data_report.html")
