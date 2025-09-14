from pathlib import Path
import pandas as pd
import mlflow
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def make_splits(p):
    df = pd.read_parquet(p["paths"]["processed"])
    y = df[p["target"]]
    X = df.drop(columns=[p["target"]])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=p["split"]["test_size"],
        random_state=p["split"]["random_state"],
        stratify=y
    )
    Path("data/splits").mkdir(parents=True, exist_ok=True)
    pd.concat([X_train, y_train], axis=1).to_parquet(p["paths"]["train"], index=False)
    pd.concat([X_test, y_test], axis=1).to_parquet(p["paths"]["test"], index=False)

def train_and_log(p):
    mlflow.set_experiment("Insurance-ETL-Train")

    df = pd.read_parquet(p["paths"]["train"])
    y = df[p["target"]]
    X = df.drop(columns=[p["target"]])

    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()

    pre = ColumnTransformer(transformers=[
        ("num", StandardScaler(with_mean=False), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat)
    ])

    model = Pipeline(steps=[
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    with mlflow.start_run(run_name="baseline-logreg"):
        model.fit(X, y)

        Path("models").mkdir(exist_ok=True)
        joblib.dump(model, "models/model.pkl")
        mlflow.sklearn.log_model(model, "model")
