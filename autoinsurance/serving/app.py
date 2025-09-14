# not yet implemented
# FastAPI app to serve the model via REST API
# see: https://fastapi.tiangolo.com/tutorial/first-steps/

from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI()

class Claim(BaseModel):
    features: Dict[str, Any]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: Claim):
    model = joblib.load("models/model.pkl")
    X = pd.DataFrame([payload.features])
    yhat = model.predict(X)[0]
    proba = getattr(model, "predict_proba", lambda x: [[None, None]])(X)[0]
    return {"prediction": int(yhat), "proba": proba[1] if proba and len(proba) > 1 else None}
# To run the app:
# uvicorn autoinsurance.serving.app:app --host 0.0.0.0 --port 8000