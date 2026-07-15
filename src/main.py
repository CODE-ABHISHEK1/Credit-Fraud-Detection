from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")


class Transaction(BaseModel):
    features: list[float]  # exactly 30 features


app = FastAPI(title="Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "API is running. Use POST /predict"}


@app.post("/predict")
def predict(transaction: Transaction):
    if len(transaction.features) != 30:
        raise HTTPException(status_code=400, detail="Must have 30 features.")
    data = np.array(transaction.features).reshape(1, -1)
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)[0]
    prob = model.predict_proba(data_scaled)[0][1]
    return {"is_fraud": bool(pred), "probability": float(prob)}
