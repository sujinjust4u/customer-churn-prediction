from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import joblib
import numpy as np

# Define request body
class Features(BaseModel):
    features: list  # list of 19 numeric features

# Initialize app
app = FastAPI()

# Allow CORS (frontend and backend on same machine)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
frontend_path = os.path.join(os.path.dirname(__file__), "../frontend")
app.mount("/frontend", StaticFiles(directory=frontend_path), name="frontend")

# Load models
BASE_DIR = os.path.dirname(__file__)
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
model = joblib.load(os.path.join(BASE_DIR, "xgb_churn_model.pkl"))

@app.get("/")
def read_root():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict(features: Features):
    X = np.array(features.features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)
    return {"prediction": int(prediction[0])}
