# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
import pickle
from pydantic import BaseModel

app = FastAPI()

# Load trained model and encoders with error handling
try:
    model = pickle.load(open("best_model.pkl", "rb"))
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", str(e))
    model = None  # Set to None to prevent crashes

try:
    encoder = pickle.load(open("encoder.pkl", "rb"))
    print("✅ Encoder loaded successfully")
except Exception as e:
    print("❌ Error loading encoder:", str(e))
    encoder = None

try:
    scaler = pickle.load(open("scaler.pkl", "rb"))
    print("✅ Scaler loaded successfully")
except Exception as e:
    print("❌ Error loading scaler:", str(e))
    scaler = None

# Define request schema
class FlightInput(BaseModel):
    Airline: str
    Source: str
    Destination: str
    Duration: float
    Total_Stops: int

@app.post("/predict")
def predict_flight_price(data: FlightInput):
    try:
        if model is None or encoder is None or scaler is None:
            raise HTTPException(status_code=500, detail="❌ Model, encoder, or scaler not loaded properly.")

        print("\n✅ Received data:", data.dict())  # Debugging input
        df_input = pd.DataFrame([data.dict()])

        # Encode categorical features
        cat_features = ["Airline", "Source", "Destination"]
        cat_transformed = encoder.transform(df_input[cat_features]).toarray()
        num_transformed = scaler.transform(df_input[["Duration", "Total_Stops"]])

        # Combine features
        features = np.hstack((cat_transformed, num_transformed))
        print("📌 Transformed features shape:", features.shape)  # Debugging transformation

        # Ensure correct shape
        if features.shape[1] != model.n_features_in_:
            raise ValueError(f"❌ Model expects {model.n_features_in_} features but got {features.shape[1]}")

        # Make prediction
        prediction = model.predict(features)[0]
        print("✅ Predicted price:", prediction)

        return {"predicted_price": prediction}

    except Exception as e:
        print("❌ Error:", str(e))  # Debugging error
        raise HTTPException(status_code=500, detail=str(e))
