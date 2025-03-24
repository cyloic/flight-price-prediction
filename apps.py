from fastapi import FastAPI
import pandas as pd
import joblib
import re
from pydantic import BaseModel

app = FastAPI()

# Load the trained model and encoders
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")
model = joblib.load("flight_price_model.pkl")

# Stop mapping
stops_mapping = {"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}

# Pydantic Model for input validation
class FlightInput(BaseModel):
    Airline: str
    Source: str
    Destination: str
    Duration: str
    Total_Stops: str

# Function to convert 'Duration' into minutes
def convert_duration(duration):
    """Convert '2h 50m' into total minutes (170)."""
    hours = re.search(r'(\d+)h', duration)
    minutes = re.search(r'(\d+)m', duration)
    total_minutes = (int(hours.group(1)) * 60 if hours else 0) + (int(minutes.group(1)) if minutes else 0)
    return total_minutes

@app.post("/predict")
def predict_price(input_data: FlightInput):
    """Predict flight price based on input features."""
    df_input = pd.DataFrame([input_data.dict()])

    # Convert Duration
    df_input["Duration"] = df_input["Duration"].apply(convert_duration)
    
    # Convert Stops
    df_input["Total_Stops"] = df_input["Total_Stops"].map(stops_mapping)

    # Apply transformations
    categorical_cols = ["Airline", "Source", "Destination"]
    numerical_cols = ["Duration", "Total_Stops"]

    cat_transformed = encoder.transform(df_input[categorical_cols]).toarray()
    num_transformed = scaler.transform(df_input[numerical_cols])

    # Merge features & predict
    X_input = pd.concat([pd.DataFrame(cat_transformed), pd.DataFrame(num_transformed)], axis=1)
    predicted_price = model.predict(X_input)

    return {"Predicted_Price": round(predicted_price[0], 2)}

