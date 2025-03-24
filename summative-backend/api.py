from fastapi import FastAPI
from pydantic import BaseModel, confloat
import joblib
import numpy as np
import uvicorn

# Load the trained model
model = joblib.load("best_model.pkl")

# Define the FastAPI app
app = FastAPI(title="Flight Price Prediction API")

# Define input schema using Pydantic
class FlightInput(BaseModel):
    duration: confloat(gt=0)  # Ensure duration is a positive float
    stops: int  # Number of stops
    airline_encoded: int  # Encoded airline feature (modify as needed)
    source_encoded: int  # Encoded source feature
    destination_encoded: int  # Encoded destination feature

@app.post("/predict")
def predict(data: FlightInput):
    # Convert input to NumPy array
    input_data = np.array([[data.duration, data.stops, data.airline_encoded, data.source_encoded, data.destination_encoded]])

    # Make prediction
    prediction = model.predict(input_data)

    return {"predicted_price": float(prediction[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, confloat, validator
import joblib
import numpy as np
import uvicorn

# Load the trained model
model = joblib.load("best_model.pkl")

# Define the FastAPI app
app = FastAPI(title="Flight Price Prediction API", description="API for predicting flight prices", version="1.0")

# Define input schema using Pydantic
class FlightInput(BaseModel):
    duration: confloat(gt=0)  # Ensure duration is a positive float
    stops: int  # Number of stops
    airline_encoded: int  # Encoded airline feature (modify as needed)
    source_encoded: int  # Encoded source feature
    destination_encoded: int  # Encoded destination feature

    @validator('stops')
    def stops_must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError('Number of stops must be non-negative')
        return v

@app.post("/predict")
def predict(data: FlightInput):
    try:
        # Convert input to NumPy array
        input_data = np.array([[data.duration, data.stops, data.airline_encoded, data.source_encoded, data.destination_encoded]])

        # Make prediction
        prediction = model.predict(input_data)

        return {"predicted_price": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)