import fastapi
import pickle
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sklearn.linear_model import LinearRegression
from fastapi.middleware.cors import CORSMiddleware

# Define API
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define input data model with 5 features
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float 
    feature5: float

# Ensure model file exists or train a new one
model_path = os.path.join(os.path.dirname(__file__), 'best_model.pkl')

def train_and_save_model():
    # Generate dummy training data with 5 features
    X_train = np.random.rand(100, 5)
    y_train = X_train @ np.array([3.5, -2.1, 1.8, 0.5, -1.2]) + np.random.randn(100) * 0.1
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return model

# Load model or train a new one
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
else:
    model = train_and_save_model()

@app.get("/")
def home():
    return {"message": "Welcome to the Flight Price Prediction API. Use the /predict endpoint to make predictions."}

@app.post("/predict")
def predict(data: InputData):
    try:
        # Use all 5 features for prediction
        input_features = np.array([[data.feature1, data.feature2, data.feature3, data.feature4, data.feature5]])
        prediction = model.predict(input_features)
        return {"prediction": prediction.tolist()[0]}  # Return single value instead of list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)