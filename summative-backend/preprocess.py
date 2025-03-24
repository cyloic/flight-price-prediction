import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the original dataset
df = pd.read_excel("Data_Train.xlsx")  # Change to CSV if needed

# Identify categorical & numerical features
categorical_cols = ["airline", "source", "destination"]  # Adjust as needed
numerical_cols = ["duration", "stops"]  # Adjust as needed

# Create and fit encoders/scalers
encoder = OneHotEncoder(handle_unknown="ignore")
scaler = StandardScaler()

encoder.fit(df[categorical_cols])
scaler.fit(df[numerical_cols])

# Save preprocessor objects
joblib.dump(encoder, "encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Preprocessing files saved successfully!")
