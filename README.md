Flight Price Prediction App

Project Overview

This project is a flight price prediction system that includes a Flutter mobile application and a FastAPI backend powered by machine learning models. The system allows users to input flight details and receive predicted ticket prices based on trained models.

Features

Mobile App (Flutter): User-friendly interface for entering flight details and getting predictions.

API (FastAPI): Handles prediction requests and processes input data.

Machine Learning Models: Utilizes Linear Regression, Decision Trees, and Random Forest to predict flight prices.

Swagger UI: Interactive API documentation for testing the model endpoints.

Model Selection & Performance

We evaluated three machine learning models using the Mean Squared Error (MSE) loss metric:

Linear Regression: Struggled with capturing non-linearity in the dataset.

Decision Trees: Improved accuracy but prone to overfitting.

Random Forest: Achieved the best performance, reducing overfitting while maintaining predictive accuracy.

Final Model: Based on the evaluation, we selected Random Forest for deployment due to its superior balance of accuracy and generalization.

Installation & Setup

1. Clone the Repository

git clone https://github.com/your-repo/flight-price-prediction.git
cd flight-price-prediction

2. Backend Setup (FastAPI)

Install Dependencies

pip install -r requirements.txt

Run the API

uvicorn main:app --reload

Access the API documentation at: http://127.0.0.1:8000/docs

3. Mobile App Setup (Flutter)

Install Dependencies

flutter pub get

Run the App

flutter run

API Endpoints

Endpoint

Method

Description

/predict

POST

Takes in flight details and returns predicted price

Video Demo

Watch the demo showcasing the app, API, and model performance: Demo Link

Contributors

Cyusa Loic - Developer

License

This project is licensed under the MIT License.

