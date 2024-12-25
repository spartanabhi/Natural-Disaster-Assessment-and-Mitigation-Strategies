from flask import Flask, request, jsonify, render_template
import pandas as pd
import requests
from datetime import datetime
import csv
import pickle
import os
from joblib import load


app = Flask(__name__)

WEATHER_API_KEY = "f18acc43e2da39df93a8293f004bea83"
WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"

model_path = 'models/flood_model.pkl'
weather_path = 'models/weather_models.pkl'

# Verify file existence
if not os.path.exists(model_path) or not os.path.exists(weather_path):
    raise FileNotFoundError("Model files are missing. Please re-run 'ml.py' to generate them.")

# Load models
# Ensure `model_path` is the path to your .pkl file
# model_path = "path_to_your_model.pkl"  # Replace with the actual file path

# Open the file in binary read mode
flood_model, flood_scaler = load(model_path)




temp_model, humidity_model, windspeed_model, rainfall_model = load(weather_path)

# Load dataset
data = pd.read_csv("Data.csv")
features = ['rainfall', 'temp', 'windspeed', 'humidity']

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    global data
    try:
        location = request.json.get("location")
        if not location:
            return jsonify({"error": "Location is required"}), 400

        # Fetch current weather data from OpenWeather API
        weather_response = requests.get(WEATHER_API_URL, params={
            "q": location,
            "appid": WEATHER_API_KEY,
            "units": "metric"
        })
        if weather_response.status_code != 200:
            return jsonify({"error": "Failed to fetch weather data"}), 500

        weather_data = weather_response.json()
        current_weather = {
            "temp": weather_data["main"]["temp"],
            "humidity": weather_data["main"]["humidity"],
            "windspeed": weather_data["wind"]["speed"],
            "rainfall": weather_data.get("rain", {}).get("1h", 0)
        }

        # Predict flood risk for current weather
        last_5_days = data[features].tail(5)
        current_weather_df = pd.DataFrame([current_weather])
        prediction_input = pd.concat([last_5_days, current_weather_df], ignore_index=True)
        scaled_features = flood_scaler.transform(prediction_input[features])

        risk_probabilities = flood_model.predict_proba(scaled_features)
        risk_prediction = flood_model.predict(scaled_features)[-1]
        risk_percentage = risk_probabilities[-1][1] * 100

        # Predict weather for the next day
        current_weather_values = [current_weather[feature] for feature in features]
        next_day_temp = temp_model.predict([current_weather_values])[0]
        next_day_humidity = humidity_model.predict([current_weather_values])[0]
        next_day_windspeed = windspeed_model.predict([current_weather_values])[0]
        next_day_rainfall = rainfall_model.predict([current_weather_values])[0]

        next_day_weather = {
            "temp": next_day_temp,
            "humidity": next_day_humidity,
            "windspeed": next_day_windspeed,
            "rainfall": next_day_rainfall
        }

        # Predict next day's flood risk
        next_day_weather_df = pd.DataFrame([next_day_weather])
        next_day_scaled = flood_scaler.transform(next_day_weather_df[features])
        next_day_risk_probabilities = flood_model.predict_proba(next_day_scaled)
        next_day_risk_prediction = flood_model.predict(next_day_scaled)[0]
        next_day_risk_percentage = next_day_risk_probabilities[0][1] * 100

        # Save today’s data if not already present
        today_date = datetime.now().strftime("%Y-%m-%d")
        flood_value = 1 if risk_percentage > 75 else 0

        if today_date not in data['datetime'].values:
            new_row = {
                "datetime": today_date,
                "temp": current_weather["temp"],
                "humidity": current_weather["humidity"],
                "windspeed": current_weather["windspeed"],
                "rainfall": current_weather["rainfall"],
                "FLOOD": flood_value
            }
            with open('Data.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(new_row.values())

            data = pd.read_csv("Data.csv")

        return jsonify({
            "location": location,
            "current_temperature": current_weather["temp"],
            "current_humidity": current_weather["humidity"],
            "current_windspeed": current_weather["windspeed"],
            "current_rainfall": current_weather["rainfall"],
            "current_flood_risk": "High" if risk_prediction == 1 else "Low",
            "current_risk_percentage": f"{risk_percentage:.2f}%",
            "next_day_temperature": next_day_temp,
            "next_day_humidity": next_day_humidity,
            "next_day_windspeed": next_day_windspeed,
            "next_day_rainfall": next_day_rainfall,
            "next_day_flood_risk": "High" if next_day_risk_prediction == 1 else "Low",
            "next_day_risk_percentage": f"{next_day_risk_percentage:.2f}%"
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
