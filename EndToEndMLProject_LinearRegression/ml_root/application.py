from flask import Flask, request, jsonify, render_template, app
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler



## import ridge regressor and standard scalar

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # folder: ml_root
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")      # go one step above /ml_root

ridge_model = joblib.load(os.path.join(MODEL_DIR, "ridge_model.joblib"))
scaler_model = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
TEMPLATE_DIR = os.path.join(BASE_DIR, "..", "templates")  # go up and then to template

application = Flask(__name__, template_folder=TEMPLATE_DIR)

@application.route('/')
def home():
    return render_template('index.html')

@application.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        # Only accept POST here; GET should just render the form
        fields = [
            "Temperature", "RH", "WS", "Rain",
            "FFMC", "DMC", "ISI", "Classes", "Region"
        ]

        # Collect raw strings and validate presence
        raw = {}
        missing = []
        for f in fields:
            v = request.form.get(f, None)
            if v is None or str(v).strip() == "":
                missing.append(f)
            else:
                raw[f] = v

        if missing:
            return render_template('home.html', error=f"Missing fields: {', '.join(missing)}")

        # Convert to floats with validation
        try:
            temperature = float(raw["Temperature"])
            rh = float(raw["RH"])
            ws = float(raw["WS"])
            rain = float(raw["Rain"])
            ffmc = float(raw["FFMC"])
            dmc = float(raw["DMC"])
            isi = float(raw["ISI"])
            classes = float(raw["Classes"])
            region = float(raw["Region"])
        except ValueError as ve:
            return render_template('home.html', error="Please enter valid numeric values for all fields.")

        # Build numpy array in correct shape
        input_data = np.array([[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]])

        try:
            # Scale then predict
            scaled = scaler_model.transform(input_data)  # transform, not predict
            print("prediction: ", ridge_model.predict(scaled))
            prediction = ridge_model.predict(scaled)[0]


            return render_template('home.html', prediction=round(float(prediction), 2))

        except Exception as e:
            # Log if you want: print(e)
            return render_template('home.html', error="Model prediction failed. " + str(e))



    else:
        return render_template('home.html', prediction=None)

if __name__ == '__main__':
    application.run(host='0.0.0.0', debug=True, port=5001)


