from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None

    if request.method == "POST":
        # Get inputs
        time = float(request.form["time"])
        amount = float(request.form["amount"])

        pca_features = []
        for i in range(1, 29):
            val = float(request.form[f"V{i}"])
            pca_features.append(val)

        # Scale amount
        amount_scaled = scaler.transform([[amount]])[0][0]

        # Prepare features
        features = np.array([time] + pca_features + [amount_scaled]).reshape(1, -1)

        # Predict
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

        prediction = "Fraud Detected 🚨" if pred == 1 else "Legitimate ✅"
        probability = f"{prob:.2f}"

    return render_template("index.html", prediction=prediction, probability=probability)

if __name__ == "__main__":
    app.run(debug=True, port=8503)
