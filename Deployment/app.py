import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("models/model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        data = [
            float(request.form.get("age")),
            float(request.form.get("cp")),
            float(request.form.get("trestbps")),
            float(request.form.get("chol")),
            float(request.form.get("thalach")),
            float(request.form.get("exang")),
            float(request.form.get("oldpeak")),
            float(request.form.get("slope")),
            float(request.form.get("ca")),
            float(request.form.get("thal"))
        ]
        scaled = scaler.transform([data])
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1]
        return render_template("index.html", prediction=pred, probability=round(prob * 100, 2))

if __name__ == '__main__':
    app.run(host="0.0.0.0")