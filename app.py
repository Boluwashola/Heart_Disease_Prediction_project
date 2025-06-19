import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("Models/model.pkl", "rb"))
scaler = pickle.load(open("Models/scaler.pkl", "rb"))

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
        prob = model.predict_proba(scaled)[0]

        print("Model classes:", model.classes_)
        print("Probabilities:", prob)

        
        import matplotlib.pyplot as plt
        import os

        classes = model.classes_

        plt.figure(figsize=(6, 4))
        plt.bar([str(classes[0]), str(classes[1])], prob, color=['blue', 'red'])
        plt.title("Prediction Probability")
        plt.ylim([0, 1])
        plt.ylabel("Probability")
        plt.grid(axis='y')
        image_path = os.path.join('static', 'prob_plot.png')
        plt.savefig(image_path)
        plt.close()
        
        return render_template("index.html", prediction=pred, probability=round(prob[1] * 100, 2), image_path=image_path)

if __name__ == '__main__':
    app.run(host="0.0.0.0")