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
    try:
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
        
        import matplotlib.pyplot as plt
        import os
        
        labels = ['No Heart Disease', 'Heart Disease']
        colors = ['green', 'red']
        
        plt.figure(figsize=(6, 3))
        bars = plt.bar(labels, prob, color=colors, alpha=0.7)
        plt.title("Risk Assessment", fontsize=14, fontweight='bold')
        plt.ylim([0, 1])
        plt.ylabel("Probability", fontsize=10)
        
        for bar, p in zip(bars, prob):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{p:.2%}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        image_path = os.path.join('static', 'prob_plot.png')
        plt.savefig(image_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return render_template("index.html", prediction=pred, probability=round(prob[1] * 100, 2), image_path=image_path)
    
    except Exception as e:
        return render_template("index.html", error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)