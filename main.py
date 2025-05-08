from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load a pre-trained model
model = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    player = data['player']
    stat = data['stat']

    # Dummy input for now
    features = np.array([0.298, 0.421, 0.570, 39]).reshape(1, -1)
    prediction = model.predict_proba(features)[0][1]
    confidence = f"{int(prediction * 100)}%"

    return jsonify({
        "player": player,
        "stat": stat,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run()
