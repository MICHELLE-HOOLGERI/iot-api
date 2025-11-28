from flask import Flask, request, jsonify
from iot_model import IoTModel
import numpy as np

app = Flask(__name__)
model = IoTModel()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Validate fields
        required = ["pulse", "body_temp", "farm_temp", "humidity",
                    "milk_yield", "feeding_time", "movement"]

        for key in required:
            if key not in data:
                return jsonify({"error": f"Missing value for: {key}"}), 400

        # Convert inputs to float and prepare in correct shape
        features = np.array([[
            float(data["pulse"]),
            float(data["body_temp"]),
            float(data["farm_temp"]),
            float(data["humidity"]),
            float(data["milk_yield"]),
            float(data["feeding_time"]),
            float(data["movement"])
        ]])

        # Get prediction from model
        result = model.predict(features)

        return jsonify({
            "model": "iot",
            "prediction": result
        })

    except Exception as e:
        # Return exact error instead of 500
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return jsonify({"message": "IoT Model API is running"})


if __name__ == "__main__":
    app.run(debug=True)
