from flask import Flask, request, jsonify
from iot_model import IoTModel

app = Flask(__name__)
model = IoTModel()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Convert all received values to float (IMPORTANT FIX)
    features = {
        "pulse": float(data.get("pulse")),
        "body_temp": float(data.get("body_temp")),
        "farm_temp": float(data.get("farm_temp")),
        "humidity": float(data.get("humidity")),
        "milk_yield": float(data.get("milk_yield")),
        "feeding_time": float(data.get("feeding_time")),
        "movement": float(data.get("movement"))
    }

    result = model.predict(features)

    return jsonify({
        "model": "iot",
        "prediction": result
    })

@app.route("/")
def home():
    return jsonify({"message": "IoT Model API is running"})

if __name__ == "__main__":
    app.run(debug=True)

