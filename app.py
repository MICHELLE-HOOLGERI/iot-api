from flask import Flask, request, jsonify
from models.iot_model import IoTModel

app = Flask(__name__)
model = IoTModel()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    result = model.predict(data)

    return jsonify({
        "model": "iot",
        "prediction": result
    })

@app.route("/")
def home():
    return jsonify({"message": "IoT Model API is running"})

if __name__ == "__main__":
    app.run(debug=True)
