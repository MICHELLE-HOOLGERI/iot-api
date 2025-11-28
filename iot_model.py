import joblib
import numpy as np

IOT_CLASSES = ["healthy", "heat_stress", "dehydration"]

class IoTModel:
    def __init__(self):
        self.model = joblib.load("iot_model.pkl")

    def predict(self, data):
        features = np.array([[
            data.get("pulse"),
            data.get("body_temp"),
            data.get("farm_temp"),
            data.get("humidity"),
            data.get("milk_yield"),
            data.get("feeding_time"),
            data.get("movement")
        ]])

        pred_idx = self.model.predict(features)[0]
        return IOT_CLASSES[pred_idx]
