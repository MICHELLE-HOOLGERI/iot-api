import joblib
import numpy as np

# Correct class labels (same as your training order)
IOT_CLASSES = ["healthy", "heat_stress", "dehydration"]

class IoTModel:
    def __init__(self):
        # Load trained model
        self.model = joblib.load("iot_model.pkl")

    def predict(self, features):
        """
        features = numpy array of shape (1, 7)
        """
        # Model prediction (returns class index: 0, 1, or 2)
        pred_idx = self.model.predict(features)[0]

        # Convert index to class label
        return IOT_CLASSES[pred_idx]
