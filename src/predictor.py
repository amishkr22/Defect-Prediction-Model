import joblib
import numpy as np

def load_model(model_path='models/rf_model.pkl', scaler_path='models/scaler.pkl'):
    """Load the model and scaler."""
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        raise ValueError(f"Error loading model or scaler: {e}")

def make_prediction(features, model, scaler):
    """Make a prediction using the loaded model and scaler."""
    try:
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        confidence = max(model.predict_proba(features_scaled)[0])
        return {"DefectStatus": "High" if prediction == 1 else "Low", "Confidence": confidence}
    except Exception as e:
        raise ValueError(f"Error making prediction: {e}")