from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib
import pandas as pd

def train_model(data_path='uploaded_data.csv', model_path='models/rf_model.pkl', scaler_path='models/scaler.pkl'):
    """Train a SVM model and save it."""
    try:
        # Load data
        data = pd.read_csv(data_path)
        features = [
            'ProductionVolume', 'ProductionCost', 'SupplierQuality', 'DefectRate',
            'MaintenanceHours', 'EnergyConsumption', 'EnergyEfficiency'
        ]
        target = 'DefectStatus'

        X = data[features]
        y = data[target]

        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train model
        model = SVC(random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Save model and scaler
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        return {"accuracy": accuracy, "f1_score": f1, "message": "Model trained successfully"}
    except Exception as e:
        raise ValueError(f"Error training model: {e}")
