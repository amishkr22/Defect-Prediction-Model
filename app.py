from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from src.data_handler import save_uploaded_file
from src.model_trainer import train_model
from src.predictor import load_model, make_prediction
from src.utils import check_file_exists

app = FastAPI()

# Load model and scaler
model, scaler = None, None
if check_file_exists('models/rf_model.pkl') and check_file_exists('models/scaler.pkl'):
    model, scaler = load_model()

class PredictionInput(BaseModel):
    ProductionVolume: int
    ProductionCost: float
    SupplierQuality: float
    DefectRate: float
    MaintenanceHours: int
    EnergyConsumption: float
    EnergyEfficiency: float

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        response = save_uploaded_file(file)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
def train():
    try:
        response = train_model()
        global model, scaler
        model, scaler = load_model()  # Reload after training
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        if model is None or scaler is None:
            raise HTTPException(status_code=500, detail="Model not trained or loaded")
        
        features = [
            input_data.ProductionVolume,
            input_data.ProductionCost,
            input_data.SupplierQuality,
            input_data.DefectRate,
            input_data.MaintenanceHours,
            input_data.EnergyConsumption,
            input_data.EnergyEfficiency
        ]
        response = make_prediction(features, model, scaler)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)