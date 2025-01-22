# Manufacturing Defect Prediction API

## Overview
This project provides a RESTful API for predicting manufacturing defects using a machine learning model. The API supports uploading a dataset, training the model, and predicting defect status based on manufacturing parameters.

## Features
1. Upload a CSV file containing manufacturing data.
2. Train a Random Classifier model to predict defect status.
3. Predict defect status for new input data with confidence scores.

## Setup Instructions

### Prerequisites
- Python 3.8 or above
- Install required dependencies using `pip`.

### Steps
1. Clone the repository:
   ```bash
   git clone <https://github.com/amishkr22/Defect-Prediction-Model.git>
   ```

2. Navigate to the project directory:
   ```bash
   cd <project_directory>
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```

5. Open the API documentation:
   Navigate to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive documentation.

## API Endpoints

### `/upload` - Upload Dataset
- **Method**: POST
- **Description**: Upload a CSV file for training.
- **Request**:
  - Form-data: `file` (CSV file containing the dataset).
- **Response**:
  ```json
  {
    "message": "File uploaded successfully"
  }
  ```

### `/train` - Train the Model
- **Method**: POST
- **Description**: Train the model on the uploaded dataset.
- **Response**:
  ```json
  {
    "accuracy": 0.88,
    "f1_score": 0.93,
    "message": "Model trained successfully"
  }
  ```

### `/predict` - Predict Defect Status
- **Method**: POST
- **Description**: Predict defect status based on input features.
- **Request**:
  - **JSON Body**:
    ```json
    {
      "ProductionVolume": 500,
      "ProductionCost": 15000,
      "SupplierQuality": 90,
      "DefectRate": 2.5,
      "MaintenanceHours": 10,
      "EnergyConsumption": 3000,
      "EnergyEfficiency": 0.3
    }
    ```
- **Response**:
  ```json
  {
    "DefectStatus": "Low",
    "Confidence": 0.85
  }
  ```

## Dependencies
- `FastAPI`
- `uvicorn`
- `scikit-learn`
- `pandas`
- `joblib`

## Directory Structure
```
Dataset/
├── manufacturing_defect_dataset.csv          # Dataset used
models/
├── rf_model.pkl                              # Prediction model
├── scaler.pkl                                # Scaling model
src/
├── __init__.py                               # Package initializer
├── data_handler.py                           # Handles data upload and processing
├── model_trainer.py                          # Model training logic
├── predictor.py                              # Prediction logic
├── utils.py                                  # Utility functions
app.py                                        # FastAPI application entry point
requirements.txt
```

## Example Usage
1. **Upload a dataset**:
   - Use Postman or cURL to send a CSV file to `/upload`.
2. **Train the model**:
   - Call the `/train` endpoint to train the model.
3. **Make predictions**:
   - Send a JSON payload to `/predict` with the required input features.

## Deployment
**Run Locally**:
   - Start the server using `uvicorn app:app --reload`.
