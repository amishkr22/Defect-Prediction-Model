import pandas as pd

def save_uploaded_file(file, save_path='uploaded_data.csv'):
    """Save the uploaded file to disk."""
    try:
        data = pd.read_csv(file.file)
        data.to_csv(save_path, index=False)
        return {"message": "File uploaded successfully"}
    except Exception as e:
        raise ValueError(f"Error saving file: {e}")
