'''
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from .model import NodeLevelGNN

print("Starting API...")

app = FastAPI()
print("FastAPI initialized...")

print("Loading model...")
try:
    model = NodeLevelGNN.load_from_checkpoint("models/model.pt")  # Indlæs modellen
    print("Model loaded successfully!") # hvis det lykkes at indlæse
    model.eval()
    print("Model set to eval mode.")
except Exception as e:
    print(f"Error loading model: {e}") # hvis indlæsning fejler
    raise #Re-raise exception så uvicorn stopper og fejl vises i terminalen

class TextData(BaseModel):
    text: str

@app.post("/predict")
async def predict(data: TextData):
    print(f"Received prediction request: {data}") # se input data
    with torch.no_grad():
        try:
            prediction = model(data.text)  
            print(f"Raw prediction: {prediction}") #se output data
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {"error": str(e)} # Returner en fejlbesked i JSON format
    return {"prediction": prediction.tolist()}  # Konverter til liste for JSON

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

'''

from fastapi import FastAPI
from pydantic import BaseModel
from google.cloud import storage
import torch  # Or your model's framework
import os

# Configuration
PROJECT_ID = "mlops-448012"
BUCKET_NAME = "mlops-proj-group3-bucket"
MODEL_FILENAME = "best_model.pt"
CREDENTIALS_JSON = "path/to/your/service_account_key.json" # Path to service account key

app = FastAPI()

# Load the model outside the request handler (on startup)
try:
    storage_client = storage.Client.from_service_account_json(CREDENTIALS_JSON)
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_FILENAME)
    blob.download_to_filename("local_model.pt") # download to local file
    
    from .model import NodeLevelGNN # import your model class
    model = NodeLevelGNN.load_from_checkpoint("local_model.pt")
    os.remove("local_model.pt") # remove the file after loading
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise # Re-raise exception to stop uvicorn if model loading fails


class TextData(BaseModel):
    text: str

@app.post("/predict")
async def predict(data: TextData):
    try:
        with torch.no_grad():
            prediction = model(data.text)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": str(e)}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)