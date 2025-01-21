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

# Chat-skelet til kørsel med cloud:

'''
import os
import subprocess
import json
from google.cloud import storage

# Konfiguration
PROJECT_ID = 
BUCKET_NAME = 
MODEL_FILENAME = 
ENDPOINT_ID = 
REGION = 

def deploy_model():
    # 1. Download modellen fra Cloud Storage
    print("Downloading model from Cloud Storage...")
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_FILENAME)
    blob.download_to_filename(MODEL_FILENAME)
    print("Model downloaded.")

    # 2. Deploy modellen til Vertex AI Endpoint
    print("Deploying model to Vertex AI Endpoint...")

    # Forbered request body
    request_body = {
        "instances": [{"text": "Dette er en test tekst."}]
    }

    # Laver curl kommandoen, så den kan køres i python
    curl_command = [
        "curl",
        "-X", "POST",
        "-H", "Authorization: Bearer $(gcloud auth print-access-token)",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(request_body),
        f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}:predict"
    ]

    try:
        # Udfører curl kommandoen
        process = subprocess.run(curl_command, capture_output=True, text=True, check=True)
        print("Model deployed and prediction made successfully!")
        print("Response:", process.stdout)

    except subprocess.CalledProcessError as e:
        print(f"Error deploying model or making prediction: {e}")
        print("Stderr:", e.stderr)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    deploy_model()

'''