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

# Chat-skelet til kørsel med cloud:


import os
from google.cloud import aiplatform
from google.cloud import storage

# Configuration
PROJECT_ID = "mlops-448012"
BUCKET_NAME = "mlops-proj-group3-bucket"
MODEL_FILENAME = "best_model.pt"
MODEL_DISPLAY_NAME = "wiki-classifier-model"  # A user-friendly name
REGION = "europe-west1"
SERVING_CONTAINER_IMAGE_URI = "europe-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.1-12:latest" # Use the correct image

def deploy_model():
    # 1. Check if model exists in bucket
    print("Checking if model exists in Cloud Storage...")
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_FILENAME)

    if not blob.exists():
        print(f"Error: Model file '{MODEL_FILENAME}' not found in bucket '{BUCKET_NAME}'.")
        return

    print("Model found in Cloud Storage.")

    # 2. Upload the model to Vertex AI Model Registry (pointing to the bucket)
    print("Uploading model to Vertex AI Model Registry (from bucket)...")
    aiplatform.init(project=PROJECT_ID, location=REGION)

    try:
        model = aiplatform.Model.upload(
            display_name=MODEL_DISPLAY_NAME,
            artifact_uri=f"gs://{BUCKET_NAME}",  # Point directly to the bucket
            serving_container_image_uri=SERVING_CONTAINER_IMAGE_URI,
        )
        print(f"Model uploaded. Model name: {model.resource_name}")

        # 3. Deploy the model to an Endpoint
        print("Deploying model to Endpoint...")
        endpoint = aiplatform.Endpoint.create(
            display_name="wiki-classifier-endpoint",
            location=REGION
        )

        model_deployed = model.deploy(
            endpoint=endpoint,
            machine_type="n1-standard-2",  # Choose an appropriate machine type
        )
        print(f"Model deployed to endpoint: {endpoint.resource_name}")
        print(f"Endpoint ID: {endpoint.name}")

    except Exception as e:
        print(f"An error occurred during model upload or deployment: {e}")

if __name__ == "__main__":
    deploy_model()

