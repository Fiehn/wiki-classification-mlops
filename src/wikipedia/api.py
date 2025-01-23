import os
import datetime
import json
from typing import List
from contextlib import asynccontextmanager
from torch_geometric.datasets import WikiCS
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl

import torch
from torch_geometric.data import Data
from fastapi import BackgroundTasks, FastAPI, HTTPException
from google.cloud import storage
from pydantic import BaseModel

# from src.wikipedia.train import download_from_gcs
# from src.wikipedia.model import NodeLevelGNN

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from model import NodeLevelGNN

# Global variables
model = None  # Initialize as None

# Configuration
BUCKET_NAME = "mlops-proj-group3-bucket"
MODEL_FILE_NAME = "models/best_model.pt"
METADATA_FILE_NAME = "models/best_model_metadata.json"
LOCAL_MODEL_PATH = "models/best_model.pt"
LOCAL_METADATA_PATH = "models/best_model_metadata.json"
LOCAL_DATA_FOLDER = "data"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Input and output schemas
class NodeInput(BaseModel):
    x: List[List[float]]  # Node features as a 2D list
    edge_index: List[List[int]]  # Edge list as a 2D list

class PredictionOutput(BaseModel):
    node_predictions: List[int]  # Predicted class for each node

# FastAPI app
app = FastAPI()

def download_file_from_gcs(bucket_name: str, source_blob_name: str, destination_file_name: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}")

def upload_file_to_gcs(bucket_name: str, destination_blob_name: str, content: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(content)
    print(f"Uploaded predictions to {destination_blob_name}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    # Download model and metadata if not available locally
    if not os.path.exists(LOCAL_MODEL_PATH):
        download_file_from_gcs(BUCKET_NAME, MODEL_FILE_NAME, LOCAL_MODEL_PATH)
    if not os.path.exists(LOCAL_METADATA_PATH):
        download_file_from_gcs(BUCKET_NAME, METADATA_FILE_NAME, LOCAL_METADATA_PATH)

    # Load model checkpoint
    checkpoint = torch.load(LOCAL_MODEL_PATH, map_location=device)

    # Load hyperparameters
    hyperparameters = checkpoint['hyperparameters']
    c_hidden = hyperparameters['c_hidden']
    num_layers = hyperparameters['num_layers']
    dp_rate = hyperparameters['dp_rate']
    c_in = 300  # Input feature size
    c_out = 10  # Output classes

    # Initialize model
    model = NodeLevelGNN(c_in=c_in, c_hidden=c_hidden, c_out=c_out, num_layers=num_layers, dp_rate=dp_rate)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print("Model loaded and ready for inference.")

    yield
    del model

# Attach the lifespan context to the FastAPI app
app = FastAPI(lifespan=lifespan)


@app.get("/")
async def read_root():
    return {"message": "Welcome to the MNIST model inference API!"}

@app.post("/predict", response_model=PredictionOutput)
async def predict_node_classes(node_input: NodeInput, background_tasks: BackgroundTasks):
    try:
        # Convert input to PyTorch tensors
        x = torch.tensor(node_input.x, dtype=torch.float32).to(device)
        edge_index = torch.tensor(node_input.edge_index, dtype=torch.long).to(device)

        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index)

        # Perform prediction using the model's predict method
        with torch.no_grad():
            logits = model.predict(data)  # Use predict to get logits
            predicted_classes = logits.argmax(dim=-1).cpu().tolist()

        # Save predictions to GCS
        timestamp = datetime.datetime.now(tz=datetime.UTC).isoformat()
        predictions_json = {
            "input": node_input.dict(),
            "predictions": predicted_classes,
            "timestamp": timestamp,
        }
        predictions_filename = f"predictions/gnn_predictions_{timestamp}.json"
        background_tasks.add_task(upload_file_to_gcs, BUCKET_NAME, predictions_filename, json.dumps(predictions_json))

        # userinput_filename = f"userinput/user_input_{timestamp}.json"
        # background_tasks.add_task(upload_file_to_gcs, BUCKET_NAME, userinput_filename, json.dumps(data))

        # Convert PyTorch Geometric Data object to a JSON-compatible dictionary
        data_dict = {
            "x": data.x.tolist(),  # Convert tensor to list
            "edge_index": data.edge_index.tolist()  # Convert tensor to list
        }

        # Save as JSON
        userinput_filename = f"userinput/user_input_{timestamp}.json"
        background_tasks.add_task(upload_file_to_gcs, BUCKET_NAME, userinput_filename, json.dumps(data_dict))

        return PredictionOutput(node_predictions=predicted_classes)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e




# Run this in the first terminal - this will start the API: 
# uvicorn src.wikipedia.api:app --reload

# Run this in the second terminal - this will send a POST request to the API:
# Go into try.py and run the code to simulate user input and send a POST request to the API.
