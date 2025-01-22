
import os
import datetime
import json
from contextlib import asynccontextmanager
from typing import List

import torch
from torch_geometric.data import Data
from fastapi import BackgroundTasks, FastAPI, HTTPException
from google.cloud import storage
from pydantic import BaseModel

from src.wikipedia.model import NodeLevelGNN

# Configuration
BUCKET_NAME = "mlops-proj-group3-bucket"
MODEL_FILE_NAME = "models/best_model.pt"  # Name of the model file in GCS
LOCAL_MODEL_PATH = "models/best_model.pt"  # Local path to store the downloaded model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Input and output schemas
class NodeInput(BaseModel):
    x: List[List[float]]  # Node features as a 2D list
    edge_index: List[List[int]]  # Edge list as a 2D list

class PredictionOutput(BaseModel):
    node_predictions: List[int]  # Predicted class for each node


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model when the app starts and clean up when the app stops."""
    global model
    if not os.path.exists(LOCAL_MODEL_PATH):
        download_model_from_gcp()  # Download the model from GCP

    # Load the checkpoint
    checkpoint = torch.load(LOCAL_MODEL_PATH, map_location=device)
    print("Loaded checkpoint hyperparameters") #,checkpoint["hyperparameters"])

    # Extract hyperparameters from the checkpoint
    c_hidden = checkpoint["hyperparameters"]["c_hidden"]
    num_layers = checkpoint["hyperparameters"]["num_layers"]
    dp_rate = checkpoint["hyperparameters"]["dp_rate"]
    c_in = 300  # Input feature size during training (you need to set this manually)
    c_out = 10  # Output classes during training (you need to set this manually)

    # Initialize the GNN model
    model = NodeLevelGNN(
        c_in=c_in,
        c_hidden=c_hidden,
        c_out=c_out,
        num_layers=num_layers,
        dp_rate=dp_rate,
    )

    # Load the model weights
    model.load_state_dict(checkpoint["model_state_dict"])  # Access the actual weights
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")

    yield

    del model


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)


def download_model_from_gcp():
    """Download the model from GCP bucket."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_FILE_NAME)
    blob.download_to_filename(LOCAL_MODEL_PATH)
    print(f"Model {MODEL_FILE_NAME} downloaded from GCP bucket {BUCKET_NAME}.")


def save_prediction_to_gcp(input_data: dict, predictions: List[int]):
    """Save the prediction results to GCP bucket."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    time = datetime.datetime.now(tz=datetime.UTC)

    # Prepare prediction data
    data = {
        "input_data": input_data,
        "predictions": predictions,
        "timestamp": time.isoformat(),
    }

    blob = bucket.blob(f"gnn_prediction_{time}.json")
    blob.upload_from_string(json.dumps(data))
    print("Prediction saved to GCP bucket.")


@app.post("/predict", response_model=PredictionOutput)
async def predict_node_classes(node_input: NodeInput, background_tasks: BackgroundTasks):
    """Predict node classes for a graph."""
    try:
        # Prepare input data
        x = torch.tensor(node_input.x, dtype=torch.float32).to(device)
        edge_index = torch.tensor(node_input.edge_index, dtype=torch.long).to(device)

        # Create a PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index)

        # Model prediction
        with torch.no_grad():
            predictions = model(data, mode="test")  # Mode should be set appropriately
            predicted_classes = predictions.argmax(dim=-1).cpu().tolist()

        # Save the prediction results asynchronously
        background_tasks.add_task(save_prediction_to_gcp, node_input.dict(), predicted_classes)

        return PredictionOutput(node_predictions=predicted_classes)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    


# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# checkpoint = torch.load("/Users/clarareginekold/Desktop/dtu/DTU/02476 Machine Learning Operations/wiki-classification-mlops/models/best_model.pt", map_location=device)
# print(checkpoint.keys())  # Check available keys

# # If "hyperparameters" is available in the checkpoint, inspect it
# if "hyperparameters" in checkpoint:
#     print(checkpoint["hyperparameters"])
