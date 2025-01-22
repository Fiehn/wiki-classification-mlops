
# import os
# import datetime
# import json
# from contextlib import asynccontextmanager
# from typing import List

# import torch
# from torch_geometric.data import Data
# from fastapi import BackgroundTasks, FastAPI, HTTPException
# from google.cloud import storage
# from pydantic import BaseModel

# from src.wikipedia.model import NodeLevelGNN

# # Configuration
# BUCKET_NAME = "mlops-proj-group3-bucket"
# MODEL_FILE_NAME = "models/best_model.pt"  # Name of the model file in GCS
# LOCAL_MODEL_PATH = "models/best_model.pt"  # Local path to store the downloaded model
# METADATA_FILE_NAME = "models/best_model_metadata.json"  # Name of the metadata file in GCS
# LOCAL_METADATA_PATH = "models/best_model_metadata.json"  # Local path to store the downloaded metadata

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Input and output schemas
# class NodeInput(BaseModel):
#     x: List[List[float]]  # Node features as a 2D list
#     edge_index: List[List[int]]  # Edge list as a 2D list
#     # test_mask: List[bool]  # Test mask as a 2D list

# class PredictionOutput(BaseModel):
#     node_predictions: List[int]  # Predicted class for each node

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Load the model when the app starts and clean up when the app stops."""
#     global model, metadata
#     if not os.path.exists(LOCAL_MODEL_PATH):
#         download_model_from_gcp()  # Download the model from GCP
#     if not os.path.exists(LOCAL_METADATA_PATH):
#         download_file_from_gcp(METADATA_FILE_NAME, LOCAL_METADATA_PATH)  # Download metadata from GCP


#     # Load the checkpoint
#     checkpoint = torch.load(LOCAL_MODEL_PATH, map_location=device)
#     print("Loaded checkpoint hyperparameters") #,checkpoint["hyperparameters"])

#     # Load the metadata
#     with open(LOCAL_METADATA_PATH, "r") as metadata_file:
#         metadata = json.load(metadata_file)
#     print("Loaded metadata:", metadata)

#     # Extract hyperparameters from the checkpoint
#     c_hidden = checkpoint["hyperparameters"]["c_hidden"]
#     num_layers = checkpoint["hyperparameters"]["num_layers"]
#     dp_rate = checkpoint["hyperparameters"]["dp_rate"]
#     c_in = 300  # Input feature size during training (you need to set this manually)
#     c_out = 10  # Output classes during training (you need to set this manually)

#     # Initialize the GNN model
#     model = NodeLevelGNN(
#         c_in=c_in,
#         c_hidden=c_hidden,
#         c_out=c_out,
#         num_layers=num_layers,
#         dp_rate=dp_rate,
#     )

#     # Load the model weights
#     model.load_state_dict(checkpoint["model_state_dict"])  # Access the actual weights
#     model = model.to(device)
#     model.eval()
#     print("Model loaded successfully")

#     yield

#     del model


# # Initialize FastAPI app
# app = FastAPI(lifespan=lifespan)

# def download_model_from_gcp():
#     """Download the model from GCP bucket."""
#     client = storage.Client()
#     bucket = client.bucket(BUCKET_NAME)
#     blob = bucket.blob(MODEL_FILE_NAME)
#     blob.download_to_filename(LOCAL_MODEL_PATH)
#     print(f"Model {MODEL_FILE_NAME} downloaded from GCP bucket {BUCKET_NAME}.")

# def download_file_from_gcp(file_name: str, local_path: str):
#     """Download a file from the GCP bucket."""
#     client = storage.Client()
#     bucket = client.bucket(BUCKET_NAME)
#     blob = bucket.blob(file_name)
#     blob.download_to_filename(local_path)
#     print(f"File {file_name} downloaded from GCP bucket {BUCKET_NAME} to {local_path}.")

# def save_prediction_to_gcp(input_data: dict, predictions: List[int]):
#     """Save the prediction results to GCP bucket."""
#     client = storage.Client()
#     bucket = client.bucket(BUCKET_NAME)
#     time = datetime.datetime.now(tz=datetime.UTC)

#     # Prepare prediction data
#     data = {
#         "input_data": input_data,
#         "predictions": predictions,
#         "timestamp": time.isoformat(),
#     }

#     blob = bucket.blob(f"gnn_prediction_{time}.json")
#     blob.upload_from_string(json.dumps(data))
#     print("Prediction saved to GCP bucket.")



# @app.post("/predict", response_model=PredictionOutput)
# async def predict_node_classes(node_input: NodeInput, background_tasks: BackgroundTasks):
#     """Predict node classes for a graph."""
#     try:
#         # print(f"Received test_mask: {node_input.test_mask}")  # Debug log

#         # Prepare input data
#         x = torch.tensor(node_input.x, dtype=torch.float32).to(device)
#         edge_index = torch.tensor(node_input.edge_index, dtype=torch.long).to(device)
#         # test_mask = torch.tensor(node_input.test_mask, dtype=torch.long).to(device)

#         # Create a PyTorch Geometric Data object
#         data = Data(x=x, edge_index=edge_index) #, test_mask=test_mask)

#         # Model prediction
#         with torch.no_grad():
#             predictions = model(data, mode="test")  # Mode should be set appropriately
#             predicted_classes = predictions.argmax(dim=-1).cpu().tolist()

#         # Save the prediction results asynchronously
#         background_tasks.add_task(save_prediction_to_gcp, node_input.dict(), predicted_classes)

#         return PredictionOutput(node_predictions=predicted_classes)

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e)) from e
    

############################################################################################################
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

from src.wikipedia.train import download_from_gcs
from src.wikipedia.model import NodeLevelGNN

from google.cloud import secretmanager

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

# @app.post("/predict", response_model=PredictionOutput)
# async def predict_node_classes(node_input: NodeInput, background_tasks: BackgroundTasks):
#     try:
#         # Convert input to PyTorch tensors
#         x = torch.tensor(node_input.x, dtype=torch.float32).to(device)
#         edge_index = torch.tensor(node_input.edge_index, dtype=torch.long).to(device)

#         # Create PyTorch Geometric Data object
#         data = Data(x=x, edge_index=edge_index)

#         # Perform prediction
#         with torch.no_grad():
#             predictions = model(data, mode="test")
#             predicted_classes = predictions.argmax(dim=-1).cpu().tolist()

#         # Save predictions to GCS
#         timestamp = datetime.datetime.now(tz=datetime.UTC).isoformat()
#         predictions_json = {
#             "input": node_input.dict(),
#             "predictions": predicted_classes,
#             "timestamp": timestamp
#         }
#         predictions_filename = f"predictions/gnn_predictions_{timestamp}.json"
#         background_tasks.add_task(upload_file_to_gcs, BUCKET_NAME, predictions_filename, json.dumps(predictions_json))

#         return PredictionOutput(node_predictions=predicted_classes)


# @app.post("/predict", response_model=PredictionOutput)
# async def predict_node_classes(node_input: NodeInput, background_tasks: BackgroundTasks):
#     try:
#         # Convert input to PyTorch tensors
#         x = torch.tensor(node_input.x, dtype=torch.float32).to(device)
#         edge_index = torch.tensor(node_input.edge_index, dtype=torch.long).to(device)

#         # Download and prepare test data
#         data_path = download_from_gcs(BUCKET_NAME, "torch_geometric_data", LOCAL_DATA_FOLDER)
#         dataset = WikiCS(root=data_path, is_undirected=True)
#         data = dataset[0]

#         # Prepare test data loader
#         test_data = data.clone()
#         test_loader = DataLoader([test_data], batch_size=1)

#         # Perform prediction
#         with torch.no_grad():
#             for batch in test_loader:
#                 batch = batch.to(device)
#                 logits, _ = model(batch, mode="test")  # Unpack logits and ignore loss/accuracy
#                 predicted_classes = logits.argmax(dim=-1).cpu().tolist()

#         # Show results
#         trainer = pl.Trainer(
#             accelerator="auto",
#             enable_progress_bar=True,
#             enable_model_summary=True,
#         )
        
#         # Run test
#         test_results = trainer.test(model, test_loader, verbose=True)
#         print(f"Test Results: {test_results}")

#         # Save predictions to GCS
#         timestamp = datetime.datetime.now(tz=datetime.UTC).isoformat()
#         predictions_json = {
#             "input": node_input.dict(),
#             "predictions": predicted_classes,
#             "timestamp": timestamp
#         }
#         predictions_filename = f"predictions/gnn_predictions_{timestamp}.json"
#         background_tasks.add_task(upload_file_to_gcs, BUCKET_NAME, predictions_filename, json.dumps(predictions_json))

#         return PredictionOutput(node_predictions=predicted_classes)

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

        return PredictionOutput(node_predictions=predicted_classes)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# Run: uvicorn src.wikipedia.api:app --reload


# curl -X POST "http://127.0.0.1:8000/predict" \
# -H "Content-Type: application/json" \
# -d '{
#   "x": [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 
#         [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], 
#         [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
#   ],
#   "edge_index": [[0, 1, 2], [1, 2, 0]], 
#     "test_mask": [true, false, true]
# }'



# # load the model from file pt 
# import torch
# model = torch.load("/Users/clarareginekold/Desktop/dtu/DTU/02476 Machine Learning Operations/wiki-classification-mlops/models/best_model.pt")

# # what is in the model 
# print(model.keys())  # Check available keys
# print(model["model_state_dict"])  # Check available keys
# print(model['hyperparameters'])  # Check available keys