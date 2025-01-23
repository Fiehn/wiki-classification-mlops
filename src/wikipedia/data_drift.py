import json
import os
import logging
import wandb
import typer
import pandas as pd
import numpy as np

from torch_geometric.datasets import WikiCS
from google.cloud import storage
from google.cloud import secretmanager
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
# from evidently.dashboard import Dashboard

# Local imports
from model import NodeLevelGNN


def get_secret(secret_name):
    # Create the Secret Manager client
    client = secretmanager.SecretManagerServiceClient()
    
    # Access the secret version
    project_id = "dtumlops-448012"	
    name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(name=name)
    
    # Decode the secret payload
    secret = response.payload.data.decode('UTF-8')
    return secret


if "WANDB_API_KEY" in os.environ or wandb.api.api_key == "":
# if os.environ.get("WANDB_API_KEY") == "" or os.environ.get("WANDB_API_KEY") == None or wandb.api.api_key == "":  
    # Get the WandB API key from Secret Manager
    wandb_api_key = get_secret("WANDB_API_KEY")

    # Log in to WandB using the API key
    os.environ["WANDB_API_KEY"] = wandb_api_key
    
app = typer.Typer()


def download_from_gcs(bucket_name, source_folder, destination_folder):
    """Download files from a GCS bucket."""

    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ and os.path.exists("cloud/dtumlops-448012-37e77e52cd8f.json"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cloud/dtumlops-448012-37e77e52cd8f.json"

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Ensure destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    blobs = bucket.list_blobs(prefix=source_folder)
    # print("Items in bucket:", [blob.name for blob in blobs])
    for blob in blobs:
        # Skip directories
        if blob.name.endswith("/"):
            continue

        # Construct the file path relative to the destination folder
        file_path = os.path.join(destination_folder, os.path.relpath(blob.name, source_folder))

        # Ensure the directory for the file exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Download the file to the constructed file path
        blob.download_to_filename(file_path)
        print(f"Downloaded {blob.name} to {file_path}")

    return destination_folder

 # Download original data from GCS
data_path = download_from_gcs("mlops-proj-group3-bucket", "torch_geometric_data", "data")
data_module = WikiCS(root=data_path, is_undirected=True)
reference_data = data_module[0]
# Extract features and convert to DataFrame
reference_features = reference_data.x.numpy()  # Convert to NumPy
reference_df = pd.DataFrame(reference_features)

# Download new data
data_path_current = download_from_gcs("mlops-proj-group3-bucket", "userinput", "data")

all_data = [] # Initialize a list to store all data
# Iterate through all JSON files in the folder
for file_name in os.listdir(data_path_current): 
    if file_name.endswith(".json"):
        current_file = os.path.join(data_path_current, file_name)
        
        # Load the JSON file
        with open(current_file, "r") as f:
            current_data = json.load(f)
    
        all_data.append(current_data)

# Combine all data into a single DataFrame
all_features = []

for data in all_data:
    features = np.array(data["x"])  # Extract features
    all_features.append(features)

# Concatenate all feature arrays into a single array
combined_features = np.vstack(all_features)

# Convert combined features to a DataFrame
current_df = pd.DataFrame(combined_features)

#Print shapes and column names
#print("Reference DataFrame Shape:", reference_df.shape)
#print("Reference DataFrame Columns:", reference_df.columns.tolist())
#print("Current DataFrame Shape:", current_df.shape)
#print("Current DataFrame Columns (before alignment):", current_df.columns.tolist())


# Align schemas and data types
current_df = current_df.astype(np.float32)
current_df.columns = reference_df.columns

# Initialize the report
report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])
report.run(reference_data=reference_df, current_data=current_df) # Run the report
#dashboard = Dashboard(tabs=[report])
#dashboard.show() 

report.save("reports/my_datadrit_report.json")
