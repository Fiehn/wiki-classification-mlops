import os
import google.cloud.secretmanager_v1 as secretmanager
import wandb
import glob
from google.cloud import storage
import shutil
import json
import pandas as pd
import numpy as np
import logging

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

def validate_wandb_api(wand_key=os.environ["WANDB_API_KEY"]):
    if wand_key == "":
        wand_key = get_secret("WANDB_API_KEY")
        os.environ["WANDB_API_KEY"] = wand_key
    
    if wand_key != "":
        wandb.login()
    return None

def upload_to_gcs(bucket_name, source_folder, destination_folder):
    """Uploads files from a local folder to a GCS bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Iterate through all files in the source folder
    for file_path in glob.glob(f"{source_folder}/**", recursive=True):
        if os.path.isfile(file_path):  # Only upload files
            destination_blob_name = os.path.join(destination_folder, os.path.relpath(file_path, source_folder))
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(file_path)
            print(f"Uploaded {file_path} to {destination_blob_name} in bucket {bucket_name}.")

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

def upload_file_to_gcs(bucket_name: str, destination_blob_name: str, content: str = ""):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    if content == "":
        blob.upload_from_filename(destination_blob_name)
    else:
        blob.upload_from_string(content)
    print(f"Uploaded content to {destination_blob_name}")

def download_file_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

def get_user_data(bucket_name: str, source_folder: str, user_data_path: str = "data/user") -> pd.DataFrame:
    try:
        # Clear and create user data directory
        if os.path.exists(user_data_path):
            shutil.rmtree(user_data_path)
        os.makedirs(user_data_path, exist_ok=True)

        # Download data
        data_path_current = download_from_gcs(bucket_name, source_folder, user_data_path)
        all_data = []

        # Process all JSON files
        for file_name in os.listdir(data_path_current):
            if file_name.endswith(".json"):
                with open(os.path.join(data_path_current, file_name), "r") as f:
                    current_data = json.load(f)

                all_data.append(current_data)

        # Extract and combine features
        all_features = [np.array(data["x"]) for data in all_data]
        combined_features = np.vstack(all_features)
        # Create DataFrame
        current_df = pd.DataFrame(combined_features).astype(np.float32)
        
        # Cleanup
        shutil.rmtree(user_data_path)
        
        return current_df

    except Exception as e:
        logging.error(f"Error processing user data: {str(e)}")
        if os.path.exists(user_data_path):
            shutil.rmtree(user_data_path)
        raise
