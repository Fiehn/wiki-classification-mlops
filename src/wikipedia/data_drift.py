import os
import pandas as pd
import logging
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from google.cloud import storage
from torch_geometric.datasets import WikiCS
import json
import numpy as np
import shutil

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def generate_drift_report(reference_data, current_data, report_dir="reports"):
    """Generate drift report comparing reference and current data."""
    
    # Validate input data
    assert not reference_data.isna().any().any(), "Reference data contains NaN values"
    assert not current_data.isna().any().any(), "Current data contains NaN values"
    
    features = [f"feature_{i}" for i in range(reference_features.shape[1])]

    reference_data.columns = features
    current_data.columns = features
    
    # Setup report
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "drift_report.html")
    
    try:
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=reference_df.astype(float), 
            current_data=current_df.astype(float)
        )
        report.save_html(report_path)
        logging.info(f"Report saved to: {report_path}")
        return report_path
    except Exception as e:
        logging.error(f"Failed to generate report: {str(e)}")
        raise

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

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cloud/dtumlops-448012-37e77e52cd8f.json"

# Test the function
if __name__ == "__main__":
     # Download original data from GCS
    data_path = download_from_gcs("mlops-proj-group3-bucket", "torch_geometric_data", "data")
    data_module = WikiCS(root=data_path, is_undirected=True)
    reference_data = data_module[0]
    # Extract features and convert to DataFrame
    reference_features = reference_data.x.numpy()  # Convert to NumPy
    reference_df = pd.DataFrame(reference_features)

    current_df = get_user_data("mlops-proj-group3-bucket", "userinput")
    
    report_path = generate_drift_report(reference_df, current_df)
    print(f"Report status: {'Success' if report_path else 'Failed'}")