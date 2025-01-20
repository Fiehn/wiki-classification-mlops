###################################################################################################
### Load the data, saves it locally, then upload to GCS, and delete the local version

from torch_geometric.datasets import WikiCS
from google.cloud import storage
import os
import glob
import argparse
import shutil

# Set the path to your service account key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cloud/dtumlops-448012-37e77e52cd8f.json"

# Initialize the Storage client
client = storage.Client()

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

def load_data():
    """Load the WikiCS dataset."""
    dataset = WikiCS(root="data/", is_undirected=True)
    # Collapse the masks into a single mask
    dataset.train_mask = dataset.train_mask.sum(dim=1).bool()
    dataset.val_mask = dataset.val_mask.sum(dim=1).bool()
    return dataset

def load_split_data(root="data/"):
    dataset = WikiCS(root=root, is_undirected=True)
    return dataset

def cleanup_local_data(folder):
    """Delete the local data folder after upload."""
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"Deleted local folder: {folder}")
    else:
        print(f"Folder not found: {folder}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Upload processed data to Google Cloud Storage and clean up local data.")
    parser.add_argument("--bucket_name", type=str, required=True, help="GCS bucket name")
    parser.add_argument("--source_folder", type=str, required=True, help="Local source folder containing data to upload")
    parser.add_argument("--destination_folder", type=str, required=True, help="Destination folder in GCS bucket")

    args = parser.parse_args()

    # Load the data
    load_data()

    # Upload data to GCS using the provided arguments
    upload_to_gcs(args.bucket_name, args.source_folder, args.destination_folder)

    # Delete local data after upload
    cleanup_local_data(args.source_folder)


# Run in terminal: python src/wikipedia/data.py --bucket_name mlops-proj-group3-bucket --source_folder data --destination_folder torch_geometric_data
# Grant bucket access in terminal: gcloud projects add-iam-policy-binding dtumlops-448012 \    --member="serviceAccount:470583037705-compute@developer.gserviceaccount.com" \    --role="roles/storage.objectAdmin"






###################################################################################################
### Version where we load the data and saves it locally and then upload to GCS, and keep the local version

# from torch_geometric.datasets import WikiCS
# from google.cloud import storage
# import os
# import glob
# import argparse

# # Set the path to your service account key
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cloud/dtumlops-448012-37e77e52cd8f.json"

# # Now you can initialize the Storage client
# client = storage.Client()

# def upload_to_gcs(bucket_name, source_folder, destination_folder):
#     """Uploads files from a local folder to a GCS bucket."""
#     client = storage.Client()
#     bucket = client.bucket(bucket_name)

#     # Iterate through all files in the source folder
#     for file_path in glob.glob(f"{source_folder}/**", recursive=True):
#         if os.path.isfile(file_path):  # Only upload files
#             destination_blob_name = os.path.join(destination_folder, os.path.relpath(file_path, source_folder))
#             blob = bucket.blob(destination_blob_name)
#             blob.upload_from_filename(file_path)
#             print(f"Uploaded {file_path} to {destination_blob_name} in bucket {bucket_name}.")

# def load_data():
#     dataset = WikiCS(root="data/", is_undirected=True)
#     # collapse the masks into a single mask
#     dataset.train_mask = dataset.train_mask.sum(dim=1).bool()
#     dataset.val_mask = dataset.val_mask.sum(dim=1).bool()
#     return dataset

# def load_split_data():
#     dataset = WikiCS(root="data/", is_undirected=True)
#     return dataset

# if __name__ == "__main__":
#     # Set up argument parsing
#     parser = argparse.ArgumentParser(description="Upload processed data to Google Cloud Storage")
#     parser.add_argument("--bucket_name", type=str, required=True, help="GCS bucket name")
#     parser.add_argument("--source_folder", type=str, required=True, help="Local source folder containing data to upload")
#     parser.add_argument("--destination_folder", type=str, required=True, help="Destination folder in GCS bucket")

#     args = parser.parse_args()

#     # Load the data
#     load_data()

#     # Upload data to GCS using the provided arguments
#     upload_to_gcs(args.bucket_name, args.source_folder, args.destination_folder)

# # Run in terminal: python src/wikipedia/data.py --bucket_name mlops-proj-group3-bucket --source_folder data --destination_folder torch_geometric_data


