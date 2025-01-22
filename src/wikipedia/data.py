###################################################################################################
### Load the data, saves it locally, then upload to GCS, and delete the local version

import argparse
import glob
import os
import shutil

from google.cloud import storage
from torch_geometric.datasets import WikiCS

# Set the path to your service account key


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

# def load_data(root="data/"):
#     """Load the WikiCS dataset."""
#     dataset = WikiCS(root=root, is_undirected=True)
#     # Collapse the masks into a single mask
#     split_index = 0
#     dataset.train_mask = dataset.train_mask[split_index]
#     dataset.val_mask = dataset.val_mask[split_index]

#     # dataset.train_mask = dataset.train_mask.sum(dim=1).bool()
#     # dataset.val_mask = dataset.val_mask.sum(dim=1).bool()

#     return dataset

def load_split_data(root="data/"):
    dataset = WikiCS(root=root, is_undirected=True)
    return dataset


def explore_splits(dataset):
    num_splits = dataset.train_mask.shape[1]
    print(f"There are {num_splits} training/validation splits.\n")
    for i in range(num_splits):
        train_count = dataset.train_mask[:, i].sum().item()
        val_count   = dataset.val_mask[:, i].sum().item()
        stop_count  = dataset.stopping_mask[:, i].sum().item()
        print(f"Split {i}:")
        print(f"  Training nodes: {train_count}")
        print(f"  Validation nodes: {val_count}")
        print(f"  Stopping nodes: {stop_count}\n")
    # Test mask is a single vector:
    test_count = dataset.test_mask.sum().item()
    print("Test set nodes:", test_count)

    print(dataset.x.shape)
    print(dataset.edge_index.shape)
    print(dataset.y.shape)
    print(dataset.train_mask.shape)
    print(dataset.val_mask.shape)
    print(dataset.test_mask.shape)
    # We have one global test set of 5847
    print(dataset.test_mask.shape)



def explore_splits2():
    dataset = WikiCS(root="data/", is_undirected=True)
    data = dataset[0]
    num_splits = data.train_mask.shape[1]
    print(f"There are {num_splits} training/validation splits.\n")
    for i in range(num_splits):
        train_count = data.train_mask[:, i].sum().item()
        val_count   = data.val_mask[:, i].sum().item()
        stop_count  = data.stopping_mask[:, i].sum().item()
        print(f"Split {i}:")
        print(f"  Training nodes: {train_count}")
        print(f"  Validation nodes: {val_count}")
        print(f"  Stopping nodes: {stop_count}\n")
    # Test mask is a single vector:
    test_count = data.test_mask.sum().item()
    print("Test set nodes:", test_count) 
  

def cleanup_local_data(folder):
    """Delete the local data folder after upload."""
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"Deleted local folder: {folder}")
    else:
        print(f"Folder not found: {folder}")


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Upload processed data to Google Cloud Storage and clean up local data."
    )
    parser.add_argument("--bucket_name", type=str, required=True, help="GCS bucket name")
    parser.add_argument(
        "--source_folder", type=str, required=True, help="Local source folder containing data to upload"
    )
    parser.add_argument("--destination_folder", type=str, required=True, help="Destination folder in GCS bucket")

    args = parser.parse_args()

    # Load the data
    load_split_data()

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
