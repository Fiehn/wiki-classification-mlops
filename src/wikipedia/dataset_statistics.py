import matplotlib.pyplot as plt
import os
import torch
import typer
from data import load_split_data  # Adjust based on your implementation
from google.cloud import storage
from data import load_split_data  # Adjust based on your implementation


app = typer.Typer()

def download_from_gcs(bucket_name, source_folder, destination_folder):
    """Download files from a GCS bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Ensure destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    blobs = bucket.list_blobs(prefix=source_folder)
    for blob in blobs:
        if not blob.name.endswith("/"):  # Skip directories
            file_path = os.path.join(destination_folder, os.path.relpath(blob.name, source_folder))
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            blob.download_to_filename(file_path)
            print(f"Downloaded {blob.name} to {file_path}")
    return destination_folder


def dataset_statistics(
        bucket_name: str = typer.Argument(..., help="GCS bucket name for data"),
        source_folder: str = typer.Argument(..., help="Source folder in GCS bucket"),
        local_data_folder: str = typer.Argument("data", help="Local folder to store downloaded data")):
    """Compute dataset statistics."""
    
    # Download data from GCS
    data_path = download_from_gcs(bucket_name, source_folder, local_data_folder)

    # Load the dataset from the downloaded data
    dataset = load_split_data(root=data_path)

    # Basic dataset statistics
    print(f"Dataset name: {type(dataset).__name__}")
    print(f"Number of samples: {dataset.x.shape[0]}")
    print(f"Number of features per node: {dataset.num_node_features}")
    print(f"Number of classes: {dataset.y.max().item() + 1}")

    # Fix train and validation masks (reduce to 1D)
    train_mask = dataset.train_mask.any(dim=1)  # Combine folds using logical OR
    val_mask = dataset.val_mask.any(dim=1)

    print(f"Training samples: {train_mask.sum().item()}")
    print(f"Validation samples: {val_mask.sum().item()}")
    print(f"Test samples: {dataset.test_mask.sum().item()}\n")
    
    # Use masks to filter labels
    labels = dataset.y
    train_labels = labels[train_mask]
    val_labels = labels[val_mask]
    test_labels = labels[dataset.test_mask]

    # Compute label distributions
    train_label_distribution = torch.bincount(train_labels)
    val_label_distribution = torch.bincount(val_labels)
    test_label_distribution = torch.bincount(test_labels)

    # Create the "tests/datastats" subfolder
    output_folder = os.path.join("tests", "datastats")
    os.makedirs(output_folder, exist_ok=True)

    # Plot and save label distribution for training set
    plt.bar(torch.arange(train_label_distribution.size(0)), train_label_distribution)
    plt.title("Train Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_folder, "train_label_distribution.png"))
    plt.close()

    # Plot and save label distribution for validation set
    plt.bar(torch.arange(val_label_distribution.size(0)), val_label_distribution)
    plt.title("Validation Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_folder, "val_label_distribution.png"))
    plt.close()

    # Plot and save label distribution for test set
    plt.bar(torch.arange(test_label_distribution.size(0)), test_label_distribution)
    plt.title("Test Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_folder, "test_label_distribution.png"))
    plt.close()

    print("Dataset statistics and label distribution plots have been saved.")


if __name__ == "__main__":
    typer.run(dataset_statistics)

# python src/wikipedia/dataset_statistics.py mlops-proj-group3-bucket torch_geometric_data