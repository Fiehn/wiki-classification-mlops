import os
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from google.cloud import storage
from data import load_split_data  # Your existing data loading method
from model import NodeLevelGNN  # Your model definition
import shutil
import typer
from torch_geometric.loader import DataLoader

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


@app.command()
def visualize(
    bucket_name: str = typer.Argument("mlops-proj-group3-bucket", help="GCS bucket name for data"),
    source_folder: str = typer.Argument("torch_geometric_data", help="Source folder in GCS bucket"),
    local_data_folder: str = typer.Argument("data", help="Local folder to store downloaded data"),
    model_checkpoint: str = typer.Argument("models/model.pt", help="Path to the trained model checkpoint"),
    figure_name: str = typer.Option("embedding_clusters.png", help="Name of the output figure"),
):
    """Visualize model embeddings using PCA and t-SNE."""

    # Step 1: Download data from GCS
    data_path = download_from_gcs(bucket_name, source_folder, local_data_folder)

    # Step 2: Load the dataset
    dataset = load_split_data(root=data_path)

    # Step 3: Load the trained model
    model = NodeLevelGNN(
        c_in=dataset.num_node_features,
        c_hidden=16,  # Match this to your trained model's parameters
        c_out=dataset.y.max().item() + 1,
        num_layers=2,
        dp_rate=0.5,
    )
    print(f"Loading model checkpoint from: {model_checkpoint}")
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()

    # Step 4: Use DataLoader to iterate through the dataset
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    embeddings, targets = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to("cpu")  # Ensure data is on the CPU
            # Forward pass to extract embeddings
            embedding = model.predict(data)
            embeddings.append(embedding)
            targets.append(data.y)

    embeddings = torch.cat(embeddings, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()

    # Step 5: Reduce dimensionality
    if embeddings.shape[1] > 500:
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Step 6: Visualization
    ### Each class in the dataset is represented by a unique color. 
    ### The scatterplot then shows how well-separated the classes are in the embedding space.
    plt.figure(figsize=(10, 10))
    for i in range(dataset.y.max().item() + 1):  # Loop over classes
        mask = targets == i
        plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], label=str(i), alpha=0.6)
    plt.legend()
    plt.title("Embedding Clusters")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    
    # Save the figure
    output_dir = "reports/figures"
    os.makedirs(output_dir, exist_ok=True)
    figure_path = os.path.join(output_dir, figure_name)
    plt.savefig(figure_path)
    print(f"Saved visualization to {figure_path}")
    plt.close()

    # Clean up local data folder
    shutil.rmtree(local_data_folder, ignore_errors=True)
    print(f"Cleaned up local folder: {local_data_folder}")


if __name__ == "__main__":
    app()
