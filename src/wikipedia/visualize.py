import os
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from data import load_split_data  
from model import NodeLevelGNN 
import shutil
import typer
from torch_geometric.loader import DataLoader
from gcp_utils import download_from_gcs
from data import load_split_data
from model import load_model

app = typer.Typer()

@app.command()
def visualize(
    checkpoint_path: str = typer.Argument("models/best_model.pt", help="Path to the trained model checkpoint"),
    bucket_name: str = typer.Argument("mlops-proj-group3-bucket", help="GCS bucket name for data"),
    source_folder: str = typer.Argument("torch_geometric_data", help="Source folder in GCS bucket"),
    local_data_folder: str = typer.Argument("data", help="Local folder to store downloaded data"),
    # model_checkpoint: str = typer.Argument("models/best_model.pt", help="Path to the trained model checkpoint"),
    figure_name: str = typer.Option("embedding_clusters.png", help="Name of the output figure"),
):
    """Visualize model embeddings using PCA and t-SNE."""

    data_path = download_from_gcs(bucket_name, source_folder, local_data_folder)
    dataset = load_split_data(root=data_path)
    data = dataset[0]

    # Initialize model with same architecture
    c_in = data.num_node_features
    c_out = data.y.max().item() + 1
    model = load_model(checkpoint_path, c_in, c_out)
   
    model.eval()

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

    if embeddings.shape[1] > 500:
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

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
