import os
import typer
import torch
import pytorch_lightning as pl
from torch_geometric.datasets import WikiCS
from torch_geometric.loader import DataLoader

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from wikipedia.data import load_split_data, prepare_test_loader
from wikipedia.model import load_model
from wikipedia.gcp_utils import download_from_gcs

app = typer.Typer()

@app.command()
def test(
    checkpoint_path: str = typer.Argument("models/best_model.pt", help="Path to the trained model checkpoint"),
    bucket_name: str = typer.Argument("mlops-proj-group3-bucket", help="GCS bucket name for data"),
    source_folder: str = typer.Argument("torch_geometric_data", help="Source folder in GCS bucket"),
    local_data_folder: str = typer.Argument("data", help="Local folder to store downloaded data"),
    split_idx: int = typer.Option(0, help="Split index to test on")
):
    """Test the model on the test set."""
    
    # Download and load data
    data_path = download_from_gcs(bucket_name, source_folder, local_data_folder)
    dataset = load_split_data(root=data_path)
    data = dataset[0]
    
    # Initialize model with same architecture
    c_in = data.num_node_features
    c_out = data.y.max().item() + 1
    model = load_model(checkpoint_path, c_in, c_out)
    
    # Prepare test data
    test_loader = prepare_test_loader(data)
    
    # Initialize trainer
    trainer = pl.Trainer(
        accelerator="auto",
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Run test
    test_results = trainer.test(model=model, dataloaders=test_loader, verbose=True)
    print(f"Test Results: {test_results}")
    
    return test_results

if __name__ == "__main__":
    app()

