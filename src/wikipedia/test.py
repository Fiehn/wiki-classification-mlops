import os
import typer
import torch
import pytorch_lightning as pl
from torch_geometric.datasets import WikiCS
from torch_geometric.loader import DataLoader

from model import NodeLevelGNN
from train import download_from_gcs

app = typer.Typer()

def load_model(checkpoint_path, c_in, c_out):
    """Load model and hyperparameters from checkpoint."""
    # Load checkpoint
    try: 
        checkpoint = torch.load(checkpoint_path)
        #print(f"Checkpoint loaded successfully. {checkpoint.keys()}")
        # Get hyperparameters from checkpoint
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return
    
    hyperparameters = checkpoint['hyper_parameters']
    #print(f"Hyperparameters: {hyperparameters}")
    
    # Initialize model with saved hyperparameters
    model = NodeLevelGNN(
        c_in=c_in,
        c_hidden=hyperparameters['c_hidden'],
        c_out=c_out,
        num_layers=hyperparameters['num_layers'],
        dp_rate=hyperparameters['dp_rate'],
    )
    
    # Load the state dict
    model.load_state_dict(checkpoint['state_dict'])
    
    # # Configure optimizer if needed
    # model.configure_optimizers(
    #     learning_rate=hyperparameters['learning_rate'],
    #     weight_decay=hyperparameters['weight_decay'],
    #     optimizer_name=hyperparameters['optimizer_name']
    # )
    
    return model

@app.command()
def test(
    checkpoint_path: str = typer.Argument(..., help="Path to model checkpoint"),
    bucket_name: str = typer.Argument("mlops-proj-group3-bucket", help="GCS bucket name for data"),
    source_folder: str = typer.Argument("torch_geometric_data", help="Source folder in GCS bucket"),
    local_data_folder: str = typer.Argument("data", help="Local folder to store downloaded data"),
    split_idx: int = typer.Option(0, help="Split index to test on")
):
    """Test the model on the test set."""
    
    # Download and load data
    data_path = download_from_gcs(bucket_name, source_folder, local_data_folder)
    dataset = WikiCS(root=data_path, is_undirected=True)
    data = dataset[0]
    
    # Initialize model with same architecture
    c_in = data.num_node_features
    c_out = data.y.max().item() + 1
    model = load_model(checkpoint_path, c_in, c_out)
    
    # Prepare test data
    test_data = data.clone()
    test_loader = DataLoader([test_data], batch_size=1)
    
    # Initialize trainer
    trainer = pl.Trainer(
        accelerator="auto",
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Run test
    test_results = trainer.test(model, test_loader, verbose=True)
    print(f"Test Results: {test_results}")
    
    return test_results

if __name__ == "__main__":
    app()

