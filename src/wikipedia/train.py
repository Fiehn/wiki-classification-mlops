import torch
import copy
import typer
import logging
import wandb
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

import os
import shutil
from google.cloud import storage
from model import NodeLevelGNN

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch_geometric.datasets import WikiCS

from model import NodeLevelGNN
from data import load_data, load_split_data, explore_splits


# Adjust verbosity
logging.getLogger("pytorch_lightning").setLevel(logging.FATAL) # WARNING, ERROR, CRITICAL, DEBUG, INFO, FATAL
logging.getLogger("lightning").setLevel(logging.FATAL)
# Redirect stdout and stderr to /dev/null to suppress further logs
# import os
# import sys
# sys.stdout = open(os.devnull, 'w')  # Suppress standard output
# sys.stderr = open(os.devnull, 'w')  # Suppress error output

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
def train(
    bucket_name: str = typer.Argument(..., help="GCS bucket name for data"),
    source_folder: str = typer.Argument(..., help="Source folder in GCS bucket"),
    local_data_folder: str = typer.Argument("data", help="Local folder to store downloaded data"),
    hidden_channels: int = typer.Option(16, help="Number of hidden channels"),
    hidden_layers: int = typer.Option(2, help="Number of hidden layers"),
    dropout: float = typer.Option(0.5, help="Dropout rate"),
    lr: float = typer.Option(0.01, help="Learning rate"),
    num_epochs: int = typer.Option(100, help="Number of epochs"),
    batch_size: int = typer.Option(32, help="Batch size"),
    model_checkpoint_callback: bool = typer.Option(True, help="Whether to use model checkpointing"),
) -> None:
    
    # Download data from GCS
    data_path = download_from_gcs(bucket_name, source_folder, local_data_folder)

    # Initialize WandbLogger
    wandb_logger = WandbLogger(
        project="wiki_classification",
        config={
            "hidden_channels": hidden_channels,
            "hidden_layers": hidden_layers,
            "dropout": dropout,
            "lr": lr,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
        },
    )
    wandb_logger.experiment.log({"test_log": "Wandb is active!"})

    # Load the dataset from the downloaded data
    data_module = load_split_data(root=data_path)

    c_in = data_module.num_node_features
    c_out = data_module.y.max().item() + 1

class DeviceInfoCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        # Get the device from the model (e.g., cuda:0, cpu, etc.)
        device = pl_module.device
        print(f"Training on device: {device}")


def initialize_model(c_in, c_out, hidden_channels, hidden_layers, dropout, learning_rate, weight_decay, optimizer_name):
    """Initialize the model and optimizer."""

    model = NodeLevelGNN(
        c_in=c_in,
        c_hidden=hidden_channels,
        c_out=c_out,
        num_layers=hidden_layers,
        dp_rate=dropout,
    )

    model.configure_optimizers(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_name=optimizer_name,
    )
    return model

def train_on_split(data, split_idx, hidden_channels, hidden_layers, dropout, learning_rate, weight_decay, optimizer_name, num_epochs, batch_size, wandb_logger, model_checkpoint_callback):
    """Train and evaluate the model on a specific split.
    Save one model checkpoint per split."""
    train_data = data.clone()
    val_data = data.clone()
    train_data.train_mask = data.train_mask[:, split_idx]  # 1D mask for training
    train_data.val_mask = None  # Not needed during training
    val_data.val_mask = data.val_mask[:, split_idx]  # 1D mask for validation
    val_data.train_mask = None  # Not needed during validation

    train_loader = DataLoader([train_data], batch_size=1, num_workers=7, shuffle=False)
    val_loader = DataLoader([val_data], batch_size=1, num_workers=7)


    # Initialize model
    c_in = data.num_node_features
    c_out = data.y.max().item() + 1
    model = initialize_model(c_in, c_out, hidden_channels, hidden_layers, dropout, learning_rate, weight_decay, optimizer_name)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        dirpath=f"checkpoints/split_{split_idx}",  # separate dir per split
        filename="best_model-{epoch:02d}-{val_acc:.4f}"
    )
    # Callbacks
    callbacks = [DeviceInfoCallback()]
    if model_checkpoint_callback:
        callbacks.append(checkpoint_callback)

    # Trainer
    trainer = pl.Trainer(
        logger=wandb_logger, # WandbLogger integration
        max_epochs=num_epochs,
        accelerator="auto",
        callbacks=callbacks,
        enable_progress_bar=True,  # Show training progress in the terminal
        log_every_n_steps=1,
    )


    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    #val_acc = trainer.callback_metrics.get("val_acc", None) # Relies on intermediate logging - limited to last epoch

    # Validation
    best_ckpt_path = checkpoint_callback.best_model_path
    val_results = trainer.validate(ckpt_path=best_ckpt_path, dataloaders=val_loader, verbose=False)
    val_acc = val_results[0].get('val_acc', None) # more robust
    
    # Test
    test_data = data.clone()  # test_data.test_mask remains as is
    # Create a DataLoader for the test data (wrap in a list)
    test_loader = DataLoader([test_data], batch_size=1, num_workers=7)
    test_results = trainer.test(ckpt_path=best_ckpt_path, dataloaders=test_loader, verbose=False)
    test_acc = test_results[0].get('test_acc', None)

    # Save the model
    torch.save(model.state_dict(), f"models/split_{split_idx}_model.pt")

    return val_acc, test_acc

@app.command("train_model")
def train_model(
    #data_path: str = typer.Argument(..., help="Path to the data"),
    hidden_channels: int = typer.Option(32, help="Number of hidden channels"),
    hidden_layers: int = typer.Option(2, help="Number of hidden layers"),
    dropout: float = typer.Option(0.1, help="Dropout rate"),
    learning_rate: float = typer.Option(0.002, help="Learning rate"),
    weight_decay: float = typer.Option(1e-4, help="Weight decay"),
    num_epochs: int = typer.Option(200, help="Number of epochs"),
    batch_size: int = typer.Option(11701, help="Batch size"),
    optimizer_name: str = typer.Option("RMSprop", help="Optimizer name"),
    model_checkpoint_callback: bool = typer.Option(True, help="Whether to use model checkpointing"),
    ) -> None:
    """
    Main training function for both standalone runs and W&B sweeps.
    """
    pl.seed_everything(42)

    # Initialize WandbLogger
    wandb_logger = WandbLogger(project="wiki_classification", entity="mlops2025")
    # Log parameters
    wandb_logger.experiment.config.update({
        "hidden_channels": hidden_channels,
        "hidden_layers": hidden_layers,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "optimizer_name": optimizer_name,
        "model_checkpoint_callback": model_checkpoint_callback,
    })

    data_module = WikiCS(root="data/", is_undirected=True)
    data = data_module[0]
   
    # Run over all 20 splits and then average the results
    #num_splits = data_module.train_mask.shape[1]
    # num_splits = data_module[0].train_mask.shape[1]
    num_splits = 1
    print(f"Total splits: {num_splits}")

    # early_stop_callback = EarlyStopping(
    #     monitor="val_acc",
    #     mode="max",
    #     patience=10  # adjust as needed
    # )

    val_accuracies = []
    test_accuracies = []
    for split in range(num_splits):
        print(f"Training on split {split}")
        val_acc, test_acc = train_on_split(
            data, split, hidden_channels, hidden_layers, dropout, learning_rate,
            weight_decay, optimizer_name, num_epochs, batch_size, wandb_logger, model_checkpoint_callback
        )
        val_accuracies.append(val_acc)
        test_accuracies.append(test_acc)
    

    # Save the model as a W&B artifact
    # artifact = wandb.Artifact('model', type='model', metadata=dict({
    #       "Accuracies": trainer.callback_metrics,
    #        "hidden_channels": hidden_channels,
    #        "hidden_layers": hidden_layers,
    #        "dropout": dropout,
    #        "lr": lr,
    #        "num_epochs": num_epochs,
    #        "batch_size": batch_size,
    #    }))
    # artifact.add_file('models/model.pt')
    # wandb.log_artifact(artifact, aliases=["latest_model"])

    # Clean up local data folder
    # shutil.rmtree(local_data_folder, ignore_errors=True)
    # print(f"Cleaned up local folder: {local_data_folder}")

    # Log average accuracy
    avg_val_acc = sum(val_accuracies) / len(val_accuracies)
    avg_test_acc = sum(test_accuracies) / len(test_accuracies)
    print(f"Average validation accuracy across splits: {avg_val_acc:.4f}")
    print(f"Average test accuracy across splits: {avg_test_acc:.4f}")
    wandb.log({"avg_val_acc": avg_val_acc, "avg_test_acc": avg_test_acc}) # , batch_size=data.num_nodes 
    # give the best hyperparameters

    wandb.finish()

if __name__ == "__main__":
    app()

