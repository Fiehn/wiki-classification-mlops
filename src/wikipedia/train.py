import os
import shutil
import logging
import wandb
import typer
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch_geometric.loader import DataLoader

from google.cloud import storage
from google.cloud import secretmanager


# Local imports
from data import load_split_data, explore_splits, download_from_gcs, upload_model, download_file, prepare_data_loaders, prepare_test_loader
from model import NodeLevelGNN

# Adjust verbosity
logging.getLogger("pytorch_lightning").setLevel(logging.FATAL) # WARNING, ERROR, CRITICAL, DEBUG, INFO, FATAL
logging.getLogger("lightning").setLevel(logging.FATAL)
# Redirect stdout and stderr to /dev/null to suppress further logs
# import os
# import sys
# sys.stdout = open(os.devnull, 'w')  # Suppress standard output
# sys.stderr = open(os.devnull, 'w')  # Suppress error output


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

if "WANDB_API_KEY" not in os.environ or wandb.api.api_key == "":
        
    # Get the WandB API key from Secret Manager
    wandb_api_key = get_secret("WANDB_API_KEY")

    # Log in to WandB using the API key
    os.environ["WANDB_API_KEY"] = wandb_api_key
    
app = typer.Typer()

class DeviceInfoCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        # Get the device from the model (e.g., cuda:0, cpu, etc.)
        device = pl_module.device
        print(f"Training on device: {device}")

def initialize_model(c_in, c_out, hidden_channels, hidden_layers, dropout, learning_rate, weight_decay, optimizer_name):
    """Initialize the model and optimizer."""
    # Input validation
    if hidden_channels <= 0:
        raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")
    if hidden_layers <= 0:
        raise ValueError(f"hidden_layers must be positive, got {hidden_layers}")
    if not isinstance(optimizer_name, str) or optimizer_name not in ["Adam", "AdamW", "NAdam", "RMSprop", "SGD"]:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

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

def train_on_split(data, split_idx, hidden_channels, hidden_layers, dropout, learning_rate, weight_decay, optimizer_name, 
                   num_epochs, batch_size, model_checkpoint_callback, bucket_name, group_name, enable_early_stopping):
    """Train and evaluate the model on a specific split.
    Save one model checkpoint per split."""

    # Initialize WandbLogger
    wandb_logger = WandbLogger(project="wiki_classification", entity="mlops2025", group=group_name, name=f"{group_name}_split_{split_idx}")
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

    train_loader, val_loader, c_in, c_out = prepare_data_loaders(data, split_idx)
    model = initialize_model(c_in, c_out, hidden_channels, hidden_layers, dropout, learning_rate, weight_decay, optimizer_name)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath=f"checkpoints/split_{split_idx}",  # separate dir per split
        filename="best_model-{epoch:02d}-{val_acc:.4f}"
    )
    # Callbacks
    callbacks = [DeviceInfoCallback()]
    if model_checkpoint_callback:
        callbacks.append(checkpoint_callback)
    if enable_early_stopping:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=10))

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
    
    test_loader = prepare_test_loader(data)
    test_results = trainer.test(ckpt_path=best_ckpt_path, dataloaders=test_loader, verbose=False)
    test_acc = test_results[0].get('test_acc', None)

    # Save the model with hyperparameters
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparameters': {
            'c_hidden': hidden_channels,
            'num_layers': hidden_layers,
            'dp_rate': dropout,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'optimizer_name': optimizer_name,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
        }
    }, f"models/split_{split_idx}_model.pt")

    upload_model(bucket_name, f"models/split_{split_idx}_model.pt")
    wandb.finish()

    return val_acc, test_acc

@app.command("train_model")
def train_model(
    #data_path: str = typer.Argument(..., help="Path to the data"),
    bucket_name: str = typer.Argument("mlops-proj-group3-bucket", help="GCS bucket name for data"),
    source_folder: str = typer.Argument("torch_geometric_data", help="Source folder in GCS bucket"),
    local_data_folder: str = typer.Argument("data", help="Local folder to store downloaded data"),
    hidden_channels: int = typer.Option(115, help="Number of hidden channels"),
    hidden_layers: int = typer.Option(2, help="Number of hidden layers"),
    dropout: float = typer.Option(0.3236, help="Dropout rate"),
    learning_rate: float = typer.Option(0.001666, help="Learning rate"),
    weight_decay: float = typer.Option(1e-4, help="Weight decay"),
    num_epochs: int = typer.Option(300, help="Number of epochs"),
    num_splits: int = typer.Option(20, help="Number of splits to train on"),
    batch_size: int = typer.Option(11701, help="Batch size"),
    optimizer_name: str = typer.Option("Adam", help="Optimizer name"),
    model_checkpoint_callback: bool = typer.Option(True, help="Whether to use model checkpointing"),
    enable_early_stopping: bool = typer.Option(True, help="Whether to use early stopping"),
    ) -> None:
    """
    Main training function for both standalone runs and W&B sweeps.
    """
    pl.seed_everything(42)
    wandb.login()
    group_name = wandb.util.generate_id()

    # Download data from GCS
    data_path = download_from_gcs(bucket_name, source_folder, local_data_folder)
    data_module = load_split_data(data_path)
    data = data_module[0]

    # Param to tell how many splits to train on - check for invalid input
    if num_splits >= data.train_mask.shape[1]:
        # Run over all 20 splits and then average the results
        num_splits = data.train_mask.shape[1] 
    print(f"Total splits: {num_splits}")

    val_accuracies = []
    test_accuracies = []
    for split in range(num_splits):
        print(f"Training on split {split}")
        val_acc, test_acc = train_on_split(
            data, split, hidden_channels, hidden_layers, dropout, learning_rate, weight_decay, optimizer_name, num_epochs,
            batch_size, model_checkpoint_callback, bucket_name, group_name, enable_early_stopping)
        val_accuracies.append(val_acc)
        test_accuracies.append(test_acc)

    # Log average accuracy
    avg_val_acc = sum(val_accuracies) / len(val_accuracies)
    avg_test_acc = sum(test_accuracies) / len(test_accuracies)
    # Log to WandB
    wandb.init(project="wiki_classification", entity="mlops2025", group=group_name, name=f"{group_name}_summary")
    wandb.log({
        "avg_val_acc": avg_val_acc,
        "avg_test_acc": avg_test_acc,
    })
    wandb.finish()

if __name__ == "__main__":
    app()
