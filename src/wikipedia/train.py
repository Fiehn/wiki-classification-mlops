import os
import shutil

import pytorch_lightning as pl
import torch
import typer
from data import load_data, load_split_data  # Adjust based on your implementation
from google.cloud import storage
from model import NodeLevelGNN
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.loader import DataLoader

# Logging
import wandb

from google.cloud import secretmanager

if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cloud/dtumlops-448012-e5cfd43b6fd8.json"

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

if os.environ["WANDB_API_KEY"] == "":
        
    # Get the WandB API key from Secret Manager
    wandb_api_key = get_secret("WANDB_API_KEY")

    # Log in to WandB using the API key
    os.environ["WANDB_API_KEY"] = wandb_api_key
    
app = typer.Typer()

def download_from_gcs(bucket_name, source_folder, destination_folder):
    """Download files from a GCS bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Ensure destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    blobs = bucket.list_blobs(prefix=source_folder)
    print("Items in bucket:", [blob.name for blob in blobs])
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


def upload_model(bucket_name,source_folder):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_folder)
    blob.upload_from_filename("models/model.pt")
    print(f"Uploaded model to {source_folder} in bucket {bucket_name}.")

@app.command()
def train(
    bucket_name: str = typer.Argument("mlops-proj-group3-bucket", help="GCS bucket name for data"),
    source_folder: str = typer.Argument("torch_geometric_data", help="Source folder in GCS bucket"),
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
    
    wandb.login()
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
    data_module = load_data(root=data_path)

    c_in = data_module.num_node_features
    c_out = data_module.y.max().item() + 1
    model = NodeLevelGNN(
        c_in=c_in,
        c_hidden=hidden_channels,
        c_out=c_out,
        num_layers=hidden_layers,
        dp_rate=dropout,
    )

    model.configure_optimizers(lr=lr)

    callbacks = []
    # Callbacks for early stopping and model checkpointing
    if model_checkpoint_callback:
        model_checkpoint = pl.callbacks.ModelCheckpoint(
            save_weights_only=True, mode="max", monitor="val_acc", filename="models/current_best_model"
        )
        callbacks.append(model_checkpoint)

    # Trainer
    trainer = pl.Trainer(
        logger=wandb_logger,  # WandbLogger integration
        max_epochs=num_epochs,
        accelerator="auto",
        callbacks=callbacks,
        enable_progress_bar=True,  # Show training progress in the terminal
        log_every_n_steps=1,
    )

    node_data_loader = DataLoader(data_module, batch_size=batch_size, num_workers=7)
    # Train the model
    trainer.fit(model, node_data_loader, node_data_loader)

    # Testing the model in the train loop
    trainer.test(model, node_data_loader)
    # Save the model
    torch.save(model.state_dict(), "models/model.pt")

    # Save the model as a W&B artifact
    artifact = wandb.Artifact(
        "model",
        type="model",
        metadata=dict(
            {
                "Accuracies": trainer.callback_metrics,
                "hidden_channels": hidden_channels,
                "hidden_layers": hidden_layers,
                "dropout": dropout,
                "lr": lr,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
            }
        ),
    )
    artifact.add_file("models/model.pt")
    wandb.log_artifact(artifact, aliases=["latest_model"])

    # Clean up local data folder
    shutil.rmtree(local_data_folder, ignore_errors=True)
    print(f"Cleaned up local folder: {local_data_folder}")

    # Finish Wandb run
    wandb.finish()
    upload_model(bucket_name,"models/model.pt")


if __name__ == "__main__":
    app()

# Run in terminal: python src/wikipedia/train.py mlops-proj-group3-bucket torch_geometric_data

########################################################################################################
# Old verison

# import torch
# import pytorch_lightning as pl
# from torch_geometric.loader import DataLoader
# import typer

# from model import NodeLevelGNN
# from data import load_data, load_split_data

# # Logging
# import wandb
# from pytorch_lightning.loggers import WandbLogger

# app = typer.Typer()

# @app.command()
# def train(
#     # data_path: str = typer.Argument(..., help="Path to the data"),
#     hidden_channels: int = typer.Option(16, help="Number of hidden channels"),
#     hidden_layers: int = typer.Option(2, help="Number of hidden layers"),
#     dropout: float = typer.Option(0.5, help="Dropout rate"),
#     lr: float = typer.Option(0.01, help="Learning rate"),
#     num_epochs: int = typer.Option(100, help="Number of epochs"),
#     batch_size: int = typer.Option(32, help="Batch size"),
#     model_checkpoint_callback: bool = typer.Option(True, help="Whether to use model checkpointing"),
# ) -> None:

#     # Initialize WandbLogger
#     wandb_logger = WandbLogger(
#         project="wiki_classification",
#         config={
#             "hidden_channels": hidden_channels,
#             "hidden_layers": hidden_layers,
#             "dropout": dropout,
#             "lr": lr,
#             "num_epochs": num_epochs,
#             "batch_size": batch_size,
#         },
#     )
#     wandb_logger.experiment.log({"test_log": "Wandb is active!"})

#     data_module = load_data()

#     c_in = data_module.num_node_features
#     c_out = data_module.y.max().item() + 1
#     model = NodeLevelGNN(c_in=c_in,
#         c_hidden=hidden_channels,
#         c_out=c_out,
#         num_layers=hidden_layers,
#         dp_rate=dropout,
#         )

#     model.configure_optimizers(lr=lr)

#     callbacks = []
#     # Callbacks for early stopping and model checkpointing
#     if model_checkpoint_callback:
#         model_checkpoint = pl.callbacks.ModelCheckpoint(save_weights_only=True,
#                                                         mode="max",
#                                                         monitor="val_acc",
#                                                         filename="models/current_best_model")
#         callbacks.append(model_checkpoint)

#     # Trainer
#     trainer = pl.Trainer(
#         logger=wandb_logger,  # WandbLogger integration
#         max_epochs=num_epochs,
#         accelerator="auto",
#         callbacks=callbacks,
#         enable_progress_bar=True,  # Show training progress in the terminal
#         log_every_n_steps=1,
#     )

#     node_data_loader = DataLoader(data_module,batch_size=batch_size, num_workers=7)
#     # Train the model
#     trainer.fit(model, node_data_loader, node_data_loader)

#     # Testing the model in the train loop
#     trainer.test(model, node_data_loader)
#     # Save the model
#     torch.save(model.state_dict(), "models/model.pt")

#     # Save the model as a W&B artifact
#     artifact = wandb.Artifact('model', type='model', metadata=dict({
#             "Accuracies": trainer.callback_metrics,
#             "hidden_channels": hidden_channels,
#             "hidden_layers": hidden_layers,
#             "dropout": dropout,
#             "lr": lr,
#             "num_epochs": num_epochs,
#             "batch_size": batch_size,
#         }))
#     artifact.add_file('models/model.pt')
#     wandb.log_artifact(artifact, aliases=["latest_model"])

#     # Finish Wandb run
#     wandb.finish()


# if __name__ == "__main__":
#     app()
