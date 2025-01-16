import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import typer
import torch_geometric.transforms as T

#from data import WikiDataset
#from data import WikiDataModule
from torch_geometric.loader import DataLoader

from model import GNNModel
from model import NodeLevelGNN
from torch_geometric.datasets import WikiCS

from data import load_data

# Logging
import wandb
from pytorch_lightning.loggers import WandbLogger



app = typer.Typer()

@app.command()
def train(
    # data_path: str = typer.Argument(..., help="Path to the data"),
    hidden_channels: int = typer.Option(16, help="Number of hidden channels"),
    hidden_layers: int = typer.Option(2, help="Number of hidden layers"),
    dropout: float = typer.Option(0.5, help="Dropout rate"),
    lr: float = typer.Option(0.01, help="Learning rate"),
    num_epochs: int = typer.Option(100, help="Number of epochs"),
    batch_size: int = typer.Option(32, help="Batch size"),
    early_stopping_callback: bool = typer.Option(False, help="Whether to use early stopping"),
    model_checkpoint_callback: bool = typer.Option(False, help="Whether to use model checkpointing"),
    
) -> None:
    
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
            "early_stopping_callback": early_stopping_callback,
            "model_checkpoint_callback": model_checkpoint_callback,
            
        },
    )
    wandb_logger.experiment.log({"test_log": "Wandb is active!"})

    data_module = load_data()

    c_in = data_module.num_node_features
    c_out = data_module.y.max().item() + 1
    model = NodeLevelGNN(c_in=c_in,
        c_hidden=hidden_channels,
        c_out=c_out,
        num_layers=hidden_layers,
        dp_rate=dropout,
        )

    model.configure_optimizers(lr=lr)

    callbacks = []
    # Callbacks for early stopping and model checkpointing
    if early_stopping_callback:
        early_stopping = pl.callbacks.EarlyStopping(patience=5, monitor="val_loss")
        callbacks.append(early_stopping)
    if model_checkpoint_callback:
        model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss", filename="best_model")
        callbacks.append(model_checkpoint)

    # Trainer
    trainer = pl.Trainer(
        logger=wandb_logger,  # WandbLogger integration
        max_epochs=num_epochs,
        # gpus=1 if torch.cuda.is_available() else None,
        callbacks=callbacks,
        enable_progress_bar=True,  # Show training progress in the terminal
    )

    node_data_loader = DataLoader(data_module, batch_size=32)
    # Train the model
    trainer.fit(model, node_data_loader, node_data_loader)
 
    # Save the model
    torch.save(model.state_dict(), "models/model.pt")

    # Finish Wandb run
    wandb.finish()


if __name__ == "__main__":
    app()