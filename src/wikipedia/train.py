import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import typer
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

from data import WikiDataset
from model import GCN
import logger
from pathlib import Path


app = typer.Typer()

@app.command()
def train(
    #data_path: str = typer.Argument(..., help="Path to the data"),
    hidden_channels: int = typer.Option(16, help="Number of hidden channels"),
    dropout: float = typer.Option(0.5, help="Dropout rate"),
    lr: float = typer.Option(0.01, help="Learning rate"),
    num_epochs: int = typer.Option(100, help="Number of epochs"),
    batch_size: int = typer.Option(32, help="Batch size"),
    early_stopping_callback: bool = typer.Option(True, help="Whether to use early stopping"),
    model_checkpoint_callback: bool = typer.Option(True, help="Whether to use model checkpointing"),
) -> None:
    # Load the data

    dataset = WikiDataset()

    print(f"Dataset loaded: {dataset}")
    print(f"Number of features: {dataset.num_features}, Number of classes: {dataset.num_classes}")

    # load the model
    model = GCN(hidden_channels=hidden_channels, num_features=dataset.num_features, num_classes=dataset.num_classes, dropout=dropout)
    model.configure_optimizers(lr=lr)

    callbacks = []
    # Callbacks for early stopping and model checkpointing
    if early_stopping_callback:
        early_stopping = pl.callbacks.EarlyStopping(patience=5,monitor="val_loss")
        callbacks.append(early_stopping)
    if model_checkpoint_callback:
        model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss",filename="best_model")
        callbacks.append(model_checkpoint)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        # gpus=1 if torch.cuda.is_available() else None,
        callbacks=callbacks,
        logger=True,  # Default logger prints to console
        enable_progress_bar=True  # Show training progress in the terminal
    )


    # Train the model
    trainer.fit(model=model, 
                train_dataloaders=DataLoader(dataset, batch_size=batch_size, shuffle=True),
                val_dataloaders=DataLoader(dataset, batch_size=batch_size, shuffle=False))
                #datamodule=DataLoader(dataset, batch_size=batch_size, shuffle=True))
    
    # Save the model 
    torch.save(model.state_dict(), "model.pt")
    # save to wandb and gcloud?

if __name__ == "__main__":
    app()
