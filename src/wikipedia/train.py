import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import typer
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from data import MyDataset
from model import GCN
import wandb
from pathlib import Path


app = typer.Typer()

def train(
    data_path: str = typer.Argument(..., help="Path to the data.",flag="--data_path"),
    hidden_channels: int = typer.Option(16, help="Number of hidden channels.", flag="--hidden-channels"),
    dropout: float = typer.Option(0.5, help="Dropout rate.", flag="--dropout"),
    lr: float = typer.Option(0.01, help="Learning rate.", flag="--lr"),
    num_epochs: int = typer.Option(100, help="Number of epochs.", flag="--num-epochs"),
    batch_size: int = typer.Option(32, help="Batch size.", flag="--batch-size"),
    early_stopping_callback: bool = typer.Option(True, help="Whether to use early stopping.", flag="--early-stopping"),
    model_checkpoint_callback: bool = typer.Option(True, help="Whether to use model checkpointing.", flag="--model-checkpoint"),
) -> None:
    # Load the data
    raw_data_path = Path(data_path)
    dataset = MyDataset(raw_data_path)

    # load the model
    model = GCN(hidden_channels=hidden_channels, num_features=dataset.num_features, num_classes=dataset.num_classes, dropout=dropout)
    model.configure_optimizers(lr=lr)

    callbacks = []
    # Callbacks
    if early_stopping_callback:
        early_stopping = pl.callbacks.EarlyStopping(patience=5,monitor="val_loss")
        callbacks.append(early_stopping)
    if model_checkpoint_callback:
        model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss",filename="best_model")
        callbacks.append(model_checkpoint)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=callbacks,
        logger=pl.loggers.WandbLogger(),
    )

    # Train the model
    trainer.fit(model=model, 
                datamodule=DataLoader(dataset, batch_size=batch_size, shuffle=True))
    
    # Save the model
    torch.save(model.state_dict(), "model.pt")
    # save to wandb and gcloud?
    
