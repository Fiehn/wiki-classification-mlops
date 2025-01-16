import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import typer
import torch_geometric.transforms as T

from data import WikiDataset
from torch_geometric.loader import DataLoader

from model import GNNModel
from model import NodeLevelGNN
from torch_geometric.datasets import WikiCS

# Logging
import wandb
from pytorch_lightning.loggers import WandbLogger

app = typer.Typer()

@app.command()
def train(
    # data_path: str = typer.Argument(..., help="Path to the data"),
    hidden_channels: int = typer.Option(16, help="Number of hidden channels"),
    dropout: float = typer.Option(0.5, help="Dropout rate"),
    lr: float = typer.Option(0.01, help="Learning rate"),
    num_epochs: int = typer.Option(100, help="Number of epochs"),
    batch_size: int = typer.Option(32, help="Batch size"),
    early_stopping_callback: bool = typer.Option(True, help="Whether to use early stopping"),
    model_checkpoint_callback: bool = typer.Option(True, help="Whether to use model checkpointing"),
) -> None:
    
    # Initialize WandbLogger
    wandb_logger = WandbLogger(
        project="wiki_classification",
        config={
            "hidden_channels": hidden_channels,
            "dropout": dropout,
            "lr": lr,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "early_stopping_callback": early_stopping_callback,
            "model_checkpoint_callback": model_checkpoint_callback,
        },
    )
    wandb_logger.experiment.log({"test_log": "Wandb is active!"})

    # Load the data
    dataset = WikiDataset()

    print(f"Dataset loaded: {dataset}")
    print(f"Number of features: {dataset.num_features}, Number of classes: {dataset.num_classes}")

    # Log dataset details to Wandb
    wandb_logger.experiment.config.update(
        {
            "num_features": dataset.num_features,
            "num_classes": dataset.num_classes,
        }
    )

    # load the model
    model = GCN(
        hidden_channels=hidden_channels,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        dropout=dropout,
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

    # Train the model
    trainer.fit(model=model, train_dataloaders=DataLoader(dataset.dataset), val_dataloaders=DataLoader(dataset.dataset))
    # datamodule=DataLoader(dataset, batch_size=batch_size, shuffle=True))

    # Save the model
    torch.save(model.state_dict(), "model.pt")

    # Finish Wandb run
    wandb.finish()


def train_node_classifier(model_name, dataset, **model_kwargs):
    #pl.seed_everything(42)
    node_data_loader = DataLoader(dataset, batch_size=20)

    # Create a PyTorch Lightning trainer
    root_dir = "logs/"
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        accelerator="auto",
        max_epochs=20,
        enable_progress_bar=True,
    )  # 0 because epoch size is 1

    # Check whether pretrained model exists. If yes, load it and skip training
    model = NodeLevelGNN(model_name, **model_kwargs)
    trainer.fit(model, node_data_loader, node_data_loader)
    
    # Test best model on the test set
    test_result = trainer.test(model, dataloaders=node_data_loader, verbose=False)
    batch = next(iter(node_data_loader))
    batch = batch.to(model.device)
    _, train_acc = model.forward(batch, mode="train")
    _, val_acc = model.forward(batch, mode="val")
    result = {"train": train_acc, "val": val_acc, "test": test_result[0]["test_acc"]}
    return model, result


# Small function for printing the test scores
def print_results(result_dict):
    if "train" in result_dict:
        print("Train accuracy: %4.2f%%" % (100.0 * result_dict["train"]))
    if "val" in result_dict:
        print("Val accuracy:   %4.2f%%" % (100.0 * result_dict["val"]))
    print("Test accuracy:  %4.2f%%" % (100.0 * result_dict["test"]))


if __name__ == "__main__":
    data = WikiCS(root="data/")

    # if data.train_mask.dim() == 2:
    #     data.train_mask = data.train_mask[:, 0]
    # if data.val_mask.dim() == 2:
    #     data.val_mask = data.val_mask[:, 0]
    # if data.test_mask.dim() == 2:
    #     data.test_mask = data.test_mask[:, 0]

    node_mlp_model, node_mlp_result = train_node_classifier(
        model_name="MLP",
        dataset=data,
        c_in=data.num_node_features,
        c_hidden=32,
        c_out=data.y.max().item() + 1,
        num_layers=2,
        dp_rate=0.1,
    )

    print_results(node_mlp_result)
