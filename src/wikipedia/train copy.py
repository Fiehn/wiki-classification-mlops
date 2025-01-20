import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
import typer
from model import NodeLevelGNN
from data import load_data, load_split_data, explore_splits
import logging
import wandb
from pytorch_lightning.loggers import WandbLogger

# Adjust verbosity
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR) # WARNING, ERROR, CRITICAL, DEBUG, INFO, FATAL
app = typer.Typer()

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



@app.command("train_model")
def train_model(
    #data_path: str = typer.Argument(..., help="Path to the data"),
    hidden_channels: int = typer.Option(16, help="Number of hidden channels"),
    hidden_layers: int = typer.Option(2, help="Number of hidden layers"),
    dropout: float = typer.Option(0.5, help="Dropout rate"),
    learning_rate: float = typer.Option(0.01, help="Learning rate"),
    weight_decay: float = typer.Option(1e-4, help="Weight decay"),
    num_epochs: int = typer.Option(100, help="Number of epochs"),
    batch_size: int = typer.Option(11701, help="Batch size"),
    optimizer_name: str = typer.Option("Adam", help="Optimizer name"),
    model_checkpoint_callback: bool = typer.Option(True, help="Whether to use model checkpointing"),
    ) -> None:
    """
    Main training function for both standalone runs and W&B sweeps.
    """
    
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

    data_module = load_data()

    # Run over all 20 splits and then average the results
    num_splits = data_module.train_mask.shape[1]
    val_accuracies = []
    for split in range(num_splits):
        data_module.train_mask = data_module.train_mask[:, split]
        data_module.val_mask = data_module.val_mask[:, split]

        # Load the data and initialize the model
        c_in = data_module.num_node_features
        c_out = data_module.y.max().item() + 1
        model = NodeLevelGNN(
            c_in=c_in,
            c_hidden=hidden_channels,
            c_out=c_out,
            num_layers=hidden_layers,
            dp_rate=dropout,
            )

        model.configure_optimizers(learning_rate=learning_rate, weight_decay=weight_decay, optimizer_name=optimizer_name)

        # Callbacks for early stopping and model checkpointing
        callbacks = []
        if model_checkpoint_callback:
            model_checkpoint = pl.callbacks.ModelCheckpoint(save_weights_only=True, 
                                                            mode="max", 
                                                            monitor="val_acc", 
                                                            filename="models/current_best_model")
            callbacks.append(model_checkpoint)

        # Trainer
        trainer = pl.Trainer(
            logger=wandb_logger,  # WandbLogger integration
            max_epochs=num_epochs,
            accelerator="auto",
            callbacks=callbacks + [DeviceInfoCallback()],
            enable_progress_bar=True,  # Show training progress in the terminal
            log_every_n_steps=1,
        )

        node_data_loader = DataLoader(data_module, batch_size=batch_size, num_workers=7)
    
        # Train the model
        trainer.fit(model, node_data_loader, node_data_loader)
        
        # Testing the model
        results = trainer.test(model, node_data_loader)
        val_accuracies.append(results[0]['val_acc'])

        # Save the model
        torch.save(model.state_dict(), "models/model.pt")
        
        # Save the model as a W&B artifact
        artifact = wandb.Artifact('model', type='model', metadata=dict({
                "Accuracies": trainer.callback_metrics,
                "hidden_channels": hidden_channels,
                "hidden_layers": hidden_layers,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "optimizer_name": optimizer_name,
                "model_checkpoint_callback": model_checkpoint_callback,
        }))
        artifact.add_file('models/model.pt')
        wandb.log_artifact(artifact, aliases=["latest_model"])
        wandb.finish()

    avg_acc = sum(val_accuracies) / len(val_accuracies)
    print(f"Average validation accuracy across splits: {avg_acc:.4f}")
 
if __name__ == "__main__":
    app()
    #typer.run(train_model)
    
    # If a sweep is active, use wandb.config values
    # if wandb.run is not None and wandb.config is not None:
    #     print("Running sweep agent")
    #     train_model(
    #         hidden_channels=wandb.config.hidden_channels,
    #         hidden_layers=wandb.config.hidden_layers,
    #         dropout=wandb.config.dropout,
    #         learning_rate=wandb.config.learning_rate,
    #         weight_decay=wandb.config.weight_decay,
    #         num_epochs=wandb.config.epochs,
    #         batch_size=wandb.config.batch_size,
    #         optimizer_name=wandb.config.optimizer_name,
    #         model_checkpoint_callback=True,
    #     )
    # else:
    #     # Standalone run with Typer
    #     try:
    #         app(train_model)
    #     except Exception as e:
    #         print(f"An error occurred: {e}")