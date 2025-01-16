import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv  # type: ignore
import pytorch_lightning as pl


class GCN(pl.LightningModule):
    def __init__(
        self, hidden_channels: int, num_features: int, num_classes: int, dropout: float
    ) -> None:
        super().__init__()

        # Define layers explicitly
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

        # Define loss function
        self.criterion = F.nll_loss

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: handles the two required inputs for GCNConv layers.
        """
        if x.ndim != 2:
            raise ValueError(
                "Expected input is not a 2D tensor, "
                f"but got a {x.ndim}D tensor."
            )

        # Apply first GCNConv layer
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        # Apply second GCNConv layer
        x = self.conv2(x, edge_index)
        x = self.dropout(x)

        # Apply log softmax
        x = self.log_softmax(x)

        return x

    def training_step(self, batch, batch_idx):
        """
        Training step for a single batch.
        """
        x, edge_index, y = batch.x, batch.edge_index, batch.y

        # Mask the training nodes and calculate loss
        preds = self(x[batch.train_mask], edge_index)
        loss = self.criterion(preds, y[batch.train_mask])

        # Calculate accuracy
        acc = (preds.argmax(dim=1) == y[batch.train_mask]).float().mean()

        # Log loss and accuracy
        self.log("train_loss", loss)
        self.log("train_acc", acc)

        # Print for debugging
        print(f"Batch {batch_idx}: Loss = {loss.item()}, Accuracy = {acc.item()}")

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for a single batch.
        """
        x, edge_index, y = batch.x, batch.edge_index, batch.y

        # Mask the validation nodes and calculate loss
        preds = self(x[batch.val_mask], edge_index)
        loss = self.criterion(preds, y[batch.val_mask])

        # Calculate accuracy
        acc = (preds.argmax(dim=1) == y[batch.val_mask]).float().mean()

        # Log loss and accuracy
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        """
        Configure the optimizer for training.
        """
        return torch.optim.Adam(self.parameters(), lr=0.01)

    def predict(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Make predictions.
        """
        return self(x, edge_index).argmax(dim=-1)


# # Simple Graph Convolutional Network (GCN) model using PyTorch Geometric. 
# class GCN(pl.LightningModule):
#     def __init__(
#         self, hidden_channels: int, num_features: int, num_classes: int, dropout: float
#     ) -> None:
#         super().__init__()
        
#         self.criterion = F.nll_loss

#         self.model = nn.Sequential(
#             GCNConv(num_features, hidden_channels),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             GCNConv(hidden_channels, num_classes),
#             nn.Dropout(dropout),
#             nn.LogSoftmax(dim=1), # For classification
#         )

#     def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
#         if x.ndim != 2:
#             raise ValueError(
#                 "Expected input is not a 2D tensor," f"instead it is a {x.ndim}D tensor."
#             )
    
#         x = self.model(x, edge_index)
#         return x
    
#     def training_step(self, batch, batch_idx):
#         x, edge_index, y = batch.x, batch.edge_index, batch.y

#         preds = self(x[batch.train_mask], edge_index[batch.train_mask])
#         loss = self.criterion(preds, y[batch.train_mask])

#         acc = (preds.argmax(dim=1) == y).sum() / len(y)
#         self.log("train_loss", loss)
#         self.log("train_acc", acc)
        
#         return loss
    
    
#     def validation_step(self, batch) -> None:
#         x, edge_index, target = batch.x, batch.edge_index, batch.y
#         preds = self(x[batch.val_mask], edge_index[batch.val_mask])
#         loss = self.criterion(preds, target)
#         acc = (target == preds.argmax(dim=-1)).float().mean()
#         self.log('val_loss', loss, on_epoch=True)
#         self.log('val_acc', acc, on_epoch=True)
    
#     def configure_optimizers(self, lr=0.01):
#         return torch.optim.Adam(self.parameters(), lr=lr)
    
#     def predict(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
#         return self(x, edge_index).argmax(dim=-1)
    
# def loade_checkpoint(filepath):
#     checkpoint = torch.load(filepath)
#     model = GCN(checkpoint["hidden_channels"],
#                 checkpoint["num_features"],
#                 checkpoint["num_classes"],
#                 checkpoint["dropout"])
#     model.load_state_dict(checkpoint['state_dict'])

#     return model

# if __name__ == "__main__":
#     model = GCN(hidden_channels=16, num_features=300, num_classes=10, dropout=0.5)
#     print(model)
#     print("Model loaded successfully.")