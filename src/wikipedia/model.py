# Cited from https://github.com/eyhl/group5-pyg-dtu-mlops/tree/main/src/models

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv  # type: ignore
import pytorch_lightning as pl


# Simple Graph Convolutional Network (GCN) model using PyTorch Geometric. 
class GCN(pl.LightningModule):
    def __init__(
        self, hidden_channels: int, num_features: int, num_classes: int, dropout: float
    ) -> None:
        super().__init__()
        
        self.model = nn.Sequential(
            GCNConv(num_features, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            GCNConv(hidden_channels, num_classes),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                "Expected input is not a 2D tensor," f"instead it is a {x.ndim}D tensor."
            )
        if x.shape[1] != 1433:
            raise ValueError("Feature vector should be of size 1433.")
        
        x = self.model(x, edge_index)
        return x
    
    def training_step(self, batch, batch_idx, loss_fn=F.nll_loss):
        x, edge_index, y = batch.x, batch.edge_index, batch.y

        preds = self(x, edge_index)
        loss = loss_fn(preds, y)

        acc = (preds.argmax(dim=1) == y).sum() / len(y)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        
        return loss
    
    def validation_step(self, batch) -> None:
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)
    
    def configure_optimizers(self, lr=0.01):
        return torch.optim.Adam(self.parameters(), lr=lr)
    
    def predict(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self(x, edge_index).argmax(dim=-1)
        

# def load_checkpoint(filepath):
#     checkpoint = torch.load(filepath)
#     model = GCN(checkpoint["hidden_channels"],
#                 checkpoint["num_features"],
#                 checkpoint["num_classes"],
#                 checkpoint["dropout"])
#     model.load_state_dict(checkpoint['state_dict'])

#     return model

if __name__ == "__main__":
    model = GCN(hidden_channels=16, num_features=1433, num_classes=7, dropout=0.5)
    print(model)
    print("Model loaded successfully.")