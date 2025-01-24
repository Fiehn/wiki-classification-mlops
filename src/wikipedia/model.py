from torch import nn
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl
from torch_geometric import nn as geom_nn
import torch.optim as optim
import torchmetrics

class GNNModel(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        c_out,
        num_layers=2,
        dp_rate=0.1,
        **kwargs,
    ):
        """GNNModel.

        Args:
            c_in: Dimension of input features
            c_hidden: Dimension of hidden features
            c_out: Dimension of the output features. Usually number of classes in classification
            num_layers: Number of "hidden" graph layers
            layer_name: String of the graph layer to use
            dp_rate: Dropout rate to apply throughout the network
            kwargs: Additional arguments for the graph layer (e.g. number of heads for GAT)

        """
        super().__init__()
        gnn_layer = GCNConv


        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels=in_channels, out_channels=out_channels, **kwargs),
                nn.ReLU(inplace=True),
                #nn.Linear(out_channels, c_hidden),
                #nn.ReLU(inplace=True),
                nn.Dropout(dp_rate),
                #nn.BatchNorm1d(out_channels),
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels, out_channels=c_out, **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        """Forward.

        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)

        """
        for layer in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(layer, geom_nn.MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x

# Simple Graph Convolutional Network (GCN) model using PyTorch Geometric. 
class NodeLevelGNN(pl.LightningModule):
    def __init__(self, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()

        self.model = GNNModel(**model_kwargs)
        self.loss_module = nn.CrossEntropyLoss()

        # Initialize metrics using torchmetrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=model_kwargs['c_out'])
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=model_kwargs['c_out'])
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=model_kwargs['c_out'])
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()

    def forward(self, data, mode="train"):
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index)

        # Get appropriate mask and ensure it's 1D
        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            raise ValueError(f"Unknown forward mode: {mode}")

        # Convert 2D mask to 1D if needed
        if mask.dim() == 2:
            mask = mask[:, 0]  # Take first split

        # Shape checks for debugging
        #assert mask.dim() == 1, f"Mask should be 1D, got shape {mask.shape}"
        # assert mask.shape[0] == x.shape[0], f"Mask length {mask.shape[0]} doesn't match number of nodes {x.shape[0]}"
        
        loss = self.loss_module(x[mask], data.y[mask])
        acc = (x[mask].argmax(dim=-1) == data.y[mask]).sum().float() / mask.sum()
        return loss, acc

    def configure_optimizers(self, learning_rate=0.01, weight_decay=1e-4, optimizer_name="Adam"):
        if optimizer_name == "Adam":
            optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "NAdam":
            optimizer = optim.NAdam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "RMSprop":
            optimizer = optim.RMSprop(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "SGD":
            raise NotImplementedError("SGD not optimal for GNNs")
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="val")
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="test")
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
    def predict(self, batch):
        x, edge_index = batch.x, batch.edge_index
        return self.model(x, edge_index)
    
if __name__ == "__main__":
    #model = GCN(hidden_channels=16, num_features=300, num_classes=10, dropout=0.5)
    #print(model)
    #print("Model loaded successfully.")
    pass