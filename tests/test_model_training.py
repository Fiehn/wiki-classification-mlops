import torch
from src.wikipedia.model import NodeLevelGNN
from torch_geometric.data import Data

def test_model_training():
    # Create a dummy dataset
    num_features = 300
    num_classes = 20
    hidden_channels = 16
    dropout = 0.5

    model = NodeLevelGNN(c_in=num_features, c_hidden=hidden_channels, c_out=num_classes, num_layers=2, dp_rate=dropout)
    
    data_dict = {
        "x": torch.randn(100, num_features),  # 100 nodes with 300 features
        "edge_index": torch.randint(0, 100, (2, 200)),  # 200 edges
        "y": torch.randint(0, num_classes, (100,)),  # Node labels
        "train_mask": torch.randint(0, 2, (100, 1)),  # Binary mask
        "val_mask": torch.randint(0, 2, (100, 1)),  # Binary mask
    }
    data = Data(**data_dict)

    # Forward pass
    output = model.model(data.x, data.edge_index)
    assert output.shape == (100, num_classes), "Output shape mismatch"

    # Loss calculation (example)
    loss = model.loss_module(output, data.y)
    assert loss.item() > 0, "Loss should be greater than 0"

    print("Model forward pass and training test passed.")