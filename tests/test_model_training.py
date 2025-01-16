import torch
from src.wikipedia.model import GCN


def test_model_training():
    # Create a dummy dataset
    num_features = 300
    num_classes = 20
    hidden_channels = 16
    dropout = 0.5

    model = GCN(hidden_channels, num_features, num_classes, dropout)
    x = torch.randn(100, num_features)  # 100 nodes with 300 features
    edge_index = torch.randint(0, 100, (2, 200))  # 200 edges
    y = torch.randint(0, num_classes, (100,))  # Node labels

    # Forward pass
    output = model(x, edge_index)
    assert output.shape == (100, num_classes), "Output shape mismatch"

    # Loss calculation (example)
    loss = model.criterion(output, y)
    assert loss.item() > 0, "Loss should be greater than 0"

    print("Model forward pass and training test passed.")