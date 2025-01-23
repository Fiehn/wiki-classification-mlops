import torch
from src.wikipedia.model import NodeLevelGNN
from torch_geometric.data import Data
import pytest
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from unittest.mock import MagicMock

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

@pytest.fixture
def gnn_model():
    return NodeLevelGNN(c_in=300, c_hidden=16, c_out=20, num_layers=2, dp_rate=0.5)

@pytest.fixture
def mock_data():
    x = torch.randn(50, 300)
    edge_index = torch.randint(0, 50, (2, 100))
    y = torch.randint(0, 20, (50,))
    train_mask = torch.randint(0, 2, (50,), dtype=torch.bool)
    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)

@pytest.fixture
def mock_trainer():
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        logger=CSVLogger(save_dir='test_logs'),
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False
    )
    return trainer

def test_forward_pass_fixture(gnn_model, mock_data):
    output = gnn_model.model(mock_data.x, mock_data.edge_index)
    assert output.shape == (50, 20)

def test_training_step_fixture(gnn_model, mock_data, mock_trainer):
    gnn_model.trainer = mock_trainer
    mock_trainer.strategy._lightning_module = gnn_model
    gnn_model.log = MagicMock()
    loss = gnn_model.training_step(mock_data, batch_idx=0)
    assert loss.numel() == 1

def test_no_edges(gnn_model):
    x = torch.randn(10, 300)
    edge_index = torch.empty(2, 0, dtype=torch.long)  # no edges
    y = torch.randint(0, 20, (10,))
    data = Data(x=x, edge_index=edge_index, y=y)
    output = gnn_model.model(data.x, data.edge_index)
    assert output.shape == (10, 20)

def test_mismatch_input(gnn_model):
    x = torch.randn(10, 200)  # mismatch feature size
    edge_index = torch.randint(0, 10, (2, 10))
    y = torch.randint(0, 20, (10,))
    data = Data(x=x, edge_index=edge_index, y=y)
    try:
        gnn_model.model(data.x, data.edge_index)
        assert False, "Expected an error with mismatched input sizes"
    except Exception:
        pass
