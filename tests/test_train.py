# tests/test_train.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pytorch_lightning as pl
from torch_geometric.data import Data


import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from src.wikipedia.train import app  # Import the Typer app
from src.wikipedia.train import initialize_model, train_on_split
from src.wikipedia.model import NodeLevelGNN

runner = CliRunner()

# Fixtures to mock external dependencies
# @pytest.fixture
# def mock_download_from_gcs():
#     with patch('src.wikipedia.train.download_from_gcs') as mock_download:
#         mock_download.return_value = 'path/to/data'
#         yield mock_download

@pytest.fixture
def mock_load_split_data():
    with patch('src.wikipedia.train.load_split_data') as mock_load_split:
        mock_data = MagicMock()
        mock_data_module = [mock_data]
        mock_data.train_mask.shape = (100, 20)  # Example shape
        mock_load_split.return_value = mock_data_module
        yield mock_load_split

@pytest.fixture
def mock_trainer():
    trainer = MagicMock()
    trainer.callback_metrics = {"val_acc": 0.85}
    return trainer

@pytest.fixture
def mock_checkpoint_callback():
    callback = MagicMock()
    callback.best_model_path = "test/path/model.ckpt"
    return callback

@pytest.fixture
def mock_google_auth(monkeypatch):
    # Mock secretmanager client
    monkeypatch.setattr("src.wikipedia.train.secretmanager.SecretManagerServiceClient", MagicMock())
    # Mock storage client
    monkeypatch.setattr("src.wikipedia.train.storage.Client", MagicMock())
    # If needed, also mock environment variables for credentials
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/dev/null")

@pytest.fixture
def mock_train_on_split():
    with patch('src.wikipedia.train.train_on_split') as mock_train:
        mock_train.return_value = (0.8, 0.75)  # Example accuracies
        yield mock_train

@pytest.fixture
def mock_upload_model():
    with patch('src.wikipedia.train.upload_model') as mock_upload:
        yield mock_upload

@pytest.fixture
def mock_wandb_login():
    with patch('src.wikipedia.train.wandb.login') as mock_login:
        yield mock_login

@pytest.fixture
def mock_wandb_logger():
    with patch('src.wikipedia.train.WandbLogger') as mock_logger:
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance
        yield mock_logger_instance

@pytest.fixture
def mock_gnn_model():
    with patch('src.wikipedia.model.NodeLevelGNN') as mock_model:
        mock_model_instance = MagicMock()
        mock_model_instance.parameters.return_value = [MagicMock()]  # Ensure non-empty parameters
        mock_model.return_value = mock_model_instance
        yield mock_model

@pytest.mark.parametrize(
    "c_in, c_out, hidden_channels, hidden_layers, dropout, learning_rate, weight_decay, optimizer_name, expected_exception",
    [ 
        (300, 20, -16, 2, 0.5, 0.001666, 1e-4, "Adam", ValueError),
        (300, 20, 16, 0, 0.5, 0.001666, 1e-4, "Adam", ValueError),
        (300, 20, 16, 2, 0.5, 0.001666, 1e-4, "SGD", NotImplementedError),
        (300, 20, 16, 2, 0.5, 0.001666, 1e-4, "Unknown", ValueError),
    ]
)
def test_initialize_model_parametrized(
    c_in, c_out, hidden_channels, hidden_layers, dropout, learning_rate, weight_decay, optimizer_name, expected_exception
):
    if expected_exception:
        with pytest.raises(expected_exception):
            from src.wikipedia.train import initialize_model
            initialize_model(
                c_in=c_in,
                c_out=c_out,
                hidden_channels=hidden_channels,
                hidden_layers=hidden_layers,
                dropout=dropout,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                optimizer_name=optimizer_name
            )
    else:
        from src.wikipedia.train import initialize_model
        from src.wikipedia.model import NodeLevelGNN
        model = initialize_model(
            c_in=c_in,
            c_out=c_out,
            hidden_channels=hidden_channels,
            hidden_layers=hidden_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer_name=optimizer_name
        )
        assert isinstance(model, NodeLevelGNN)


# def test_train_on_split(mock_trainer, mock_checkpoint_callback):
#     # Create mock data with correct dimensions
#     data = MagicMock()
#     data.train_mask = torch.ones((100, 1))
#     data.val_mask = torch.ones((100, 1))
#     data.test_mask = torch.ones(100)
#     data.x = torch.randn(100, 300)
#     data.edge_index = torch.randint(0, 100, (2, 200))
#     data.y = torch.randint(0, 20, (100,))
#     data.num_node_features = 300
#     data.y.max = lambda: torch.tensor(19)
    
#     # Mock prepare_data_loaders and storage client
#     with patch('src.wikipedia.train.prepare_data_loaders') as mock_prepare, \
#          patch('google.cloud.storage.Client') as mock_storage_client:
#         mock_prepare.return_value = (MagicMock(), MagicMock(), 300, 20)
#         mock_storage_client.return_value = MagicMock()
        
#         # Mock the trainer and checkpoint callback
#         with patch('pytorch_lightning.Trainer', return_value=mock_trainer):
#             with patch('pytorch_lightning.callbacks.ModelCheckpoint', return_value=mock_checkpoint_callback):
#                 val_acc, test_acc =train_on_split(
#                     data=data,
#                     split_idx=0,
#                     hidden_channels=16,
#                     hidden_layers=2,
#                     dropout=0.5,
#                     learning_rate=0.001666,
#                     weight_decay=1e-4,
#                     optimizer_name="Adam",
#                     num_epochs=1,
#                     batch_size=11701,
#                     wandb_logger=None,
#                     model_checkpoint_callback=True,
#                     bucket_name="test-bucket"
#                 )
    
#     assert isinstance(val_acc, float)
#     assert isinstance(test_acc, float)

# Add more specific test for optimizer error
# def test_sgd_optimizer_raises_error():
#     with pytest.raises(NotImplementedError, match="SGD not optimal for GNNs"):
#         model = NodeLevelGNN(
#             c_in=300, 
#             c_out=20, 
#             c_hidden=16,  # This matches c_hidden parameter
#             num_layers=2, 
#             dp_rate=0.5
#         )
#         # Mock the underlying GNNModel
#         model.model = MagicMock()
#         model.configure_optimizers(optimizer_name="SGD")

# Add test for model initialization
# def test_node_level_gnn_init():
#     model = NodeLevelGNN(
#         c_in=300,
#         c_out=20,
#         c_hidden=16,
#         num_layers=2,
#         dp_rate=0.5
#     )
#     assert model.c_out == 20
#     assert isinstance(model.loss_module, torch.nn.CrossEntropyLoss)
#     assert isinstance(model.model, torch.nn.Module)


