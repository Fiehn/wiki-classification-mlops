# tests/test_train.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
import pytorch_lightning as pl
import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from src.wikipedia.train import app  # Import the Typer app
from src.wikipedia.train import initialize_model, train_on_split
from src.wikipedia.model import NodeLevelGNN
import warnings
warnings.filterwarnings("ignore", ".*datetime.datetime.utcnow.*", DeprecationWarning)

runner = CliRunner()


@pytest.fixture
def mock_trainer():
    trainer = MagicMock()
    trainer.callback_metrics = {"val_acc": 0.85}  # Ensure val_acc is a float
    trainer.validate.return_value = [{"val_acc": 0.85}]
    trainer.test.return_value = [{"test_acc": 0.83}]
    return trainer

@pytest.fixture
def mock_checkpoint_callback():
    callback = MagicMock()
    callback.best_model_path = "test/path/model.ckpt"
    return callback

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

def test_train_on_split(mock_trainer, mock_checkpoint_callback):
    # Create mock data with correct dimensions
    data = MagicMock()
    data.train_mask = torch.ones((100, 1))
    data.val_mask = torch.ones((100, 1))
    data.test_mask = torch.ones(100)
    data.x = torch.randn(100, 300)
    data.edge_index = torch.randint(0, 100, (2, 200))
    data.y = torch.randint(0, 20, (100,))
    data.num_node_features = 300
    data.y.max = lambda: torch.tensor(19)
    
    # Mock prepare_data_loaders and storage client
    with patch('src.wikipedia.data.prepare_data_loaders') as mock_prepare, \
         patch('google.cloud.storage.Client') as mock_storage_client:
        mock_prepare.return_value = (MagicMock(), MagicMock(), 300, 20)
        mock_storage_client.return_value = MagicMock()
        
        # Mock the trainer and checkpoint callback
        with patch('pytorch_lightning.Trainer', return_value=mock_trainer):
            with patch('pytorch_lightning.callbacks.ModelCheckpoint', return_value=mock_checkpoint_callback):
                val_acc, test_acc = train_on_split(
                    data=data,
                    split_idx=0,
                    hidden_channels=16,
                    hidden_layers=2,
                    dropout=0.5,
                    learning_rate=0.001666,
                    weight_decay=1e-4,
                    optimizer_name="Adam",
                    num_epochs=1,
                    batch_size=11701,
                    model_checkpoint_callback=True,
                    bucket_name="test-bucket",
                    group_name="test-group",
                    enable_early_stopping=True, 
                    logging=False
                )

    assert isinstance(val_acc, float)
    assert isinstance(test_acc, float)
    assert val_acc == 0.85  # Based on mock_trainer return value
    assert test_acc == 0.83  # Based on mock_trainer return value