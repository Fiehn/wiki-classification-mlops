import pytest
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.data import Data

from src.wikipedia.model import NodeLevelGNN

def test_model():
    models = NodeLevelGNN(c_in=300, c_hidden=16, c_out=20, num_layers=2, dp_rate=0.5)
    assert models is not None
    assert models.loss_module is not None
    assert models.model is not None
    assert models.forward is not None
    assert models.training_step is not None
    assert models.validation_step is not None
    assert models.configure_optimizers is not None

class MockGraphData:
    def __init__(self, num_nodes=100, num_features=300, num_classes=20):
        x = torch.randn(num_nodes, num_features)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        y = torch.randint(0, num_classes, (num_nodes,))
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[:int(num_nodes * 0.6)] = True
        val_mask[int(num_nodes * 0.6):int(num_nodes * 0.8)] = True
        test_mask[int(num_nodes * 0.8):] = True
        
        self.data = Data(
            x=x, 
            edge_index=edge_index, 
            y=y, 
            train_mask=train_mask, 
            val_mask=val_mask, 
            test_mask=test_mask
        )

@pytest.fixture
def gnn_model():
    return NodeLevelGNN(
        c_in=300, 
        c_hidden=16, 
        c_out=20, 
        num_layers=2, 
        dp_rate=0.5
    )

@pytest.fixture
def mock_trainer():
    return pl.Trainer(
        max_epochs=1, 
        limit_train_batches=1,
        logger=CSVLogger(save_dir='test_logs'),
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False
    )

def test_model_initialization(gnn_model):
    """Test model initialization and basic properties."""
    assert gnn_model is not None
    assert hasattr(gnn_model, 'model')
    assert hasattr(gnn_model, 'loss_module')
    assert isinstance(gnn_model.loss_module, torch.nn.CrossEntropyLoss)

def test_invalid_forward_mode(gnn_model):
    """Test handling of invalid forward mode."""
    mock_data = MockGraphData().data
    with pytest.raises(AssertionError):
        gnn_model.forward(mock_data, mode="invalid")

def test_empty_graph_data(gnn_model):
    """Test handling of empty graph data."""
    empty_data = Data(x=torch.empty(0, 300), edge_index=torch.empty(2, 0), y=torch.empty(0))
    with pytest.raises(Exception):
        gnn_model.forward(empty_data)

def test_prediction_properties(gnn_model):
    """Validate prediction properties."""
    predictions = gnn_model.predict(MockGraphData().data)
    
    assert predictions.shape[0] == 100
    assert predictions.shape[1] == 20

def test_forward_pass(gnn_model):
    """Test forward pass with mock graph data."""
    mock_data = MockGraphData().data
    
    loss, acc = gnn_model.forward(mock_data, mode="train")
    
    assert isinstance(loss, torch.Tensor)
    assert isinstance(acc, torch.Tensor)
    assert loss.numel() == 1
    assert 0 <= acc <= 1

def test_optimizers(gnn_model):
    """Test different optimizer configurations."""
    optimizer_names = ["Adam", "AdamW", "NAdam", "RMSprop"]
    
    for opt_name in optimizer_names:
        optimizer = gnn_model.configure_optimizers(
            learning_rate=0.01, 
            weight_decay=1e-4, 
            optimizer_name=opt_name
        )
        assert optimizer is not None

def test_invalid_optimizer(gnn_model):
    """Test handling of an invalid optimizer."""
    with pytest.raises(AssertionError):
        gnn_model.configure_optimizers(optimizer_name="InvalidOptimizer")

def test_training_step(gnn_model, mock_trainer):
    """Test training step with mock data and trainer."""
    from unittest.mock import MagicMock
    mock_data = MockGraphData().data
    gnn_model.trainer = mock_trainer
    mock_trainer.strategy._lightning_module = gnn_model
    gnn_model.log = MagicMock()
    
    loss = gnn_model.training_step(mock_data, batch_idx=0)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.numel() == 1

def test_validation_step(gnn_model, mock_trainer):
    """Test validation step with mock trainer."""
    from unittest.mock import MagicMock
    mock_data = MockGraphData().data
    gnn_model.trainer = mock_trainer
    mock_trainer.strategy._lightning_module = gnn_model
    gnn_model.log = MagicMock()
    
    gnn_model.validation_step(mock_data, batch_idx=0)
    

def test_predict(gnn_model):
    """Test prediction method."""
    mock_data = MockGraphData().data
    
    predictions = gnn_model.predict(mock_data)
    
    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape[0] == mock_data.x.shape[0]
    assert predictions.shape[1] == 20  # num_classes

def test_metric_tracking(gnn_model, mock_trainer):
    """Verify metric tracking during training."""
    from unittest.mock import MagicMock
    mock_data = MockGraphData().data
    gnn_model.trainer = mock_trainer
    mock_trainer.strategy._lightning_module = gnn_model
    gnn_model.log = MagicMock()
    
    gnn_model.training_step(mock_data, batch_idx=0)
    gnn_model.validation_step(mock_data, batch_idx=0)
    
    # You might need to mock or capture logging to fully test this
    assert hasattr(gnn_model, 'train_acc')
    assert hasattr(gnn_model, 'train_loss')
    assert hasattr(gnn_model, 'val_acc')
    assert hasattr(gnn_model, 'val_loss')
    assert hasattr(gnn_model, 'test_acc')
    assert hasattr(gnn_model, 'test_loss')
