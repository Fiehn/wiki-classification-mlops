
import pytest
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.data import Data
from src.wikipedia.data import load_split_data, explore_splits, download_from_gcs, upload_model, download_file


def test_dataset():
    """Test the WikiDataset class."""
    data = load_split_data()
    assert data is not None
    assert len(data) > 0
    assert hasattr(data, "num_node_features")
    assert hasattr(data, "num_classes")
    assert hasattr(data, "train_mask")
    assert hasattr(data, "val_mask")
    assert data.num_node_features > 0
    assert data.num_classes > 0
    assert data.train_mask.dim() == 1
    assert data.val_mask.dim() == 1

    data = load_split_data()
    assert data is not None
    assert data.train_mask.dim() > 1
    assert data.val_mask.dim() > 1
