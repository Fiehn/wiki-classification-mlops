import pytest
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.data import Data
from src.wikipedia.data import (
    load_split_data,
    explore_splits,
    download_from_gcs,
    upload_model,
    download_file,
    prepare_data_loaders,
    prepare_test_loader
)

@pytest.fixture
def dataset():
    """Fixture to load the dataset."""
    return load_split_data()

@pytest.fixture
def split_idx():
    """Fixture to provide a specific split index."""
    return 0  # Replace with a valid split index as needed

def test_dataset_not_empty(dataset):
    """Test that the dataset is not empty."""
    assert dataset is not None
    assert len(dataset) > 0

def test_dataset_properties(dataset):
    """Test the properties of the dataset."""
    assert hasattr(dataset, "num_node_features")
    assert dataset.num_node_features > 0
    assert hasattr(dataset, "num_classes")
    assert dataset.num_classes > 0

def test_data_splits(dataset):
    """Test the presence of data split masks."""
    assert hasattr(dataset, "train_mask")
    assert hasattr(dataset, "val_mask")
    assert hasattr(dataset, "test_mask")

@pytest.mark.parametrize("split", ["train_mask", "val_mask", "test_mask"])
def test_split_masks(dataset, split):
    """Test each data split mask for correct type and non-emptiness."""
    mask = getattr(dataset, split)
    assert isinstance(mask, torch.BoolTensor)
    assert mask.sum().item() > 0

def test_prepare_data_loaders(dataset, split_idx):
    """Test the data loaders preparation."""
    train_loader, val_loader, c_in, c_out = prepare_data_loaders(dataset._data, split_idx)
    assert train_loader is not None
    assert val_loader is not None
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(val_loader, torch.utils.data.DataLoader)
    assert isinstance(c_in, int)
    assert isinstance(c_out, int)

def test_prepare_test_loader(dataset):
    """Test the test loader preparation."""
    test_loader = prepare_test_loader(dataset._data)
    assert test_loader is not None
    assert isinstance(test_loader, torch.utils.data.DataLoader)
