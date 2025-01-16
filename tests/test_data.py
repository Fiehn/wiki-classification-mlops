from src.wikipedia.data import WikiDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def test_dataset_class():
    """Test the WikiDataset class."""
    dataset = WikiDataset()
    assert isinstance(dataset, WikiDataset), "Dataset should be an instance of the WikiDataset class"
    assert dataset.dataset.x is not None, "Dataset x attribute should not be None"
    assert dataset.dataset.y is not None, "Dataset y attribute should not be None"
    assert dataset.dataset.edge_index is not None, "Dataset edge_index attribute should not be None"
    assert dataset.dataset.train_mask is not None, "Dataset train_mask attribute should not be None"
    assert dataset.dataset.val_mask is not None, "Dataset val_mask attribute should not be None"
    assert dataset.dataset.test_mask is not None, "Dataset test_mask attribute should not be None"
    assert isinstance(dataset.dataset, Data), "Dataset should be an instance of the PyG Data class"

def test_data_loaders():
    dataset = WikiDataset()
    train_loader = dataset.data_loader()

    assert len(dataset.dataset.train_mask) == 11701, "Length of the train mask should be 11701"

    assert isinstance(train_loader, DataLoader), "Train loader should be an instance of the DataLoader class"
    #assert isinstance(val_loader, DataLoader), "Val loader should be an instance of the DataLoader class"
    assert len(train_loader) == 1, "Length of the train loader should be 1"
    #assert len(val_loader) == 1, "Length of the val loader should be 1"


