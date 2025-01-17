from src.wikipedia.data import load_data, load_split_data


def test_dataset():
    """Test the WikiDataset class."""
    data = load_data()
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
