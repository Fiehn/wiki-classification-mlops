import os
import shutil
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import WikiCS

# Set the path to your service account key

def load_split_data(root="data/"):
    dataset = WikiCS(root=root, is_undirected=True)
    return dataset

def prepare_data_loaders(data, split_idx):
    """
    Prepare train and validation data loaders for a specific split.
    
    Args:
        data: The original data object
        split_idx: Index of the current split
    
    Returns:
        tuple: (train_loader, val_loader, c_in, c_out)
    """
    train_data = data.clone()
    val_data = data.clone()
    train_data.train_mask = data.train_mask[:, split_idx]  # 1D mask for training
    train_data.val_mask = None  # Not needed during training
    val_data.val_mask = data.val_mask[:, split_idx]  # 1D mask for validation
    val_data.train_mask = None  # Not needed during validation

    train_loader = DataLoader([train_data], batch_size=1, num_workers=4, shuffle=False)
    val_loader = DataLoader([val_data], batch_size=1, num_workers=4)

    # Get model dimensions
    c_in = data.num_node_features
    c_out = data.y.max().item() + 1
    
    return train_loader, val_loader, c_in, c_out

def prepare_test_loader(data):
    """Prepare test data loader for model evaluation."""
    test_data = data.clone()  # test_data.test_mask remains as is
    # Create a DataLoader for the test data (wrap in a list)
    test_loader = DataLoader([test_data], batch_size=1, num_workers=4)
    return test_loader

def explore_splits(dataset=None):
    if dataset is None:
        dataset = WikiCS(root="data/", is_undirected=True)
        dataset = dataset[0]
    num_splits = dataset.train_mask.shape[1]
    print(f"There are {num_splits} training/validation splits.\n")
    for i in range(num_splits):
        train_count = dataset.train_mask[:, i].sum().item()
        val_count   = dataset.val_mask[:, i].sum().item()
        stop_count  = dataset.stopping_mask[:, i].sum().item()
        print(f"Split {i}:")
        print(f"  Training nodes: {train_count}")
        print(f"  Validation nodes: {val_count}")
        print(f"  Stopping nodes: {stop_count}\n")
    # Test mask is a single vector:
    test_count = dataset.test_mask.sum().item()
    print("Test set nodes:", test_count)

    print(dataset.x.shape)
    print(dataset.edge_index.shape)
    print(dataset.y.shape)
    print(dataset.train_mask.shape)
    print(dataset.val_mask.shape)
    print(dataset.test_mask.shape)
    # We have one global test set of 5847
    print(dataset.test_mask.shape)
  

def cleanup_local_data(folder):
    """Delete the local data folder after upload."""
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"Deleted local folder: {folder}")
    else:
        print(f"Folder not found: {folder}")


if __name__ == "__main__":
    load_split_data()