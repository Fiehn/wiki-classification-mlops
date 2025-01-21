from torch_geometric.datasets import WikiCS
from torch_geometric.transforms import NormalizeFeatures

# Downloaded from: https://github.com/pmernyei/wiki-cs-dataset
# Using: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/wikics.html#WikiCS

def load_data():
    dataset = WikiCS(root="data/", is_undirected=True)
    # collapse the masks into a single mask
    split_index = 0
    dataset.train_mask = dataset.train_mask[split_index]
    dataset.val_mask = dataset.val_mask[split_index]

    # dataset.train_mask = dataset.train_mask.sum(dim=1).bool()
    # dataset.val_mask = dataset.val_mask.sum(dim=1).bool()
    return dataset

def load_split_data():
    dataset = WikiCS(root="data/", is_undirected=True)
    return dataset

def explore_splits(dataset):
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



def explore_splits2():
    dataset = WikiCS(root="data/", is_undirected=True)
    data = dataset[0]
    num_splits = data.train_mask.shape[1]
    print(f"There are {num_splits} training/validation splits.\n")
    for i in range(num_splits):
        train_count = data.train_mask[:, i].sum().item()
        val_count   = data.val_mask[:, i].sum().item()
        stop_count  = data.stopping_mask[:, i].sum().item()
        print(f"Split {i}:")
        print(f"  Training nodes: {train_count}")
        print(f"  Validation nodes: {val_count}")
        print(f"  Stopping nodes: {stop_count}\n")
    # Test mask is a single vector:
    test_count = data.test_mask.sum().item()
    print("Test set nodes:", test_count)

