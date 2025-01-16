from torch_geometric.datasets import WikiCS
from torch_geometric.transforms import NormalizeFeatures

# Downloaded from: https://github.com/pmernyei/wiki-cs-dataset
# Using: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/wikics.html#WikiCS

def load_data():
    dataset = WikiCS(root="data/", is_undirected=True)
    # collapse the masks into a single mask
    dataset.data.train_mask = dataset.data.train_mask.sum(dim=1).bool()
    dataset.data.val_mask = dataset.data.val_mask.sum(dim=1).bool()
    return dataset

def load_split_data():
    dataset = WikiCS(root="data/")
    return dataset
