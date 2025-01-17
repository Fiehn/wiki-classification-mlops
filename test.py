
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Load the tuple
loaded_tuple = torch.load("data/processed/data_undirected.pt",weights_only=True)


# Assume it's something like (data_object, None)
if isinstance(loaded_tuple, tuple):
    data = loaded_tuple[0]  # The PyG Data-like object
else:
    data = loaded_tuple

# Make sure it's a PyG Data object
if not isinstance(data, Data):
    data = Data(**data._asdict()) if hasattr(data, "_asdict") else Data(**dict(data))

# Now you can create a DataLoader
loader = DataLoader([data], batch_size=2, shuffle=False)
# DataBatch(x=[11701, 300], edge_index=[2, 431726]
#  y=[11701], train_mask=[11701, 20], val_mask=[11701, 20],
#  test_mask=[11701], stopping_mask=[11701, 20], batch=[11701], ptr=[2])


import torch
from torch_geometric.data import Data
from src.wikipedia.data import WikiDataset

# Assume 'data' is the loaded WikiCS dataset
data = WikiDataset()
data = data.dataset

# Function to create a subgraph based on a mask
def create_subgraph(data, mask):
    # Extract nodes that are True in the mask
    node_indices = mask.nonzero(as_tuple=True)[0]

    # Map original indices to subgraph indices
    subgraph_x = data.x[node_indices]
    subgraph_y = data.y[node_indices]
    subgraph_train_mask = data.train_mask[node_indices]
    subgraph_val_mask = data.val_mask[node_indices]
    subgraph_test_mask = data.test_mask[node_indices]

    # Filter edges where both nodes are in the subgraph
    edge_mask = torch.isin(data.edge_index[0], node_indices) & torch.isin(data.edge_index[1], node_indices)
    subgraph_edge_index = data.edge_index[:, edge_mask]

    # Remap edge indices to be zero-indexed
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(node_indices.tolist())}
    subgraph_edge_index = subgraph_edge_index.apply_(lambda idx: index_map[idx])

    # Create the new subgraph Data object
    subgraph = Data(
        x=subgraph_x,
        edge_index=subgraph_edge_index,
        y=subgraph_y,
        train_mask=subgraph_train_mask,
        val_mask=subgraph_val_mask,
        test_mask=subgraph_test_mask
    )
    return subgraph


# Create subgraphs for train, validation, and test
train_subgraph = create_subgraph(data, data.train_mask)
val_subgraph = create_subgraph(data, data.val_mask)
test_subgraph = create_subgraph(data, data.test_mask)

# Check the subgraphs
print(train_subgraph)
print(val_subgraph)
print(test_subgraph)
