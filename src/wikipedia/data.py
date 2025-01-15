from torch_geometric.datasets import WikiCS
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import typer

# Downloaded from: https://github.com/pmernyei/wiki-cs-dataset
# Using: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/wikics.html#WikiCS

class WikiDataset():
    """ Wiki dataset class. 
    Can download, generate and load the data.
    Can also return dataloaders for training, validation and testing.
    """
    def __init__(self, 
                 data_path: str = typer.Argument("", help="Path to the data."),
                 generate_data: bool = typer.Option(False)) -> None:
        """
        Initialize the dataset. Downloads the data to the path if it does not exist.
        Creates raw and processed data folders itself so give it "data/"
        Then saves a PyG Data object in the dataset attribute.
        Args:
            data_path (str): Path to the data
            generate_data (bool): Whether to generate the data from torch_geometric.datasets.WikiCS
        """
        self.data_path = data_path
        
        if generate_data:
            self.get_data(self.data_path)

        self.dataset = self.load_data()

    def get_data(data_path: str,
                normalize_bool: bool = False) -> None:
        """
        Pytorch Geometric has an inbuilt function for the WikiCS dataset.
        This function downloads the raw data and then it is possible to process it.
        There are some transformations that can be applied to the data.    
        Args:
            data_path (str): Path to the data
            normalize_bool (bool): Whether to normalize
        """
        # Ensure the output folder exists
        transform = NormalizeFeatures() if normalize_bool else None
        
        data = WikiCS(root=data_path, transform=transform)
        data.process()
        print("Data preprocessing complete.")

    def load_data(self):
        """
        The data is loaded from the path.
        Saving the data in get_data gives a tuple with (true_data, None). 
        This is wrong so we take the data from the tuple.
        It then becomes a python dict, we want a PyG Data object.
        Returns: PyG Data object with the data.
        """
        loaded_tuple = torch.load("data/processed/data_undirected.pt")
        # Assume it's something like (data_object, None)
        if isinstance(loaded_tuple, tuple):
            data = loaded_tuple[0]  # The PyG Data-like object
        else:
            data = loaded_tuple

        # Make sure it's a PyG Data object
        if not isinstance(data, Data):
            data = Data(**data._asdict()) if hasattr(data, "_asdict") else Data(**dict(data))

        return data

    def __len__(self):
        return len(self.dataset.x)
    def __getitem__(self, idx):
        return self.dataset.x[idx], self.dataset.edge_index, self.dataset.y
    
    def train_loader(self, batch_size: int = 32):
        """ Dataloader for training data. Takes the training mask and returns the data. """
        train_mask = self.dataset.train_mask
        train_data = self.dataset[train_mask]
        return DataLoader([train_data], batch_size=batch_size, shuffle=True)
    def val_loader(self, batch_size: int = 32):
        """ Dataloader for validation data. Takes the validation mask and returns the data. """
        val_mask = self.dataset.val_mask
        val_data = self.dataset[val_mask]
        return DataLoader([val_data], batch_size=batch_size, shuffle=False)
    def test_loader(self, batch_size: int = 32):
        """ Dataloader for test data. Takes the test mask and returns the data. """
        test_mask = self.dataset.test_mask
        test_data = self.dataset[test_mask]
        return DataLoader([test_data], batch_size=batch_size, shuffle=False)
