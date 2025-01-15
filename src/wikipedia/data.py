from torch_geometric.datasets import WikiCS
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import typer
import os

# Downloaded from: https://github.com/pmernyei/wiki-cs-dataset
# Using: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/wikics.html#WikiCS

class WikiDataset():
    """ Wiki dataset class. 
    Can download, generate and load the data.
    Can also return dataloaders for training, validation and testing.
    """
    def __init__(self) -> None:
        """
        Initialize the dataset. Downloads the data to the path if it does not exist.
        Creates raw and processed data folders itself so give it "data/"
        Then saves a PyG Data object in the dataset attribute.
        Args:
            data_path (str): Path to the data
            generate_data (bool): Whether to generate the data from torch_geometric.datasets.WikiCS
        """
        
        # Check if the data exists
        if not os.path.exists("data/processed/data_undirected.pt"):
            print("Data does not exist. Downloading...")
            self.get_data()
            
        self.load_data()
        self.dataset = self.load_data()

    def __len__(self):
        """ Returns the length of the dataset. """
        return len(self.dataset)

    def __getitem__(self, idx):
        """ Returns the item at the index. """
        return self.dataset[idx]
    
    @staticmethod
    def download_data(self):
        """ Downloads the data to the path. """
        WikiCS(self.data_path)
        print("Data downloaded.")
    
    @staticmethod
    def get_data(normalize_bool: bool = False) -> None:
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
        
        data = WikiCS(root="data/", transform=transform)

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
        loaded_tuple = torch.load("data/processed/data_undirected.pt",weights_only=True)
        # Assume it's something like (data_object, None)
        if isinstance(loaded_tuple, tuple):
            data = loaded_tuple[0]  # The PyG Data-like object
        else:
            data = loaded_tuple

        # Make sure it's a PyG Data object
        if not isinstance(data, Data):
            data = Data(**data._asdict()) if hasattr(data, "_asdict") else Data(**dict(data))

        return data

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


if __name__ == "__main__":
    da = WikiDataset()
    