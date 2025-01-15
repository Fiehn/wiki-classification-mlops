from pathlib import Path
from torch_geometric.datasets import WikiCS
from torch.utils.data import Dataset
import typer

# Downloaded from: https://github.com/pmernyei/wiki-cs-dataset
# Using: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/wikics.html#WikiCS

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        """
        Initialize the dataset.
        
        Args:
            raw_data_path (Path): Path where raw data is or will be downloaded.
        """
        self.data_path = raw_data_path
        self.dataset = None  # Placeholder for the WikiCS dataset

    def __len__(self) -> int:
        """Return the length of the dataset."""
        if self.dataset is None:
            raise ValueError("Dataset is not loaded. Call preprocess() first.")
        return len(self.dataset)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        if self.dataset is None:
            raise ValueError("Dataset is not loaded. Call preprocess() first.")
        return self.dataset[index]

    def preprocess(self, output_folder: Path) -> None:
        """
        Preprocess the raw data and save it to the output folder.
        
        Args:
            output_folder (Path): Directory where processed data will be saved.
        """
        print(f"Downloading and processing data to: {output_folder}...")
        self.dataset = WikiCS(root=str(output_folder))
        print("Data preprocessing complete.")


def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    """
    Preprocess the WikiCS dataset and save it to the specified output folder.
    
    Args:
        output_folder (Path): Directory where processed data will be saved.
    """
    # Ensure the output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)

