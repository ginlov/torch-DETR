import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from src.data.dataset import CPPE5Dataset
import torch

def test_cppe5_dataset_loading():
    """
    Test the CPPE5Dataset class for loading data correctly.
    """
    dataset_path = os.path.join(parent, 'assets', 'CPPE-5')  # Adjust path as necessary
    dataset = CPPE5Dataset(folder_path=dataset_path, partition='train')

    # Check that dataset is not empty
    assert len(dataset) > 0, "Dataset is empty."

    # Check that each item in the dataset has the correct keys and types
    for i in range(min(5, len(dataset))):  # Check first 5 samples
        images, targets = dataset[i]
        assert 'boxes' in targets, "Sample missing 'boxes' key."
        assert 'labels' in targets, "Sample missing 'labels' key."
        assert isinstance(images, torch.Tensor), "'image' is not a torch.Tensor."
        assert isinstance(targets['boxes'], torch.Tensor), "'boxes' is not a torch.Tensor."
        assert isinstance(targets['labels'], torch.Tensor), "'labels' is not a torch.Tensor."

    print("CPPE5Dataset loading test passed.")