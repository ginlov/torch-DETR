import sys
import os
import pytest
import torch

# Add project root to sys.path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from src.data.dataset import CPPE5Dataset, collate_fn


@pytest.fixture(scope="module")
def cppe5_dataset():
    """
    Fixture to construct the CPPE5Dataset only once per test module.
    """
    dataset_path = os.path.join(parent, 'assets', 'CPPE-5')  # Adjust path as necessary
    dataset = CPPE5Dataset(folder_path=dataset_path, partition='train')
    return dataset


def test_cppe5_dataset_loading(cppe5_dataset):
    """
    Test the CPPE5Dataset class for loading data correctly.
    """
    dataset = cppe5_dataset

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


def test_cppe5_collate_fn(cppe5_dataset):
    """
    Test the collate_fn function for batching data correctly.
    """
    dataset = cppe5_dataset

    # Create a small batch
    batch_size = 4
    samples = [dataset[i] for i in range(batch_size)]
    data_batch = collate_fn(samples)
    images = data_batch["images"]
    masks = data_batch["masks"]
    targets = data_batch["targets"]

    # Check that images is a tensor of correct shape
    assert isinstance(images, torch.Tensor), "'images' is not a torch.Tensor."
    assert images.shape[0] == batch_size, f"Batch size mismatch: expected {batch_size}, got {images.shape[0]}."
    assert images.shape[1] == 3, f"Image channel mismatch: expected 3, got {images.shape[1]}."

    # Check that masks is a tensor of correct shape
    assert isinstance(masks, torch.Tensor), "'masks' is not a torch.Tensor."
    assert masks.shape[0] == batch_size, f"Batch size mismatch in masks: expected {batch_size}, got {masks.shape[0]}."
    assert masks.shape[1] == images.shape[2], f"Mask height mismatch: expected {images.shape[2]}, got {masks.shape[1]}."
    assert masks.shape[2] == images.shape[3], f"Mask width mismatch: expected {images.shape[3]}, got {masks.shape[2]}."

    # Check that targets is a list of dictionaries with correct keys
    assert isinstance(targets, dict), "'targets' is not a dictioary."
    assert "boxes" in targets, "'boxes' key missing in targets."
    assert "labels" in targets, "'labels' key missing in targets."
    assert len(targets["boxes"]) == batch_size, \
    f"'boxes' length mismatch: expected {batch_size}, got {len(targets['boxes'])}."
    assert len(targets["labels"]) == batch_size, \
    f"'labels' length mismatch: expected {batch_size}, got {len(targets['labels'])}."
    assert isinstance(targets["boxes"][0], torch.Tensor), \
    "'boxes' in target is not a torch.Tensor."
    assert isinstance(targets["labels"][0], torch.Tensor), \
    "'labels' in target is not a torch.Tensor."

def test_cppe5_dataloader(cppe5_dataset):
    """
    Test the DataLoader with CPPE5Dataset and collate_fn.
    """
    from torch.utils.data import DataLoader

    batch_size = 4
    dataloader = DataLoader(cppe5_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=10)

    # Fetch a single batch
    for data_batch in dataloader:
        images = data_batch["images"]
        masks = data_batch["masks"]
        targets = data_batch["targets"]
        # Check that images is a tensor of correct shape
        assert isinstance(images, torch.Tensor), "'images' is not a torch.Tensor."
        assert images.shape[0] == batch_size, f"Batch size mismatch: expected {batch_size}, got {images.shape[0]}."
        assert images.shape[1] == 3, f"Image channel mismatch: expected 3, got {images.shape[1]}."

        # Check that masks is a tensor of correct shape
        assert isinstance(masks, torch.Tensor), "'masks' is not a torch.Tensor."
        assert masks.shape[0] == batch_size, f"Batch size mismatch in masks: expected {batch_size}, got {masks.shape[0]}."
        assert masks.shape[1] == images.shape[2], f"Mask height mismatch: expected {images.shape[2]}, got {masks.shape[1]}."
        assert masks.shape[2] == images.shape[3], f"Mask width mismatch: expected {images.shape[3]}, got {masks.shape[2]}."

        # Check that targets is a list of dictionaries with correct keys
        assert isinstance(targets, dict), "'targets' is not a dictionary."
        assert "boxes" in targets, "'boxes' key missing in targets."
        assert "labels" in targets, "'labels' key missing in targets."
        assert len(targets["boxes"]) == batch_size, \
        f"'boxes' length mismatch: expected {batch_size}, got {len(targets['boxes'])}."
        assert len(targets["labels"]) == batch_size, \
        f"'labels' length mismatch: expected {batch_size}, got {len(targets['labels'])}."
        assert isinstance(targets["boxes"][0], torch.Tensor), \
        "'boxes' in target is not a torch.Tensor."
        assert isinstance(targets["labels"][0], torch.Tensor), \
        "'labels' in target is not a torch.Tensor."

        break  # Only test one batch
