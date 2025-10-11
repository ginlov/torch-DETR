import os
import sys
import pytest
import torch

from PIL import Image

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import src.data.transforms as T

@pytest.fixture
def dummy_img():
    """Create a dummy image for testing."""
    return Image.new("RGB", (32, 32), color="white")

@pytest.fixture
def dummy_target():
    """Create a dummy target for testing."""
    return {
        "boxes": torch.tensor([[5, 5, 20, 20]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64)
    }

def test_to_tensor(dummy_img, dummy_target):
    transform = T.ToTensor()
    img, target = transform(dummy_img, dummy_target)
    assert isinstance(img, torch.Tensor)
    assert img.shape[1:] == (32, 32)
    assert target == dummy_target

def test_hflip(dummy_img, dummy_target):
    img, target = T.hflip(dummy_img, dummy_target)
    assert isinstance(img, Image.Image)
    assert target["boxes"][0, 0] == 32 - dummy_target["boxes"][0, 2]

