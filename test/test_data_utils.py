import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import numpy as np
import torch

from src.data.utils import visualize_output


def test_visualize_output():
    batch_size = 2
    num_channels = 3
    height = 128
    width = 128
    images = torch.rand(batch_size, num_channels, height, width)
    masks = torch.ones(batch_size, height, width)
    image_ids = [101, 102]
    out_labs = [torch.tensor([1, 2]), torch.tensor([2])]
    out_bboxes = [
        torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]]),
        torch.tensor([[0.5, 0.5, 0.6, 0.6]]),
    ]
    gt_labs = [torch.tensor([1]), torch.tensor([2, 3])]
    gt_bboxes = [
        torch.tensor([[0.5, 0.5, 0.6, 0.6], [0.7, 0.7, 0.8, 0.8]]),
        torch.tensor([[0.1, 0.1, 0.2, 0.2]]),
    ]
    output = visualize_output(
        images, masks, image_ids, out_labs, out_bboxes, gt_labs, gt_bboxes
    )
    assert isinstance(output, dict)
    assert len(output) == batch_size
    for img_id, img in output.items():
        assert isinstance(img_id, int)
        assert isinstance(
            img, (np.ndarray, type(None))
        )  # img can be None if visualization fails
        if img is not None:
            assert (
                img.shape[0] == height
                and img.shape[1] == width
                and img.shape[2] == num_channels
            )
