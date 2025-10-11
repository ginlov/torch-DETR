import os
import sys

import numpy as np
import torch

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from src.losses.loss import HungarianMatcher


def test_hungarian_matcher_shapes():
    """
    Test that HungarianMatcher returns correct shapes and types.
    """
    matcher = HungarianMatcher()
    labs = [torch.tensor([0, 1], dtype=torch.int64), torch.tensor([1, 2, 0], dtype=torch.int64)]
    lab_preds = torch.tensor([
        [[2.0, 1.0, 0.1], [0.1, 2.0, 1.0], [1.0, 0.1, 2.0]],
        [[0.1, 2.0, 1.0], [2.0, 1.0, 0.1], [1.0, 0.1, 2.0]]
    ])
    bbox = [
        torch.tensor([[0.2, 0.2, 0.5, 0.5], [0.1, 0.1, 0.2, 0.3]], dtype=torch.float32),
        torch.tensor([[0.3, 0.3, 0.4, 0.4], [0.2, 0.2, 0.3, 0.3], [0.1, 0.1, 0.2, 0.3]], dtype=torch.float32)
    ]
    bbox_preds = torch.tensor([
        [[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1], [0.3, 0.3, 0.4, 0.4]],
        [[0.1, 0.1, 0.2, 0.2], [0.2, 0.2, 0.3, 0.3], [0.4, 0.4, 0.5, 0.5]]
    ])
    indices = matcher.forward(labs, lab_preds, bbox, bbox_preds)
    assert isinstance(indices, list)
    assert len(indices[0]) == 2  # (gt_indices, pred_indices)
    assert all(isinstance(arr, (list, torch.Tensor, np.ndarray)) for arr in indices[0])
