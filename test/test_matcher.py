import os
import sys

import numpy as np
import torch

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from src.losses.loss import HungarianMatcher


def test_hungarian_matcher_shapes():
    matcher = HungarianMatcher()
    labs = torch.tensor([[[1], [2]]])
    lab_preds = torch.tensor([[[2.0, 1.0, 0.1], [0.1, 2.0, 1.0]]])
    bbox = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]])
    bbox_preds = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]])
    indices = matcher.forward(labs, lab_preds, bbox, bbox_preds)
    assert isinstance(indices, list)
    assert len(indices[0]) == 2  # (gt_indices, pred_indices)
    assert all(isinstance(arr, (list, torch.Tensor, np.ndarray)) for arr in indices[0])
