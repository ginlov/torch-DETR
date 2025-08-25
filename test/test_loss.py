import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
import numpy as np
from src.losses.loss import l_box, iou_loss, L1_loss, HungarianMatcher, DETRLoss, l_hung

def test_l_hung_perfect_match():
    # 2 objects, perfect prediction
    labs = torch.tensor([[1, 2]])
    lab_preds = torch.tensor([[[0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]], dtype=torch.float32)
    bbox = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]])
    bbox_preds = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]])
    loss = l_hung(labs, lab_preds, bbox, bbox_preds)
    print(loss)
    assert loss < 1e-5, f"l_hung loss should be near zero for perfect match, got {loss.item()}"

def test_l_hung_imperfect_match():
    # 2 objects, one prediction is off
    labs = torch.tensor([[1, 2]])
    lab_preds = torch.tensor([[[0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]], dtype=torch.float32)
    bbox = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]])
    bbox_preds = torch.tensor([[[0.6, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]])
    loss = l_hung(labs, lab_preds, bbox, bbox_preds)
    assert loss > 0, "l_hung loss should be positive for imperfect match"

def test_l1_loss_basic():
    bbox = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]])
    bbox_preds = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]])
    mask = torch.tensor([[1, 0]], dtype=torch.bool)
    loss = L1_loss(bbox, bbox_preds, mask)
    assert torch.isclose(loss, torch.tensor(0.0)), "L1 loss should be zero for perfect match"

def test_liou_loss_basic():
    bbox = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]])
    bbox_preds = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]])
    mask = torch.tensor([[1, 0]], dtype=torch.bool)
    print(bbox.shape)
    print(bbox_preds.shape)
    print(mask.shape)
    loss = iou_loss(bbox, bbox_preds, mask)
    print(loss)
    assert loss < 3e-6, f"IoU loss should be very close to zero for perfect match, got {loss.item()}"

def test_l_box_combined_loss():
    bbox = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]])
    bbox_preds = torch.tensor([[[0.6, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]])
    mask = torch.tensor([[1, 1]], dtype=torch.bool)
    loss = l_box(bbox, bbox_preds, mask)
    assert loss > 0, "Combined loss should be positive for imperfect match"

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

def test_l1_loss_empty_mask():
    bbox = torch.randn(1, 2, 4)
    bbox_preds = torch.randn(1, 2, 4)
    mask = torch.tensor([[0, 0]], dtype=torch.bool)
    loss = L1_loss(bbox, bbox_preds, mask)
    assert loss == 0, "L1 loss should be zero if mask is empty"

def test_liou_loss_empty_mask():
    bbox = torch.randn(1, 2, 4)
    bbox_preds = torch.randn(1, 2, 4)
    mask = torch.tensor([[0, 0]], dtype=torch.bool)
    loss = iou_loss(bbox, bbox_preds, mask)
    assert loss == 0, "IoU loss should be zero if mask is empty"

def test_detr_loss_perfect_match():
    detr_loss = DETRLoss()
    labs = torch.tensor([[1, 2]])
    lab_preds = torch.tensor([[[0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]], dtype=torch.float32)
    bbox = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]])
    bbox_preds = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]])
    loss = detr_loss(labs, lab_preds, bbox, bbox_preds)
    assert loss < 1e-5, f"DETR loss should be near zero for perfect match, got {loss.item()}"

def test_detr_loss_imperfect_match():
    detr_loss = DETRLoss()
    labs = torch.tensor([[1, 2]])
    lab_preds = torch.tensor([[[0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]], dtype=torch.float32)
    bbox = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]])
    bbox_preds = torch.tensor([[[0.6, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]])
    loss = detr_loss(labs, lab_preds, bbox, bbox_preds)
    assert loss > 0, "DETR loss should be positive for imperfect match"

# TODO: Add logical tests for HungarianMatcher
# TODO: Currently, the DETRLoss returns negative when mask is empty. This is due to the label loss accounts for
# the background class when the bbox loss is zero. Fix this.