import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
import numpy as np
import random
from src.losses.loss import l_box, iou_loss, L1_loss, DETRLoss, l_hung
from src.detr.detr import build_detr

def test_l_hung_perfect_match():
    # 2 objects, perfect prediction
    labs = torch.tensor([[1, 2]])
    lab_preds = torch.tensor([[[0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]], dtype=torch.float32)
    bbox = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]])
    bbox_preds = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]])
    loss = l_hung(labs, lab_preds, bbox, bbox_preds, torch.ones_like(labs, dtype=torch.bool))
    print(loss)
    assert loss > 0, f"l_hung loss should be positive all time, got {loss.item()}"
    assert loss < 1e-4, f"l_hung loss should be near zero for perfect match, got {loss.item()}"

def test_l_hung_imperfect_match():
    # 2 objects, one prediction is off
    labs = torch.tensor([[1, 2]])
    lab_preds = torch.tensor([[[0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]], dtype=torch.float32)
    bbox = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]])
    bbox_preds = torch.tensor([[[0.6, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]])
    loss = l_hung(labs, lab_preds, bbox, bbox_preds, torch.ones_like(labs, dtype=torch.bool))
    assert loss > 0, "l_hung loss should be positive for imperfect match"

def test_l1_loss_basic():
    bbox = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]])
    bbox_preds = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]])
    mask = torch.tensor([[1, 0]], dtype=torch.bool)
    loss = L1_loss(bbox, bbox_preds, mask)
    assert loss >= 0, f"L1 loss should be positive all time, got {loss.item()}"
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
    assert loss >= 0, f"IoU loss should be positive all time, got {loss.item()}"
    assert loss < 3e-6, f"IoU loss should be very close to zero for perfect match, got {loss.item()}"

def test_l_box_combined_loss():
    bbox = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]])
    bbox_preds = torch.tensor([[[0.6, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]])
    mask = torch.tensor([[1, 1]], dtype=torch.bool)
    loss = l_box(bbox, bbox_preds, mask)
    assert loss >= 0, "Combined loss should be positive for imperfect match"

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
    assert loss >= 0, f"DETR loss should be positive all time, got {loss.item()}"
    assert loss < 1e-4, f"DETR loss should be near zero for perfect match, got {loss.item()}"

def test_detr_loss_imperfect_match():
    detr_loss = DETRLoss()
    labs = torch.tensor([[1, 2]])
    lab_preds = torch.tensor([[[0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]], dtype=torch.float32)
    bbox = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]])
    bbox_preds = torch.tensor([[[0.6, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]])
    loss = detr_loss(labs, lab_preds, bbox, bbox_preds)
    assert loss >= 0, "DETR loss should be positive for imperfect match"

# TODO: Add logical tests for HungarianMatcher
# TODO: Currently, the DETRLoss returns negative when mask is empty. This is due to the label loss accounts for
# the background class when the bbox loss is zero. Fix this.

def test_detr_can_improve_on_detr_loss():
    """
    Test that the DETR model can reduce DETRLoss on a small random dataset.
    Checks that the average loss in the last steps is lower than in the first steps.
    """

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Small model for speed
    model = build_detr('resnet50', num_classes=5, num_queries=10, d_model=64, nhead=4, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=128)
    model.to(device)
    model.train()

    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 64, 64, device=device)
    target_labels = torch.randint(0, 6, (batch_size, 10), device=device)  # 5 classes + 1 no-object
    target_boxes = torch.rand(batch_size, 10, 4, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = DETRLoss()

    losses = []
    for step in range(300):
        optimizer.zero_grad()
        class_preds, bbox_preds = model(dummy_input)
        loss = loss_fn(target_labels, class_preds, target_boxes, bbox_preds)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    N = 50
    first_avg = sum(losses[:N]) / N
    last_avg = sum(losses[-N:]) / N
    print(f"Average DETRLoss first {N} steps: {first_avg}")
    print(f"Average DETRLoss last {N} steps: {last_avg}")
    assert last_avg < first_avg, "DETRLoss did not decrease during training"
