import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch

from src.metrics import AP
from src.utils import CONSTANTS


def test_ap_single_detection():
    """One GT and one perfect prediction (batched tensors)"""
    # batch = 1, num_queries = 2
    labels = torch.tensor([[1, CONSTANTS.NO_OBJECT_LABEL]])
    bboxes = torch.tensor([[
        [0.25, 0.25, 0.3, 0.3],
        [0.0,  0.0,  0.0, 0.0]
    ]])

    pred_logits = torch.tensor([[
        [0.0, 5.0],
        [0.0, 0.0]
    ]])
    pred_bboxes = torch.tensor([[
        [0.25, 0.25, 0.3, 0.3],
        [0.0,  0.0,  0.0, 0.0]
    ]])

    ap = AP(list(labels), list(bboxes), list(pred_logits), list(pred_bboxes),
            iou_threshold=0.5, num_classes=1)

    assert isinstance(ap, float)
    assert ap > 0.9, f"AP too low, got {ap}"


def test_ap_with_background_prediction():
    """Model predicts background only -> AP should be 0"""
    labels = torch.tensor([[1, CONSTANTS.NO_OBJECT_LABEL]])
    bboxes = torch.tensor([[
        [0.25, 0.25, 0.3, 0.3],
        [0.0,  0.0,  0.0, 0.0]
    ]])

    pred_logits = torch.tensor([[
        [5.0, 0.0],
        [0.0, 0.0]
    ]])
    pred_bboxes = torch.tensor([[
        [0.0, 0.0, 0.1, 0.1],
        [0.0, 0.0, 0.0, 0.0]
    ]])

    ap = AP(list(labels), list(bboxes), list(pred_logits), list(pred_bboxes),
            iou_threshold=0.5, num_classes=1)

    assert ap == 0.0, f"AP should be 0, got {ap}"


def test_ap_multiple_images():
    """Two images, each with one GT and correct prediction"""
    labels = torch.tensor([
        [1, CONSTANTS.NO_OBJECT_LABEL],
        [1, CONSTANTS.NO_OBJECT_LABEL]
    ])
    bboxes = torch.tensor([
        [[0.25, 0.25, 0.3, 0.3], [0.0, 0.0, 0.0, 0.0]],
        [[0.6, 0.6, 0.2, 0.2],   [0.0, 0.0, 0.0, 0.0]]
    ])

    pred_logits = torch.tensor([
        [[0.0, 5.0], [0.0, 0.0]],
        [[0.0, 5.0], [0.0, 0.0]]
    ])
    pred_bboxes = torch.tensor([
        [[0.25, 0.25, 0.3, 0.3], [0.0, 0.0, 0.0, 0.0]],
        [[0.6, 0.6, 0.2, 0.2],   [0.0, 0.0, 0.0, 0.0]]
    ])

    ap = AP(list(labels), list(bboxes), list(pred_logits), list(pred_bboxes),
            iou_threshold=0.5, num_classes=1)

    assert ap > 0.9, f"AP too low, got {ap}"
