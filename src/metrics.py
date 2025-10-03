import torch

from typing import List
from src.utils import CONSTANTS
from src.data.transforms import box_iou_matrix, box_to_xy

@torch.no_grad()
def AP(
        labels: List[torch.Tensor], 
        bboxes: List[torch.Tensor], 
        pred_logits: torch.Tensor, 
        pred_bboxes: torch.Tensor, 
        iou_threshold: float = 0.5, 
        num_classes: int = 91
    ) -> float:
    """Implementation of Average Precision (AP) metric for object detection.
    labels and bboxes are padded to the same length of num_queries with the value CONSTANTS.NO_OBJECT_LABEL

    Args:
        labels (torch.Tensor): ground truth labels [num_valid, num_queries]
        bboxes (torch.Tensor): ground truth bounding boxes [num_valid, num_queries, 4]
        pred_logits (torch.Tensor): predicted logits [num_valid, num_queries, num_classes]
        pred_bboxes (torch.Tensor): predicted bounding boxes [num_valid, num_queries, 4]
        iou_threshold (float, optional): IoU threshold to consider a prediction as true positive. Defaults to 0.5.
        num_classes (int, optional): number of classes. Defaults to 91.

    Returns:
        float: AP score
    """
    assert len(labels) == len(bboxes) == len(pred_logits) == len(pred_bboxes), "Length of labels, bboxes, pred_logits and pred_bboxes must be the same"
    all_ap = []
    for c in range(0, num_classes+1):
        if c == CONSTANTS.NO_OBJECT_LABEL:
            continue
        true_positives = []
        scores = []
        num_gt = 0

        for i in range(len(labels)):
            label = labels[i]
            bbox = bboxes[i]
            pred_logit = pred_logits[i] # [num_queries, num_classes]
            pred_bbox = pred_bboxes[i] # [num_queries, 4]

            # Filter out padding
            valid_mask = label != CONSTANTS.NO_OBJECT_LABEL
            label = label[valid_mask]
            bbox = bbox[valid_mask]
            
            num_gt += (label == c).sum().item()
            if len(label) == 0:
                continue

            # --- Predictions ---
            probs = pred_logit.softmax(-1)   # [num_queries, num_classes]
            scores_c = probs[:, c]           # confidence scores for class c [num_queries]
            boxes_c = pred_bbox              # [num_queries, 4]

            # Keep only predictions with reasonable confidence
            keep = scores_c > 0.0
            scores_c = scores_c[keep]
            boxes_c = boxes_c[keep]

            # Sort predictions by confidence
            if len(scores_c) == 0:
                continue
            scores_sorted, idx = scores_c.sort(descending=True)
            boxes_sorted = boxes_c[idx]

            matched = torch.zeros(len(label), dtype=torch.bool, device=label.device)

            for pred_box, score in zip(boxes_sorted, scores_sorted):
                scores.append(score.item())

                # Only match with GTs of this class
                gt_mask = label == c
                gt_boxes = box_to_xy(bbox[gt_mask]) # [num_gt_class, 4]
                pred_box = box_to_xy(pred_box.unsqueeze(0)).squeeze(0)  # [4]
                if len(gt_boxes) == 0:
                    # no GT of this class in this image
                    true_positives.append(0)
                    continue

                # compute IoU with all GT boxes of class c
                ious = box_iou_matrix(pred_box.unsqueeze(0), gt_boxes)[0]  # [num_gt_class]
                best_iou, best_idx = ious.max(0)
                if best_iou >= iou_threshold and not matched[gt_mask.nonzero()[best_idx]]:
                    true_positives.append(1)
                    # mark this GT as matched
                    matched[gt_mask.nonzero()[best_idx]] = True
                else:
                    true_positives.append(0)

        if num_gt == 0:
            continue

        # --- Precision-Recall curve ---
        if len(scores) == 0:
            all_ap.append(0.0)
            continue

        scores = torch.tensor(scores)
        true_positives = torch.tensor(true_positives)

        # sort all predictions by score (global, across images)
        sorted_idx = scores.argsort(descending=True)
        true_positives = true_positives[sorted_idx]

        tp_cum = true_positives.cumsum(0)
        fp_cum = torch.arange(1, len(true_positives)+1) - tp_cum

        recalls = tp_cum / num_gt
        precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

        # make precision non-increasing (interpolation)
        precisions = torch.flip(torch.cummax(torch.flip(precisions, dims=[0]), dim=0)[0], dims=[0])

        # compute AP as area under PR curve (trapezoidal rule)
        # assumes precisions, recalls are 1D torch tensors, extrapolate to (0,1) at beginning
        recalls = torch.cat([torch.tensor([0.0], device=recalls.device), recalls])
        precisions = torch.cat([torch.tensor([1.0], device=precisions.device), precisions])
        ap = torch.trapz(precisions, recalls).item()
        all_ap.append(ap)

    if len(all_ap) == 0:
        return 0.0
    return sum(all_ap) / len(all_ap)  # subtract small value to avoid returning exactly 1.0
