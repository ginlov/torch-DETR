import torch

from typing import List, Tuple
from scipy.optimize import linear_sum_assignment
from src.utils import CONSTANTS
from src.data.transforms import box_to_xy, box_iou, box_iou_matrix

def l_hung(
    labs: torch.Tensor,
    lab_preds: torch.Tensor,
    bbox: torch.Tensor,
    bbox_preds: torch.Tensor,
    masks: torch.Tensor
    ) -> torch.Tensor:
    """
    Compute the matching loss between ground truth and predictions.
     
    Args:
        labs (torch.Tensor): Ground truth labels. [bz x N]
        lab_preds (torch.Tensor): Predicted labels. [bz x N x num_classes]
        bbox (torch.Tensor): Ground truth bounding boxes. [bz x N x 4]
        bbox_preds (torch.Tensor): Predicted bounding boxes. [bz x N x 4]
    
    Returns:
        torch.Tensor: The computed matching loss.
    """
    lab_preds = torch.nn.functional.softmax(lab_preds, dim=-1) # [bz x N x num_classes]

    # Compute bbox loss class loss
    loss_label = -torch.log(lab_preds.gather(2, labs.unsqueeze(-1)).squeeze(-1)) # [bz x N]
    loss_label = loss_label.mean() # Only consider non-empty objects
    loss_bbox = l_box(bbox, bbox_preds, masks) # L_box loss

    return loss_label + loss_bbox

def l_box(bbox, bbox_preds, mask):
    """
    Compute the bounding box loss.
    
    Args:
        bbox (torch.Tensor): Ground truth bounding boxes. [bz x N x 4]
        bbox_preds (torch.Tensor): Predicted bounding boxes. [bz x N x 4]
        mask (torch.Tensor): Mask indicating which boxes to consider. [bz x N]
        
    Returns:
        torch.Tensor: The computed bounding box loss."""
    # TODO: Add weights to the losses
    l1_loss = L1_loss(bbox, bbox_preds, mask)
    liou_loss = iou_loss(bbox, bbox_preds, mask)
    return l1_loss + liou_loss

def iou_loss(bbox, bbox_preds, mask):
    """
    Compute the IoU loss between ground truth and predicted bounding boxes.
    
    Args:
        bbox (torch.Tensor): Ground truth bounding boxes. [bz x N x 4]
        bbox_preds (torch.Tensor): Predicted bounding boxes. [bz x N x 4]
        mask (torch.Tensor): Mask indicating which boxes to consider. [bz x N]
        
    Returns:
        torch.Tensor: The computed IoU loss."""
    if mask.sum() == 0:
        return torch.tensor(0.0, device=bbox.device)
    # Only keep masked items
    bbox = bbox[mask]
    bbox_preds = bbox_preds[mask]

    # Convert [cx, cy, w, h] to [x1, y1, x2, y2]
    bbox_xy = box_to_xy(bbox)
    bbox_preds_xy = box_to_xy(bbox_preds)

    # Compute IoU
    iou = box_iou(bbox_xy, bbox_preds_xy)
    loss = 1 - iou
    return loss.mean()

def L1_loss(bbox, bbox_preds, mask):
    """
    Compute the L1 loss between ground truth and predicted bounding boxes.
    
    Args:
        bbox (torch.Tensor): Ground truth bounding boxes. [bz x N x 4]
        bbox_preds (torch.Tensor): Predicted bounding boxes. [bz x N x 4]
        mask (torch.Tensor): Mask indicating which boxes to consider. [bz x N]
        
    Returns:
        torch.Tensor: The computed L1 loss."""
    if mask.sum() == 0:
        return torch.tensor(0.0, device=bbox.device)
    mask = mask.unsqueeze(-1).expand_as(bbox)  #[bz x N x 4]
    loss = torch.abs(bbox[mask] - bbox_preds[mask]).sum(dim=-1).mean()
    return loss

class HungarianMatcher(torch.nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.
    """
    def __init__(self):
        super().__init__()
        self.cost_class = 1.0
        self.cost_bbox = 1.0
        self.cost_iou = 1.0

    @torch.no_grad()
    def forward(
            self,
            labs: List[torch.Tensor],
            lab_preds: torch.Tensor,
            bbox: List[torch.Tensor],
            bbox_preds: torch.Tensor
        ):
        """
        Perform the Hungarian matching between ground truth and predictions.
        
        Args:
            # TODO: Don't need to batch the labs input here, since we compute for each
            sample anyway.
            labs (List[torch.Tensor]): Ground truth labels. [bz x any] 
            lab_preds (torch.Tensor): Predicted labels. [bz x N x num_classes]
            bbox (List[torch.Tensor]): Ground truth bounding boxes. [bz x any x 4]
            bbox_preds (torch.Tensor): Predicted bounding boxes. [bz x N x 4]
            
        Returns:
            list of tuples: Each tuple contains (ground_truth_index, prediction_index)
            for a batch item.
        """
        # Force batch dimension
        if lab_preds.dim() == 2:
            lab_preds = lab_preds.unsqueeze(0)
        if bbox_preds.dim() == 2:
            bbox_preds = bbox_preds.unsqueeze(0)

        batch_size = len(labs)
        indices = []

        for b in range(batch_size):
            tgt_labels = labs[b].squeeze(-1)  # [n_gt]
            pred_logits = lab_preds[b]        # [N, num_classes]
            tgt_bbox = bbox[b]                # [n_gt, 4]
            pred_bbox = bbox_preds[b]         # [N, 4]

            # Classification cost
            pred_probs = pred_logits.softmax(-1)  # [N, num_classes]
            cost_class = -pred_probs[:, tgt_labels]  # [N_pred, N_gt]

            # L1 bbox cost
            cost_bbox = torch.cdist(pred_bbox, tgt_bbox, p=1)  # [N_pred, N_gt]

            # IoU cost (1 - IoU)
            pred_boxes_xy = box_to_xy(pred_bbox)
            tgt_boxes_xy = box_to_xy(tgt_bbox)

            # Compute pairwise IoU matrix
            iou_matrix = box_iou_matrix(pred_boxes_xy, tgt_boxes_xy)

            cost_iou = 1 - iou_matrix  # [N_pred, N_gt]

            # Final cost matrix
            C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_iou * cost_iou
            C = C.cpu().detach().numpy() # [N_pred, N_gt]

            pred_indices, gt_indices = linear_sum_assignment(C) # [n_gt]
            indices.append((gt_indices, pred_indices))

        return indices # [bz x (n_gt, n_gt)] n_gt is different for each batch item

class DETRLoss(torch.nn.Module):
    """
    This class computes the DETR loss.
    It contains the classification loss and the bounding box loss and also the IoU loss.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_class = 1.0
        self.cost_bbox = 1.0
        self.cost_iou = 1.0
        self.matcher = HungarianMatcher()

    def forward(self,
            labs: List[torch.Tensor],
            lab_preds: torch.Tensor,
            bbox: List[torch.Tensor],
            bbox_preds: torch.Tensor
    ):
        """
        Compute the DETR loss.
        
        Args:
            labs (List[torch.Tensor]): Ground truth labels. [bz x n_gt], n_gt is different for each 
            batch item
            lab_preds (torch.Tensor): Predicted labels. [bz x N x num_classes]
            bbox (List[torch.Tensor]): Ground truth bounding boxes. [bz x n_gt x 4], n_gt is different for each batch item
            bbox_preds (torch.Tensor): Predicted bounding boxes. [bz x N x 4]
            
        Returns:
            torch.Tensor: The computed DETR loss.
        """
        indices = self.matcher(labs, lab_preds, bbox, bbox_preds) # [bz x (n_gt)]
        N = lab_preds.shape[1]  # Number of predictions
        device = lab_preds.device

        batched_labs, batched_lab_preds, batched_bbox, batched_bbox_preds, masks = [], [], [], [], []
        for i, (gt_indices, pred_indices) in enumerate(indices):
            # Ground truth
            labs_reordered = labs[i][gt_indices]
            bbox_reordered = bbox[i][gt_indices]
            pad_len = N - len(gt_indices)
            lab_dtype = labs.dtype if len(labs) > 9 else torch.int64
            bbox_dtype = bbox.dtype if len(bbox) > 9 else torch.float32
            labs_padded = torch.cat([labs_reordered, torch.full((pad_len,), CONSTANTS.NO_OBJECT_LABEL, dtype=lab_dtype, device=device)])
            bbox_padded = torch.cat([bbox_reordered, torch.zeros(pad_len, 4, dtype=bbox_dtype, device=device)])
            mask = torch.cat([torch.ones(len(gt_indices), device=device), torch.zeros(pad_len, device=device)]) # Mask for background

            # Predictions
            all_pred_indices = list(pred_indices) + [j for j in range(N) if j not in pred_indices]
            lab_preds_reordered = lab_preds[i][all_pred_indices]
            bbox_preds_reordered = bbox_preds[i][all_pred_indices]

            batched_labs.append(labs_padded)
            batched_bbox.append(bbox_padded)
            batched_lab_preds.append(lab_preds_reordered)
            batched_bbox_preds.append(bbox_preds_reordered)
            masks.append(mask)

        # Stack for batch
        batched_labs = torch.stack(batched_labs) # [bz x N]
        batched_bbox = torch.stack(batched_bbox) # [bz x N x 4]
        batched_lab_preds = torch.stack(batched_lab_preds) # [bz x N x num_classes]
        batched_bbox_preds = torch.stack(batched_bbox_preds) # [bz x N x 4]
        masks = torch.stack(masks).bool() # [bz x N]

        return l_hung(batched_labs, batched_lab_preds, batched_bbox, batched_bbox_preds, masks)
