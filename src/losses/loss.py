import torch

from src.utils import CONSTANTS
from src.data.transforms import box_to_xy, box_iou, box_iou_matrix
from scipy.optimize import linear_sum_assignment

def l_hung(labs, lab_preds, bbox, bbox_preds):
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
    # Force batch dimension
    if labs.dim() == 1:
        labs = labs.unsqueeze(0)
    if bbox.dim() == 1:
        bbox = bbox.unsqueeze(0)
    if lab_preds.dim() == 2:
        lab_preds = lab_preds.unsqueeze(0)
    if bbox_preds.dim() == 2:
        bbox_preds = bbox_preds.unsqueeze(0)

    lab_preds = torch.nn.functional.softmax(lab_preds, dim=-1) # [bz x N x num_classes]
    lab_indices_pred = torch.argmax(lab_preds, dim=-1)
    empty_mask = (labs != CONSTANTS.NO_OBJECT_LABEL)
    match_mask = (labs == lab_indices_pred)
    final_mask = empty_mask & match_mask

    # Compute bbox loss class loss
    loss_label = -lab_preds.gather(2, labs.unsqueeze(-1)).squeeze(-1) # [bz x N]
    loss_label = loss_label.mean() # Only consider non-empty objects
    loss_bbox = l_box(bbox, bbox_preds, final_mask) # L_box loss

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
    def __init__(self):
        super(HungarianMatcher, self).__init__()
        self.cost_class = 1.0
        self.cost_bbox = 1.0
        self.cost_iou = 1.0

    @torch.no_grad()
    def forward(self, labs, lab_preds, bbox, bbox_preds):
        """
        Perform the Hungarian matching between ground truth and predictions.
        
        Args:
            labs (torch.Tensor): Ground truth labels. [bz x N]
            lab_preds (torch.Tensor): Predicted labels. [bz x N x num_classes]
            bbox (torch.Tensor): Ground truth bounding boxes. [bz x N x 4]
            bbox_preds (torch.Tensor): Predicted bounding boxes. [bz x N x 4]
            
        Returns:
            list of tuples: Each tuple contains (ground_truth_index, prediction_index) for a batch item.
        """


        # Force batch dimension
        if labs.dim() == 1:
            labs = labs.unsqueeze(0)
        if bbox.dim() == 1:
            bbox = bbox.unsqueeze(0)
        if lab_preds.dim() == 2:
            lab_preds = lab_preds.unsqueeze(0)
        if bbox_preds.dim() == 2:
            bbox_preds = bbox_preds.unsqueeze(0)

        batch_size = labs.shape[0]
        indices = []

        for b in range(batch_size):
            tgt_labels = labs[b].squeeze(-1)  # [N]
            pred_logits = lab_preds[b]        # [N, num_classes]
            tgt_bbox = bbox[b]                # [N, 4]
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
            C = C.cpu().detach().numpy()

            pred_indices, gt_indices = linear_sum_assignment(C)
            indices.append((gt_indices, pred_indices))

        return indices
    
class DETRLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_class = 1.0
        self.cost_bbox = 1.0
        self.cost_iou = 1.0 
        self.matcher = HungarianMatcher()

    def forward(self, labs, lab_preds, bbox, bbox_preds):
        """
        Compute the DETR loss.
        
        Args:
            labs (torch.Tensor): Ground truth labels. [bz x N]
            lab_preds (torch.Tensor): Predicted labels. [bz x N x num_classes]
            bbox (torch.Tensor): Ground truth bounding boxes. [bz x N x 4]
            bbox_preds (torch.Tensor): Predicted bounding boxes. [bz x N x 4]
            
        Returns:
            torch.Tensor: The computed DETR loss.
        """
        indices = self.matcher(labs, lab_preds, bbox, bbox_preds)
        batch_size = labs.shape[0]

        # Regen a new labs, lab_preds, bbox, bbox_preds based on indices
        new_labs = torch.zeros_like(labs)
        new_lab_preds = torch.zeros_like(lab_preds)
        new_bbox = torch.zeros_like(bbox)
        new_bbox_preds = torch.zeros_like(bbox_preds)
        for b in range(batch_size):
            gt_indices, pred_indices = indices[b]
            new_labs[b, :len(gt_indices)] = labs[b, gt_indices]
            new_lab_preds[b, :len(pred_indices)] = lab_preds[b, pred_indices]
            new_bbox[b, :len(gt_indices)] = bbox[b, gt_indices]
            new_bbox_preds[b, :len(pred_indices)] = bbox_preds[b, pred_indices]
        return l_hung(new_labs, new_lab_preds, new_bbox, new_bbox_preds)