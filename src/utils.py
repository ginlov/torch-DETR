import torch
from types import SimpleNamespace

CONSTANTS = SimpleNamespace(
    NO_OBJECT_LABEL=0
)

def box_to_xy(bbox):
    """
    Convert bounding boxes from [cx, cy, w, h] format to [x1, y1, x2, y2] format.
    
    Args:
        bbox (torch.Tensor): Bounding boxes in [cx, cy, w, h] format. Shape: [bz, N, 4]
        
    Returns:
        torch.Tensor: Bounding boxes in [x1, y1, x2, y2] format. Shape: [bz, N, 4]
    """
    # Handle both [bz, N, 4] and [N, 4] shapes
    squeeze = False
    if bbox.dim() == 2:
        bbox = bbox.unsqueeze(0)
        squeeze = True
    bbox_xy = torch.zeros_like(bbox)
    bbox_xy[..., 0] = bbox[..., 0] - bbox[..., 2] / 2  # x1
    bbox_xy[..., 1] = bbox[..., 1] - bbox[..., 3] / 2  # y1
    bbox_xy[..., 2] = bbox[..., 0] + bbox[..., 2] / 2  # x2
    bbox_xy[..., 3] = bbox[..., 1] + bbox[..., 3] / 2  # y2
    if squeeze:
        bbox_xy = bbox_xy.squeeze(0)
    return bbox_xy