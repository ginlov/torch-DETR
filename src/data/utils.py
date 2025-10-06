import torch
import cv2
import numpy as np

from typing import List, Dict
from src.data.transforms import box_to_xy

@torch.no_grad()
def visualize_output(
    imgs: torch.Tensor,
    masks: torch.Tensor,
    image_ids: List[int],
    out_labs: List[torch.Tensor],
    out_bboxes: List[torch.Tensor],
    gt_labs: List[torch.Tensor],
    gt_bboxes: List[torch.Tensor]
) -> Dict[int, np.ndarray]:
    """
    Visualize the output of a model along with ground truth labels and bounding boxes.

    Args:
        imgs (torch.Tensor): Batch of input images of shape (B, C, H, W).
        masks (torch.Tensor): Batch of masks of shape (B, H, W).
        image_ids (List[int]): List of image IDs corresponding to each image in the batch.
        out_labs (List[torch.Tensor]): List of tensors containing predicted
        labels for each image. [B, num_gt]
        out_bboxes (List[torch.Tensor]): List of tensors containing predicted
        bounding boxes for each image. [B, num_gt, 4]
        gt_labs (List[torch.Tensor]): List of tensors containing ground truth
        labels for each image. [B, num_gt]
        gt_bboxes (List[torch.Tensor]): List of tensors containing ground truth
        bounding boxes for each image. [B, num_gt, 4]

    Returns:
        Dict[int, np.ndarray]: A dictionary mapping image IDs to their
        corresponding visualized images as numpy arrays.
    """
    visualized = {}

    imgs_np = imgs.cpu().numpy()
    masks_np = masks.unsqueeze(1).cpu().numpy()

    for idx, image_id in enumerate(image_ids):
        img = imgs_np[idx]
        mask = masks_np[idx][0]
        # If normalized, scale to [0,255]
        if img.max() <= 1.0:
            img = img * 255.0
        img = img.astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))
        img[mask == 0] = 0
        img = np.ascontiguousarray(img)

        # Draw ground truth boxes (green)
        for lab, bbox in zip(gt_labs[idx], gt_bboxes[idx]):
            xyxy = (
                    box_to_xy(bbox.cpu())
                    .numpy()
                    .astype(int)
                    .tolist()
            )
            cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(
                img,
                str(int(lab)),
                (xyxy[0], xyxy[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # Draw predicted boxes (red)
        for lab, bbox in zip(out_labs[idx], out_bboxes[idx]):
            xyxy = (
                    box_to_xy(bbox.cpu())
                    .numpy()
                    .astype(int)
                    .tolist()
            )
            cv2.rectangle( # type: ignore
                img,
                (xyxy[0], xyxy[1]),
                (xyxy[2], xyxy[3]),
                (0, 0, 255),
                2
            )
            cv2.putText( # type: ignore
                img,
                str(int(lab)),
                (xyxy[0], xyxy[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        visualized[image_id] = img

    return visualized

