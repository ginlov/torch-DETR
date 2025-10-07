import torch
import cv2
import numpy as np

from typing import List, Dict
from src.data.transforms import UnNormalize

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
    BBoxes are expected to be in normalized [x1, y1, x2, y2] format.

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
    unnormalize = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    imgs_np = imgs.cpu()
    masks_np = masks.unsqueeze(1).cpu().numpy()

    for idx, image_id in enumerate(image_ids):
        img = imgs_np[idx]

        # Concatnate gt_bboxes and out_bboxes for unnormalization
        if len(gt_bboxes[idx]) > 0 and len(out_bboxes[idx]) > 0:
            all_bboxes = torch.cat((gt_bboxes[idx], out_bboxes[idx]), dim=0)
        img, all_bboxes = unnormalize(img, {'boxes': all_bboxes})
        all_bboxes = all_bboxes['boxes']

        # Split back to gt_bboxes and out_bboxes
        if len(gt_bboxes[idx]) > 0 and len(out_bboxes[idx]) > 0:
            gt_bboxes[idx] = all_bboxes[:len(gt_bboxes[idx])]
            out_bboxes[idx] = all_bboxes[len(gt_bboxes[idx]):]
        elif len(gt_bboxes[idx]) > 0:
            gt_bboxes[idx] = all_bboxes
        elif len(out_bboxes[idx]) > 0:
            out_bboxes[idx] = all_bboxes

        img = img.permute(1, 2, 0).numpy()
        mask = masks_np[idx][0]
        # If normalized, scale to [0,255]
        if img.max() <= 1.0:
            img = img * 255.0
        img = img.astype(np.uint8)
        img[mask == 1] = 0
        img = np.ascontiguousarray(img)

        # Draw ground truth boxes (green)
        for lab, bbox in zip(gt_labs[idx], gt_bboxes[idx]):
            xyxy = (
                    bbox.cpu()
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
                    bbox.cpu()
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

