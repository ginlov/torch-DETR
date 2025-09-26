# This file contains custom data transformation classes for image preprocessing and augmentation.
# This differs from torchvision.transforms as it also handles bounding box transformations.
# Each transform class implements a __call__ method that takes in an image and an optional target

import torch
import numpy as np
import random

from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F
from abc import ABC, abstractmethod

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

def xy_to_box(bbox_xy):
    """
    Convert bounding boxes from [x1, y1, x2, y2] format to [cx, cy, w, h] format.
    
    Args:
        bbox_xy (torch.Tensor): Bounding boxes in [x1, y1, x2, y2] format. Shape: [bz, N, 4]
        
    Returns:
        torch.Tensor: Bounding boxes in [cx, cy, w, h] format. Shape: [bz, N, 4]
    """
    # Handle both [bz, N, 4] and [N, 4] shapes
    squeeze = False
    if bbox_xy.dim() == 2:
        bbox_xy = bbox_xy.unsqueeze(0)
        squeeze = True
    bbox = torch.zeros_like(bbox_xy)
    bbox[..., 0] = (bbox_xy[..., 0] + bbox_xy[..., 2]) / 2  # cx
    bbox[..., 1] = (bbox_xy[..., 1] + bbox_xy[..., 3]) / 2  # cy
    bbox[..., 2] = bbox_xy[..., 2] - bbox_xy[..., 0]        # w
    bbox[..., 3] = bbox_xy[..., 3] - bbox_xy[..., 1]        # h
    if squeeze:
        bbox = bbox.squeeze(0)
    return bbox

def crop(img, target, region):
    """
    Crop the given PIL Image to the specified region.
    
    Args:
        img (PIL.Image): The input image to be cropped.
        target (dict, optional): The target dictionary containing annotations.
        region (tuple): The crop rectangle, as a (left, upper, right, lower)-tuple.
        
    Returns:
        img (PIL.Image): The cropped image.
        target (dict, optional): The adjusted target dictionary.
    """
    img = F.crop(img, *region)
    if target is None or "boxes" not in target:
        return img, target

    boxes = target["boxes"]
    # Adjust boxes
    boxes = boxes - torch.tensor([region[0], region[1], region[0], region[1]], dtype=torch.float32)
    # Clip boxes to be within the image
    boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=region[2] - region[0])
    boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=region[3] - region[1])
    # Remove boxes that are completely outside the crop
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    target["boxes"] = boxes[keep]
    if "labels" in target:
        target["labels"] = target["labels"][keep]

    return img, target

def resize(img, target, size, max_size=None):
    """
    Resize the input PIL Image to the given size.
    
    Args:
        img (PIL.Image): The input image to be resized.
        target (dict, optional): The target dictionary containing annotations.
        size (int): The desired size of the smaller edge after resizing.
        max_size (int, optional): Maximum size of the longer side after resizing.
        
    Returns:
        img (PIL.Image): The resized image.
        target (dict, optional): The adjusted target dictionary.
    """
    w, h = img.size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    if (w <= h and w == size) or (h <= w and h == size):
        return img, target

    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)
    img = F.resize(img, [oh, ow])

    if target is None or "boxes" not in target:
        return img, target

    boxes = target["boxes"]
    boxes = boxes * torch.tensor([ow / w, oh / h, ow / w, oh / h], dtype=torch.float32)
    target["boxes"] = boxes
    return img, target

def hflip(img, target):
    """
    Horizontally flip the given PIL Image.
    
    Args:
        img (PIL.Image): The input image to be flipped.
        target (dict, optional): The target dictionary containing annotations.
        
    Returns:
        img (PIL.Image): The flipped image.
        target (dict, optional): The adjusted target dictionary.
    """
    img = F.hflip(img)
    if target is None or "boxes" not in target:
        return img, target

    w, h = img.size
    boxes = target["boxes"]
    boxes = boxes[:, [2, 1, 0, 3]]  # x1, y1, x2, y2 -> x2, y1, x1, y2
    boxes[:, [0, 2]] = w - boxes[:, [0, 2]]  # x' = w - x
    target["boxes"] = boxes
    return img, target

def box_iou_matrix(box1: torch.Tensor, box2: torch.Tensor):
    """
    Compute the IoU of two sets of boxes. Boxes are expected in [x1, y1, x2, y2] format.
    
    Args:
        box1 (torch.Tensor): Bounding boxes in [x1, y1, x2, y2] format. Shape: [N, 4]
        box2 (torch.Tensor): Bounding boxes in [x1, y1, x2, y2] format. Shape: [M, 4]
        
    Returns:
        torch.Tensor: IoU matrix of shape [N, M]
    """
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M]

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N, M, 2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou

def box_iou(box1: torch.Tensor, box2: torch.Tensor):
    """
    Compute the IoU of two sets of boxes. Boxes are expected in [x1, y1, x2, y2] format.
    
    Args:
        box1 (torch.Tensor): Bounding boxes in [x1, y1, x2, y2] format. Shape: [N, 4]
        box2 (torch.Tensor): Bounding boxes in [x1, y1, x2, y2] format. Shape: [N, 4]
        
    Returns:
        torch.Tensor: IoU for each pair of boxes. Shape: [N]
    """
    assert box1.shape == box2.shape, "box1 and box2 must have the same shape"
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [N]

    lt = torch.max(box1[:, :2], box2[:, :2])  # [N, 2]
    rb = torch.min(box1[:, 2:], box2[:, 2:])  # [N, 2]

    wh = (rb - lt).clamp(min=0)  # [N, 2]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    union = area1 + area2 - inter

    iou = inter / union
    return iou

class BaseTransform(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, img, target):
        """
        Args:
            img (torch.Tensor): The input image tensor.
            target (dict, optional): The target dictionary containing annotations. Typical format is:
                {
                    "boxes": torch.Tensor of shape [N, 4] in [cx, cy, w, h] format,
                    "labels": torch.Tensor of shape [N] with class labels,
                    ...
                }
        
        Returns:
            img (torch.Tensor): The image after transformation.
            target (dict, optional): The target dictionary which was changed accordingly.
        """
        pass

class ToTensor(BaseTransform):
    """
    Convert a PIL image to a PyTorch tensor and scale pixel values to [0, 1].
    """
    def __init__(self):
        super(ToTensor, self).__init__()

    def __call__(self, img, target):
        img = F.to_tensor(img)
        return img, target
    
class Compose(BaseTransform):
    """
    Composes several transforms together.
    """
    def __init__(self, transforms):
        """
        Args:
            transforms (list): List of transform callables to be applied sequentially.
        """
        super(Compose, self).__init__()
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target
    
class RandomSizeCrop(BaseTransform):
    """
    Crop the given PIL Image to a random size and aspect ratio.
    A crop of random size (min_size, max_size) is made.
    """
    def __init__(self, min_size, max_size):
        """
        Args:
            min_size (int): Minimum size of the crop.
            max_size (int): Maximum size of the crop.
        """
        super(RandomSizeCrop, self).__init__()
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img, target):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)

class RandomSelect:
    """
    Randomly select one of the given transforms to apply.
    """
    def __init__(self, *transforms):
        """
        Args:
            transforms (list): List of transform callables to choose from.
        """
        super(RandomSelect, self).__init__()
        self.transforms = transforms

    def __call__(self, img, target):
        transform = random.choice(self.transforms)
        return transform(img, target)
    
class RandomResize:
    """
    Resize the input PIL Image to a random size from the given list of sizes.
    """
    def __init__(self, sizes, max_size=None):
        """
        Args:
            sizes (list): List of sizes to choose from.
            max_size (int, optional): Maximum size of the longer side after resizing.
        """
        super(RandomResize, self).__init__()
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)

class RandomHorizontalFlip:
    """
    Horizontally flip the given PIL Image randomly with a given probability.
    """
    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): Probability of the image being flipped. Default value is 0.5
        """
        super(RandomHorizontalFlip, self).__init__()
        self.prob = prob

    def __call__(self, img, target):
        if random.random() < self.prob:
            return hflip(img, target)
        return img, target
    
class RandomCrop:
    """
    Crop the given PIL Image to a random location with the given size.
    """
    def __init__(self, size):
        """
        Args:
            size (tuple): Desired output size of the crop (height, width).
        """
        super(RandomCrop, self).__init__()
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)

class Normalize:
    """ 
    Normalize a tensor image with mean and standard deviation.
    """
    def __init__(self, mean, std):
        """
        Args:
            mean (list): Sequence of means for each channel.
            std (list): Sequence of standard deviations for each channel.
        """
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        img = F.normalize(img, mean=self.mean, std=self.std)
        if target is None:
            return img, None
        target = target.copy()
        h, w = img.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = xy_to_box(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return img, target