# This file contains custom data transformation classes for image preprocessing and augmentation.
# This differs from torchvision.transforms as it also handles bounding box transformations.
# Each transform class implements a __call__ method that takes in an image and an optional target
# Bounding boxes in every transform functions are expected to be in [x1, y1, x2, y2] format.
import random
import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union

import PIL
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def box_to_xy(bbox: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from [x_top_left, y_top_left, w, h] format to [x1, y1, x2, y2] format.
    This function works with both normalized [0, 1] and absolute pixel values.

    Args:
        bbox (torch.Tensor): Bounding boxes in [x_top_left, y_top_left, w, h] format. [bz, N, 4]

    Returns:
        torch.Tensor: Bounding boxes in [x1, y1, x2, y2] format. [bz, N, 4]
    """
    # Handle both [bz, N, 4] and [N, 4] shapes
    squeeze = False
    if bbox.dim() == 2:
        bbox = bbox.unsqueeze(0)
        squeeze = True
    bbox_xy = torch.zeros_like(bbox)
    bbox_xy[..., 0] = bbox[..., 0]  # x1
    bbox_xy[..., 1] = bbox[..., 1]  # y1
    bbox_xy[..., 2] = bbox[..., 0] + bbox[..., 2]  # x2
    bbox_xy[..., 3] = bbox[..., 1] + bbox[..., 3]  # y2
    if squeeze:
        bbox_xy = bbox_xy.squeeze(0)
    return bbox_xy


def xy_to_box(bbox_xy: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from [x1, y1, x2, y2] format to [x_top_left, y_top_left, w, h] format.
    This function works with both normalized [0, 1] and absolute pixel values.

    Args:
        bbox_xy (torch.Tensor): Bounding boxes in [x1, y1, x2, y2] format. Shape: [bz, N, 4]

    Returns:
        torch.Tensor: Bounding boxes in [x_top_left, y_top_left, w, h] format. Shape: [bz, N, 4]
    """
    # Handle both [bz, N, 4] and [N, 4] shapes
    squeeze = False
    if bbox_xy.dim() == 2:
        bbox_xy = bbox_xy.unsqueeze(0)
        squeeze = True
    bbox = torch.zeros_like(bbox_xy)
    bbox[..., 0] = bbox_xy[..., 0]  # x_top_left
    bbox[..., 1] = bbox_xy[..., 1]  # y_top_left
    bbox[..., 2] = bbox_xy[..., 2] - bbox_xy[..., 0]  # w
    bbox[..., 3] = bbox_xy[..., 3] - bbox_xy[..., 1]  # h
    if squeeze:
        bbox = bbox.squeeze(0)
    return bbox


def crop(img, target, region):
    """
    Crop the given PIL Image to the specified region.
    Bounding boxes in the target dictionary need to be
    in the format [x1, y1, x2, y2].
    Boxes are expected to be in absolute pixel values (not normalized).
    region is a tuple (top, left, hight, width).

    Args:
        img (PIL.Image): The input image to be cropped.
        target (dict, optional): The target dictionary containing annotations.
        region (tuple): The crop rectangle, as a (left, upper, right, lower)-tuple.

    Returns:
        img (PIL.Image): The cropped image.
        target (dict, optional): The adjusted target dictionary.
    """
    returned_img = F.crop(img, *region)
    if target is None or "boxes" not in target:
        return returned_img, target

    returned_target = copy.deepcopy(target)
    # Adjust boxes
    returned_target["boxes"] = target["boxes"] - torch.tensor(
        [region[1], region[0], region[1], region[0]], dtype=torch.float32
    )
    # Clip boxes to be within the image
    returned_target["boxes"][:, 0::2] = returned_target["boxes"][:, 0::2]\
    .clamp(min=0, max=region[3])
    returned_target["boxes"][:, 1::2] = returned_target["boxes"][:, 1::2]\
    .clamp(min=0, max=region[2])

    # Remove boxes that are completely outside the crop
    keep = (returned_target["boxes"][:, 2] > returned_target["boxes"][:, 0]) \
        & (returned_target["boxes"][:, 3] > returned_target["boxes"][:, 1])
    returned_target["boxes"] = returned_target["boxes"][keep]
    if "labels" in returned_target:
        returned_target["labels"] = returned_target["labels"][keep]

    return returned_img, returned_target


def resize(img, target, size, max_size=None):
    """
    Resize the input PIL Image to the given size.
    Bounding boxes in the target dictionary need to be
    in the format [x1, y1, x2, y2]. Boxes are expected to be in absolute pixel values (not normalized).

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
    returned_img = F.resize(img, [oh, ow])

    if target is None or "boxes" not in target:
        return returned_img, target

    returned_target = copy.deepcopy(target)
    returned_target["boxes"] = target["boxes"] * torch.tensor([ow / w, oh / h, ow / w, oh / h], dtype=torch.float32)
    return returned_img, returned_target


def hflip(img, target):
    """
    Horizontally flip the given PIL Image.
    Bounding boxes in the target dictionary need to be
    in the format [x1, y1, x2, y2]. Boxes are expected to be in absolute pixel values (not normalized).

    Args:
        img (PIL.Image): The input image to be flipped.
        target (dict, optional): The target dictionary containing annotations.

    Returns:
        img (PIL.Image): The flipped image.
        target (dict, optional): The adjusted target dictionary.
    """
    returned_img = F.hflip(img)
    if target is None or "boxes" not in target:
        return returned_img, target

    w, _ = img.size
    returned_target = copy.deepcopy(target)
    returned_target["boxes"][:, 0] = w - target["boxes"][:, 2]  # x1' = w - x2
    returned_target["boxes"][:, 2] = w - target["boxes"][:, 0]  # x2' = w - x1
    return returned_img, returned_target


def box_iou_matrix(box1: torch.Tensor, box2: torch.Tensor):
    """
    Compute the IoU of two sets of boxes.
    Boxes are expected in [x1, y1, x2, y2] format.
    This function works with both normalized [0, 1] and absolute pixel values.

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


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Compute the IoU of two sets of boxes.
    Boxes are expected in [x1, y1, x2, y2] format.
    This function works with both normalized [0, 1] and absolute pixel values.

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
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, img, target):
        """
        Args:
            img (torch.Tensor): The input image tensor.
            target (dict, optional): The target dictionary containing annotations.
            Typical format is:
                {
                    "boxes": torch.Tensor of shape [N, 4] in [cx, cy, w, h] format,
                    "labels": torch.Tensor of shape [N] with class labels,
                    ...
                }

        Returns:
            img (torch.Tensor): The image after transformation.
            target (dict, optional): The target dictionary which was changed accordingly.
        """


class ToTensor(BaseTransform):
    """
    Convert a PIL image to a PyTorch tensor and scale pixel values to [0, 1].
    """

    def __init__(self):
        super().__init__()

    def __call__(self, img, target):
        """
        Convert a PIL Image to a tensor.
        Bounding boxes in the target dictionary need to be
        in the format [x1, y1, x2, y2].

        Args:
            img (PIL.Image): The input image to be converted.
            target (dict, optional): The target dictionary containing annotations.
        """
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
        super().__init__()
        self.transforms = transforms

    def __call__(self, img, target):
        """
        Apply the composed transforms to the image and target.
        Bounding boxes in the target dictionary need to be
        in the format [x1, y1, x2, y2].

        Args:
            img (PIL.Image or torch.Tensor): The input image to be transformed.
            target (dict, optional): The target dictionary containing annotations.
        """
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
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img, target):
        """
        Randomly crop the image to a size between min_size and max_size.
        Bounding boxes in the target dictionary need to be
        in the format [x1, y1, x2, y2]. Boxes are expected to be in absolute pixel values (not normalized).

        Args:
            img (PIL.Image): The input image to be cropped.
            target (dict, optional): The target dictionary containing annotations.
        """
        max_w = min(img.width, self.max_size)
        max_h = min(img.height, self.max_size)
        if max_w < self.min_size or max_h < self.min_size:
            return img, target
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class RandomSelect(BaseTransform):
    """
    Randomly select one of the given transforms to apply.
    """

    def __init__(self, *transforms):
        """
        Args:
            transforms (list): List of transform callables to choose from.
        """
        super().__init__()
        self.transforms = transforms

    def __call__(self, img, target):
        """
        Randomly select and apply one of the transforms to the image and target.
        Bounding boxes in the target dictionary need to be
        in the format [x1, y1, x2, y2].

        Args:
            img (PIL.Image or torch.Tensor): The input image to be transformed.
            target (dict, optional): The target dictionary containing annotations.
        """
        transform = random.choice(self.transforms)
        return transform(img, target)


class RandomResize(BaseTransform):
    """
    Resize the input PIL Image to a random size from the given list of sizes.
    """

    def __init__(self, sizes, max_size=None) -> None:
        """
        Args:
            sizes (list): List of sizes to choose from.
            max_size (int, optional): Maximum size of the longer side after resizing.
        """
        super().__init__()
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(
        self, img, target
    ) -> Tuple[
        Union[PIL.Image.Image, torch.Tensor], Dict[str, Union[torch.Tensor, Any]]
    ]:
        """
        Resize the image to a random size from the list of sizes.
        Bounding boxes in the target dictionary need to be
        in the format [x1, y1, x2, y2]. Boes are expected to be in absolute pixel values (not normalized).

        Args:
            img (PIL.Image): The input image to be resized.
            target (dict, optional): The target dictionary containing annotations.
            Bounding boxes need to be in [x1, y1, x2, y2] format.
        """
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomHorizontalFlip(BaseTransform):
    """
    Horizontally flip the given PIL Image randomly with a given probability.
    """

    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): Probability of the image being flipped. Default value is 0.5
        """
        super().__init__()
        self.prob = prob

    def __call__(self, img, target):
        """
        Horizontally flip the image with the given probability.
        Bounding boxes in the target dictionary need to be
        in the format [x1, y1, x2, y2]. Boxes are expected to be in absolute pixel values (not normalized).

        Args:
            img (PIL.Image): The input image to be flipped.
            target (dict, optional): The target dictionary containing annotations.
        """
        if random.random() < self.prob:
            return hflip(img, target)
        return img, target


class RandomCrop(BaseTransform):
    """
    Crop the given PIL Image to a random location with the given size.
    """

    def __init__(self, size):
        """
        Args:
            size (tuple): Desired output size of the crop (height, width).
        """
        super().__init__()
        self.size = size

    def __call__(self, img, target):
        """
        Crop the image to a random location with the given size.
        Bounding boxes in the target dictionary need to be
        in the format [x1, y1, x2, y2]. Boxes are expected to be in absolute pixel values (not normalized).

        Args:
            img (PIL.Image): The input image to be cropped.
            target (dict, optional): The target dictionary containing annotations.
        """
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class Normalize(BaseTransform):
    """
    Normalize a tensor image with mean and standard deviation.
    """

    def __init__(self, mean, std):
        """
        Args:
            mean (list): Sequence of means for each channel.
            std (list): Sequence of standard deviations for each channel.
        """
        super().__init__()
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        """
        Normalize the image with the given mean and std.
        Bounding boxes in the target dictionary need to be
        in the format [x1, y1, x2, y2]. Boxes are expected to be in absolute pixel values (not normalized).

        Args:
            img (torch.Tensor): The input image tensor to be normalized.
            target (dict, optional): The target dictionary containing annotations.
        """
        new_img = copy.deepcopy(img)
        new_img = F.normalize(new_img, mean=self.mean, std=self.std)
        if target is None:
            return new_img, target
        new_target = copy.deepcopy(target)
        h, w = img.shape[-2:]
        if "boxes" in new_target:
            boxes = new_target["boxes"]
            # boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            new_target["boxes"] = boxes
        return new_img, new_target


class UnNormalize(BaseTransform):
    """
    Unnormalize a tensor image with mean and standard deviation.
    """

    def __init__(self, mean, std):
        """
        Args:
            mean (list): Sequence of means for each channel.
            std (list): Sequence of standard deviations for each channel.
        """
        super().__init__()
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        """
        Unnormalize the image with the given mean and std.
        Bounding boxes in the target dictionary need to be
        in the format [x1, y1, x2, y2]. Boxes are expected to be normalized to [0, 1].

        Args:
            img (torch.Tensor): The input image tensor to be unnormalized.
            target (dict, optional): The target dictionary containing annotations.
            Bounding boxes need to be in [x1, y1, x2, y2] format.
        Returns:
            img (torch.Tensor): The unnormalized image tensor.
            target (dict, optional): The target dictionary which was changed accordingly.
        """
        new_img = copy.deepcopy(img)
        for t, m, s in zip(new_img, self.mean, self.std):
            t.mul_(s).add_(m)
        if target is None:
            return new_img, None
        new_target = copy.deepcopy(target)
        h, w = img.shape[-2:]
        if "boxes" in new_target:
            boxes = new_target["boxes"]
            # boxes = boxes * torch.tensor([w, h, w, h], dtype=torch.float32)
            new_target["boxes"] = boxes
        return new_img, new_target
