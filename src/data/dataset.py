import json
import os
from abc import ABC, abstractmethod

import torch
from PIL import Image

from src.data import transforms as T
from src.utils import CONSTANTS


class DETRDataset(torch.utils.data.Dataset, ABC):
    @abstractmethod
    def __init__(self):
        pass

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        bboxes = self.boxes[idx]
        labels = self.labels[idx]
        if not isinstance(bboxes, torch.Tensor):
            bboxes = torch.stack(bboxes, dim=0)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int64)
        images = self.images[idx]
        targets = {
            "boxes": bboxes,  # [N, 4]
            "labels": labels,  # [N]
        }
        images, targets = self.transform(images, targets)
        targets["image_id"] = self.image_metadata[idx]["id"]
        return images, targets


class SFCHDDataset(DETRDataset):
    def __init__(self, folder_path, partition: str = "train"):
        super().__init__()
        self._folder_path = folder_path
        pass


class CPPE5Dataset(DETRDataset):
    def __init__(
        self,
        folder_path,
        partition: str = "train",
        transform=None,
        sanity_check: bool = False,
    ):
        """
        Original bboxes are in [x_top_left, y_top_left, w, h] format, need to apply box_to_xy to
        convert to [x1, y1, x2, y2] format.
        """
        super().__init__()
        self._folder_path = folder_path

        assert os.path.exists(
            self._folder_path
        ), f"Folder path {self._folder_path} does not exist."

        ## Get metdata and image paths
        if partition == "train":
            metadata_path = os.path.join(self._folder_path, "annotations", "train.json")
        elif partition == "val":
            metadata_path = os.path.join(self._folder_path, "annotations", "test.json")
        else:
            raise ValueError(
                f"Invalid partition: {partition}. Must be 'train' or 'val'."
            )
        image_folder = os.path.join(self._folder_path, "images")

        ## Checking images
        images_list = os.listdir(image_folder)
        not_image_files = (
            len(
                [
                    f
                    for f in images_list
                    if not f.lower().endswith(CONSTANTS.IMAGE_EXTENSIONS)
                ]
            )
            != 0
        )
        assert not not_image_files, f"Some files in {image_folder} are not images."

        ## Load metdata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        ## Check images in metadata
        self.image_metadata = sorted(metadata["images"], key=lambda x: x["id"])
        image_in_metadata = [item["file_name"] for item in self.image_metadata]
        image_not_exist = (
            len([f for f in image_in_metadata if f not in images_list]) != 0
        )
        assert (
            not image_not_exist
        ), "Some images in metadata do not exist in the image folder."

        ## Intialize normalization transform
        if not transform:
            # Default transform: Used in DETR
            # Code is borrowed from DETR repo. Thanks to Facebook AI Research.
            normalize = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

            scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

            if partition == "train":
                transform = T.Compose(
                    [
                        T.RandomHorizontalFlip(),
                        T.RandomSelect(
                            T.RandomResize(scales, max_size=1333),
                            T.Compose(
                                [
                                    T.RandomResize([400, 500, 600]),
                                    T.RandomSizeCrop(384, 600),
                                    T.RandomResize(scales, max_size=1333),
                                ]
                            ),
                        ),
                        normalize,
                    ]
                )

            if partition == "val":
                transform = T.Compose(
                    [
                        T.RandomResize([800], max_size=1333),
                        normalize,
                    ]
                )

        self.transform = transform

        ## Load images
        ## TODO: Lazy load images
        self.images = []
        for img_path in self.image_metadata:
            img_path = os.path.join(image_folder, img_path["file_name"])
            image = Image.open(img_path).convert("RGB")
            self.images.append(image)

        ## Load categories metadata
        self.categories_metadata = metadata["categories"]

        ## Load boxes and labels
        self.annotation = sorted(metadata["annotations"], key=lambda x: x["image_id"])
        num_images = len(self.image_metadata)
        self.boxes = [[] for _ in range(num_images)]
        self.labels = [[] for _ in range(num_images)]
        for item in self.annotation:
            image_id = item["image_id"]
            bbox = item["bbox"]

            # TODO: Reorder image_id for comprehensiveness
            offset = 1 if partition == "train" else 1001
            self.boxes[image_id - offset].append(
                T.box_to_xy(torch.Tensor(bbox))
            )  # image_id starts from 1
            self.labels[image_id - offset].append(
                item["category_id"]
            )  # image_id starts from 1

        if sanity_check:
            if partition == "train":
                self.images = self.images[:100]
                self.boxes = self.boxes[:100]
                self.labels = self.labels[:100]
            else:
                self.images = self.images[:20]
                self.boxes = self.boxes[:20]
                self.labels = self.labels[:20]


def collate_fn(batch):
    images = [item[0] for item in batch]  # various sizes
    targets = [item[1] for item in batch]  # various number of boxes for each image

    targets = {
        "boxes": [t["boxes"] for t in targets],
        "labels": [t["labels"] for t in targets],
        "image_id": [t["image_id"] for t in targets],
    }

    # Batch images
    max_w = max([img.shape[2] for img in images])
    max_h = max([img.shape[1] for img in images])
    batch_size = len(images)
    batched_imgs = torch.zeros(
        (batch_size, 3, max_h, max_w), dtype=images[0].dtype
    )  # [bz, 3, max_h, max_w]
    masks = torch.ones((batch_size, max_h, max_w), dtype=torch.bool)
    for i in range(batch_size):
        img = images[i]
        batched_imgs[i, :, : img.shape[1], : img.shape[2]] = img
        masks[i, : img.shape[1], : img.shape[2]] = False

    return {"images": batched_imgs, "masks": masks, "targets": targets}
