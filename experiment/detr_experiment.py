from abc import ABC
from typing import Tuple, Type, Dict, Any

import torch
from cvrunner.experiment import BaseExperiment, DataBatch, MetricType
from cvrunner.runner import BaseRunner
from cvrunner.utils.logger import get_cv_logger
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ConstantLR, _LRScheduler
from torch.utils.data import DataLoader

from experiment.detr_config import DETRConfig
from runner.detr_runner import DETRRunner
from src.data.dataset import collate_fn
from src.data.transforms import box_to_xy
from src.data.utils import visualize_mask, visualize_output
from src.detr.detr import build_detr
from src.losses.loss import DETRLoss

logger = get_cv_logger()


class DETRExperiment(BaseExperiment, ABC):
    """
    The experiment should be stateless
    """

    def __init__(
        self,
    ) -> None:
        pass

    def runner_cls(self) -> Type[BaseRunner]:
        return DETRRunner

    @property
    def sanity_check(self) -> bool:
        return False

    @property
    def wandb_project(self) -> str:
        return "DETR"

    @property
    def backbone_name(self) -> str:
        return "resnet50"

    @property
    def num_classes(self) -> int:
        return 0

    @property
    def num_queries(self) -> int:
        return 0

    @property
    def detr_config(self) -> DETRConfig:
        return DETRConfig()

    @property
    def data_folder(self) -> str:
        return ""

    @property
    def batch_size(self) -> int:
        return 2

    @property
    def val_freq(self) -> int:
        return 2

    @property
    def weight_decay(self) -> float:
        return 10e-4

    @property
    def num_epochs(self) -> int:
        return 1

    def build_model(self) -> torch.nn.Module:
        return build_detr(
            backbone_name=self.backbone_name,
            num_queries=self.num_queries,
            num_classes=self.num_classes,
            d_model=self.detr_config.d_model,
            nhead=self.detr_config.nhead,
            num_encoder_layers=self.detr_config.num_encoder_layers,
            num_decoder_layers=self.detr_config.num_decoder_layers,
            dim_feedforward=self.detr_config.dim_feedforward,
            dropout=self.detr_config.dropout,
        )

    def build_dataloader(self, partition: str) -> DataLoader:
        dataset = self.build_dataset(partition=partition)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=partition == "train",
            num_workers=10,
            collate_fn=collate_fn,
        )

    def build_optimizer_scheduler(
        self, model: torch.nn.Module
    ) -> Tuple[Optimizer, _LRScheduler]:
        backbone_params = list(
            p for _, p in model.backbone.named_parameters() if p.requires_grad
        )
        other_params = list(
            p
            for n, p in model.named_parameters()
            if not n.startswith("backbone.") and p.requires_grad
        )

        optimizer = AdamW(
            [
                {"params": backbone_params, "lr": 0.0},
                {"params": other_params, "lr": 1e-4},
            ],
            weight_decay=self.weight_decay,
        )
        # TODO: change the schuduler to the exact config in DETR paper
        lr_scheduler = ConstantLR(optimizer)
        return (optimizer, lr_scheduler)

    def build_loss_function(self) -> DETRLoss:
        # TODO: correct this loss function
        return DETRLoss()

    def save_checkpoint(self) -> None:
        pass

    def load_checkpoint(self) -> None:
        pass

    def train_epoch_start(self) -> None:
        pass

    def train_step(
        self,
        model: torch.nn.Module,
        data_batch: DataBatch,
        loss_function: DETRLoss,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler,
        device: torch.device,
    ) -> MetricType:
        # TODO: correct this logic
        optimizer.zero_grad()
        images = data_batch["images"].to(device)
        masks = data_batch["masks"].to(device)
        # label, boxes = pad_targets(data_batch['targets'], self.num_queries)
        output = model(images, masks)
        label = [item.to(device) for item in data_batch["targets"]["labels"]]
        boxes = [item.to(device) for item in data_batch["targets"]["boxes"]]
        loss, _ = loss_function(label, output[0], boxes, box_to_xy(output[1]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        lr_scheduler.step()
        return {"loss": loss.item()}

    def train_epoch_end(self):
        pass

    def val_epoch_start(self):
        pass

    def val_step(
        self,
        model: torch.nn.Module,
        data_batch: DataBatch,
        loss_function: torch.nn.Module,
        criterion: torch.nn.Module = None,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, Any]:
        output_dict = {}
        images = data_batch["images"].to(device)
        masks = data_batch["masks"].to(device)
        label = [item.to(device) for item in data_batch["targets"]["labels"]]
        boxes = [item.to(device) for item in data_batch["targets"]["boxes"]]
        output = model(images, masks)
        loss, prediction_output = loss_function(label, output[0], boxes, box_to_xy(output[1]))
        output_dict["val/val_loss"] = loss.item()
        output_dict["labels"] = label
        output_dict["bboxes"] = boxes
        output_dict["pred_logits"] = output[0]
        output_dict["pred_bboxes"] = output[1]
        if criterion:
            metrics = criterion(output, data_batch["targets"])
            metrics = {f"val/{k}": v for k, v in metrics.items()}
            output_dict.update(metrics)

        ## Visualization
        image_ids = data_batch["targets"]["image_id"]

        ## The visualized boxes should be the boxes after Hungarian matching
        pred_boxes = prediction_output["pred_boxes"]
        pred_labels = prediction_output["pred_labels"]
        gt_boxes = prediction_output["gt_boxes"]
        gt_labels = prediction_output["gt_labels"]
        output_mask = prediction_output["masks"]

        ## Transform to list of valid boxes/lables
        gt_boxes_list = [
            gt_boxes[i][output_mask[i]].cpu() for i in range(gt_boxes.size(0))
        ]
        pred_boxes_list = [
            pred_boxes[i][output_mask[i]].cpu() for i in range(pred_boxes.size(0))
        ]
        gt_labels_list = [
            gt_labels[i][output_mask[i]].cpu() for i in range(gt_labels.size(0))
        ]
        pred_labels_list = [
            pred_labels[i][output_mask[i]].cpu() for i in range(pred_labels.size(0))
        ]

        output_images = visualize_output(
            imgs=images,
            masks=masks,
            image_ids=image_ids,
            out_labs=pred_labels_list,
            out_bboxes=pred_boxes_list,
            gt_labs=gt_labels_list,
            gt_bboxes=gt_boxes_list,
        )

        output_masks = visualize_mask(masks, image_ids)
        output_dict.update({"visualization": output_images, "mask_visualization": output_masks})
        return output_dict

    def val_epoch_end(self):
        pass
