import itertools
from typing import TYPE_CHECKING, Any

import torch
from cvrunner.runner import TrainRunner
from cvrunner.utils.logger import get_cv_logger

from src.metrics import AP
from src.utils import cal_grad_norm, cal_param_norm

if TYPE_CHECKING:
    from experiment.detr_experiment import DETRExperiment

logger = get_cv_logger()


class DETRRunner(TrainRunner):
    def __init__(self, experiment: type["DETRExperiment"]):
        super().__init__(experiment=experiment)
        logger.info(f"Model config: {self.experiment.detr_config}")
        self.valid_outputs = {
            "labels": [],
            "bboxes": [],
            "pred_logits": [],
            "pred_bboxes": [],
        }

    def run(self):
        logger.info(f"Starting training for {self.experiment.num_epochs} epochs...")
        num_epoch = self.experiment.num_epochs
        val_freq = self.experiment.val_freq
        for epoch in range(num_epoch):
            logger.info(f"Starting epoch {epoch+1}/{num_epoch}...")
            self.train_epoch_start()
            self.train_epoch()
            self.train_epoch_end()
            logger.info(f"Finished epoch {epoch+1}/{num_epoch}.")

            if epoch % val_freq == 0:
                logger.info(f"Starting validation for epoch {epoch+1}/{num_epoch}...")
                self.val_epoch_start()
                self.val_epoch()
                self.val_epoch_end()
                logger.info(f"Finished validation for epoch {epoch+1}/{num_epoch}.")

    def train_epoch(self):
        num_step = len(self.train_dataloader)
        for i, data in enumerate(self.train_dataloader):
            logger.info(f"Training step {i+1}/{num_step}...")
            self.train_step(data)

    def train_step(self, data_batch: Any):
        super().train_step(data_batch)
        with torch.no_grad():
            param_norm = cal_param_norm(self.model)
            logger.log_metrics(param_norm, local_step=self.step)

            grad_norm = cal_grad_norm(self.model)
            logger.log_metrics(grad_norm, local_step=self.step)

    def val_epoch_start(self):
        super().val_epoch_start()
        self.valid_outputs = {k: [] for k in self.valid_outputs}

    def val_step(self, data: Any):
        with torch.no_grad():
            outputs = self.experiment.val_step(
                model=self.model,
                data_batch=data,
                loss_function=self.loss_function,
                device=self.device,
            )
            self.val_metrics.update({k: v for k, v in outputs.items() if "val/" in k})
            self.valid_outputs["labels"].append(outputs["labels"])
            self.valid_outputs["bboxes"].append(outputs["bboxes"])
            self.valid_outputs["pred_logits"].append(outputs["pred_logits"])
            self.valid_outputs["pred_bboxes"].append(outputs["pred_bboxes"])
            logger.log_images(
                list(outputs["visualization"].keys()),
                list(outputs["visualization"].values()),
                self.step,
            )

    def val_epoch_end(self):
        self.valid_outputs = {
            k: (
                torch.concat(v, dim=0)
                if isinstance(v[0], torch.Tensor)
                else list(itertools.chain.from_iterable(v))
            )
            for k, v in self.valid_outputs.items()
        }
        ap_50 = AP(
            labels=self.valid_outputs["labels"],
            bboxes=self.valid_outputs["bboxes"],
            pred_logits=self.valid_outputs["pred_logits"],
            pred_bboxes=self.valid_outputs["pred_bboxes"],
            iou_threshold=0.5,
            num_classes=self.model.num_classes,
        )
        ap_75 = AP(
            labels=self.valid_outputs["labels"],
            bboxes=self.valid_outputs["bboxes"],
            pred_logits=self.valid_outputs["pred_logits"],
            pred_bboxes=self.valid_outputs["pred_bboxes"],
            iou_threshold=0.75,
            num_classes=self.model.num_classes,
        )
        self.val_metrics.update({"val/AP_50": ap_50, "val/AP_75": ap_75})
        logger.log_metrics(self.val_metrics.summary(), local_step=self.step)
        self.valid_outputs = {k: [] for k in self.valid_outputs}
