import torch

from typing import Any
from cvrunner.runner import TrainRunner
from cvrunner.experiment import BaseExperiment
from cvrunner.utils.logger import get_cv_logger

from src.metrics import AP

logger = get_cv_logger()

class DETRRunner(TrainRunner):
    def __init__(
            self,
            experiment: type[BaseExperiment]
            ):
        super().__init__(experiment=experiment)
        logger.info(f'Model config: {self.experiment.detr_config}')
        self.valid_outputs = {
            "labels": [],
            "bboxes": [],
            "pred_logits": [],
            "pred_bboxes": []
        }

    def run(self):
        logger.info(f'Starting training for {self.experiment.num_epochs} epochs...')
        num_epoch = self.experiment.num_epochs
        val_freq = self.experiment.val_freq
        for epoch in range(num_epoch):
            logger.info(f'Starting epoch {epoch+1}/{num_epoch}...')
            self.train_epoch_start()
            self.train_epoch()
            self.train_epoch_end()
            logger.info(f'Finished epoch {epoch+1}/{num_epoch}.')

            if epoch % val_freq == 0:
                logger.info(f'Starting validation for epoch {epoch+1}/{num_epoch}...')
                self.val_epoch_start()
                self.val_epoch()
                self.val_epoch_end()
                logger.info(f'Finished validation for epoch {epoch+1}/{num_epoch}.')

    def train_epoch(self):
        num_step = len(self.train_dataloader)
        for i, data in enumerate(self.train_dataloader):
            logger.info(f'Training step {i+1}/{num_step}...')
            self.train_step(data)

    def val_epoch_start(self):
        super().val_epoch_start()
        self.valid_outputs = {k: [] for k in self.valid_outputs}

    def val_epoch(self):
        for data in self.val_dataloader:
            self.val_step(data)

    def val_step(self, data_batch: Any):
        with torch.no_grad():
            outputs = self.experiment.val_step(
                model=self.model,
                data_batch=data_batch,
                loss_function=self.loss_function,
                device=self.device
            )
            logger.log_metrics({k: v for k, v in outputs.items() if 'val/' in k}, local_step=self.step)
            self.valid_outputs['labels'].append(outputs['labels'])
            self.valid_outputs['bboxes'].append(outputs['bboxes'])
            self.valid_outputs['pred_logits'].append(outputs['pred_logits'])
            self.valid_outputs['pred_bboxes'].append(outputs['pred_bboxes'])

    def val_epoch_end(self):
        self.valid_outputs = {k: torch.concat(v, dim=0) for k, v in self.valid_outputs.items()}
        ap_50 = AP(
            labels=self.valid_outputs['labels'],
            bboxes=self.valid_outputs['bboxes'],
            pred_logits=self.valid_outputs['pred_logits'],
            pred_bboxes=self.valid_outputs['pred_bboxes'],
            iou_threshold=0.5,
            num_classes=self.model.num_classes
        )
        ap_75 = AP(
            labels=self.valid_outputs['labels'],
            bboxes=self.valid_outputs['bboxes'],
            pred_logits=self.valid_outputs['pred_logits'],
            pred_bboxes=self.valid_outputs['pred_bboxes'],
            iou_threshold=0.75,
            num_classes=self.model.num_classes
        )
        logger.log_metrics({'val/AP_50': ap_50, 'val/AP_75': ap_75}, local_step=self.step)
        self.valid_outputs = {k: [] for k in self.valid_outputs}