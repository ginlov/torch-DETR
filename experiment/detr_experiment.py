import torch

from typing import Tuple, Type
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import _LRScheduler, LinearLR
from abc import ABC

from cvrunner.experiment import BaseExperiment, DataBatch, MetricType
from cvrunner.runner import BaseRunner
from cvrunner.utils.logger import get_cv_logger

from experiment.detr_config import DETRConfig
from src.detr.detr import build_detr
from src.data.dataset import collate_fn
from src.losses.loss import DETRLoss
from src.utils import pad_targets
from runner.detr_runner import DETRRunner

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
        return 'resnet50'
    
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
        return ''
    
    @property
    def batch_size(self) -> int:
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
            dropout=self.detr_config.dropout
        )

    def build_dataloader(self, partition: str) -> DataLoader:
        dataset = self.build_dataset(partition=partition)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True if partition == 'train' else False,
            num_workers=10,
            collate_fn=collate_fn
        )

    def build_optimizer_scheduler(self, model: torch.nn.Module) -> Tuple[Optimizer, _LRScheduler]:
        # TODO: separate parameters to different groups as in DETR paper
        optimizer = AdamW(model.parameters(), weight_decay=self.weight_decay)
        # TODO: change the schuduler to the exact config in DETR paper
        lr_scheduler = LinearLR(optimizer)
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
            device: torch.device
            ) -> MetricType:
        # TODO: correct this logic
        images = data_batch['images'].to(device)
        masks = data_batch['masks'].to(device)
        label, boxes = pad_targets(data_batch['targets'], self.num_queries)
        output = model(images, masks)
        loss = loss_function(label.to(device), output[0], boxes.to(device), output[1])
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        return {'loss': loss.item()}

    def train_epoch_end(self):
        pass

    def val_epoch_start(self):
        pass

    def val_step(
            self,
            model: torch.nn.Module,
            data_batch: DataBatch,
            loss_function: torch.nn.Module,
            criterion: torch.nn.Module,
            device: torch.device
            ) -> None:
        pass

    def val_epoch_end(self):
        pass
