import torch

from typing import Tuple, Type
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import _LRScheduler, LinearLR

from cvrunner.experiment import BaseExperiment, DataBatch
from cvrunner.runner import BaseRunner

from experiment.detr_config import DETRConfig
from src.detr.detr import build_detr
from src.data.dataset import CPPE5Dataset, collate_fn
from src.losses.loss import DETRLoss
from runner.detr_runner import DETRRunner

class DETRExperiment(BaseExperiment):
    """
    The experiment should be stateless
    """
    def __init__(
            self,
            ) -> None:
        pass

    @property
    def runner_cls(self) -> Type[BaseRunner]:
        return DETRRunner

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
        return 'assets/CPPE-5'
    
    @property
    def batch_size(self) -> int:
        return 128
    
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
    
    def build_dataset(self, partition: str) -> Dataset:
        return CPPE5Dataset(
            folder_path=self.data_folder,
            partition=partition
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
            lr_scheduler: _LRScheduler
            ) -> None:
        # TODO: correct this logic
        output = model(data_batch)
        loss = loss_function(output, data_batch['target'])
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    def train_epoch_end(self):
        pass

    def val_epoch_start(self):
        pass

    def val_step(self,
        model: torch.nn.Module,
        data_batch: DataBatch,
        loss_function: torch.nn.Module,
        criterion: torch.nn.Module) -> None:
        pass

    def val_epoch_end(self):
        pass
