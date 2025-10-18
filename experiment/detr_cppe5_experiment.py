import os
import torch
from datetime import datetime
from pathlib import Path
from typing import Type, Tuple

from cvrunner.utils.logger import get_cv_logger
from torch.utils.data import Dataset
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from experiment.detr_config import DETRConfig, DETRCPPEConfigSanityCheck
from experiment.detr_experiment import DETRExperiment
from src.data.dataset import CPPE5Dataset
from src.utils import get_warmup_cosine_schedule

logger = get_cv_logger()


class DETRCPPE5Experiment(DETRExperiment):
    @property
    def wandb_runname(self) -> str:
        return "cppe5-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    @property
    def sanity_check(self) -> bool:
        return True

    @property
    def num_epochs(self) -> int:
        return 50 if not self.sanity_check else 400

    @property
    def batch_size(self) -> int:
        return 1

    @property
    def num_classes(self) -> int:
        return 5  # CPPE-5 has 5 classes

    @property
    def num_queries(self) -> int:
        return 10

    @property
    def data_folder(self) -> str:
        return "assets/CPPE-5"

    @property
    def detr_config(self) -> Type[DETRConfig]:
        return DETRCPPEConfigSanityCheck()

    # Loss config
    @property
    def label_loss_weight(self) -> float:
        return 2.5

    @property
    def l1_bbox_loss_weight(self) -> float:
        return 0.5

    @property
    def giou_bbox_loss_weight(self) -> float:
        return 1.75


    def build_dataset(self, partition: str) -> Dataset:
        logger.info(f"Building {partition} CPPE-5 dataset...")
        if not os.path.exists(self.data_folder):
            import kaggle

            parent = Path(self.data_folder).parent
            os.makedirs(parent, exist_ok=True)
            dataset = "ginlov/cppe-5"
            kaggle.api.dataset_download_files(dataset, path=parent, unzip=True)
            downloaded_path = os.path.join(parent, os.listdir(parent)[0])
            Path.rename(Path(downloaded_path), Path(self.data_folder))
        if partition == "val" and self.sanity_check:
            dataset = CPPE5Dataset(
                folder_path=self.data_folder,
                partition="train",
                sanity_check=self.sanity_check,
            )
        else:
            dataset = CPPE5Dataset(
                folder_path=self.data_folder,
                partition=partition,
                sanity_check=self.sanity_check,
            )
        logger.info(f"{partition} dataset size: {len(dataset)}")
        return dataset

    def build_optimizer_scheduler(
        self, model: torch.nn.Module, len_dataloader: int = 0
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
                {"params": backbone_params, "lr": 1e-4},
                {"params": other_params, "lr": 1e-4},
            ],
            weight_decay=self.weight_decay,
        )
        # TODO: change the schuduler to the exact config in DETR paper
        lr_scheduler = get_warmup_cosine_schedule(
            optimizer,
            num_warmup_steps=len_dataloader * self.num_epochs // 10 * 6,
            num_training_steps=len_dataloader * self.num_epochs,
        )
        return (optimizer, lr_scheduler)
