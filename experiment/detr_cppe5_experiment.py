import os
from datetime import datetime
from pathlib import Path
from typing import Type

from cvrunner.utils.logger import get_cv_logger
from torch.utils.data import Dataset

from experiment.detr_config import DETRConfig, DETRCPPEConfigSanityCheck
from experiment.detr_experiment import DETRExperiment
from src.data.dataset import CPPE5Dataset

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
        return 50 if not self.sanity_check else 50

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
