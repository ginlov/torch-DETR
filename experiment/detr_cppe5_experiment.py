import os

from experiment.detr_experiment import DETRExperiment
from datetime import datetime
from torch.utils.data import Dataset
from pathlib import Path

from cvrunner.utils.logger import get_cv_logger

from src.data.dataset import CPPE5Dataset

logger = get_cv_logger()

class DETRCPPE5Experiment(DETRExperiment):
    @property
    def wandb_runname(self) -> str:
        return 'cppe5' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    def build_dataset(self, partition: str) -> Dataset:
        logger.info(f'Building {partition} CPPE-5 dataset...')
        if not os.path.exists(self.data_folder):
            import kaggle
            parent = Path(self.data_folder).parent
            os.makedirs(parent, exist_ok=True)
            dataset = 'ginlov/cppe-5'
            kaggle.api.dataset_download_files(dataset, path=parent, unzip=True)
            downloaded_path = os.path.join(parent, os.listdir(parent)[0])
            Path.rename(Path(downloaded_path), Path(self.data_folder))
        dataset = CPPE5Dataset(
            folder_path=self.data_folder,
            partition=partition
        )
        logger.info(f'{partition} dataset size: {len(dataset)}')
        return dataset
