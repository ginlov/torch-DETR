from typing import Any
from cvrunner.runner import TrainRunner
from cvrunner.experiment import BaseExperiment
from cvrunner.utils.logger import get_cv_logger

logger = get_cv_logger()

# TODO: fill in the run logic
class DETRRunner(TrainRunner):
    def __init__(
            self,
            experiment: type[BaseExperiment]
            ):
        super().__init__(experiment=experiment)
        logger.info(f'Model config: {self.experiment.detr_config}')

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

    def train_epoch_start(self):
        pass

    def train_epoch(self):
        num_step = len(self.train_dataloader)
        for i, data in enumerate(self.train_dataloader):
            logger.info(f'Training step {i+1}/{num_step}...')
            self.train_step(data)

    def train_epoch_end(self):
        pass

    def val_epoch_start(self):
        pass

    def val_epoch(self):
        for data in self.val_dataloader:
            self.val_step(data)

    def val_epoch_end(self):
        pass

    def checkpoint(self):
        pass