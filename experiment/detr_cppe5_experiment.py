from experiment.detr_experiment import DETRExperiment
from datetime import datetime


class DETRCPPE5Experiment(DETRExperiment):
    @property
    def wandb_runname(self) -> str:
        return 'cppe5' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
