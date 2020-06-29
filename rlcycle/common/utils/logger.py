from datetime import datetime

from omegaconf import DictConfig, OmegaConf
import wandb


class Logger:
    """Logger for logging experiments

    Attributes:
        experiment_cfg (DictConfig): every config (hyper_params, model, etc) merged into one

    """

    def __init__(self, experiment_cfg: DictConfig):
        self.experiment_cfg = experiment_cfg

        self._initialize_wandb()

    def _initialize_wandb(self):
        """Initialize WandB logging."""
        time_info = datetime.now()
        timestamp = f"{time_info.year}-{time_info.month}-{time_info.day}"
        wandb.init(
            project=f"RLcycle-{self.experiment_cfg.experiment_info.env.name}",
            name=f"{self.experiment_cfg.experiment_info.experiment_name}/{timestamp}",
        )
        wandb.config.update(OmegaConf.to_container(self.experiment_cfg))

    def write_log(self, log_dict: dict, step: int = None):
        """Write to WandB log"""
        wandb.log(log_dict, step=step)
