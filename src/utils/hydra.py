from pytorch_lightning.loggers import WandbLogger

import wandb
from .wandb import WANDB_API_KEY


def get_wandb_run():
    wandb.login(key=WANDB_API_KEY)
    WandbLogger(project="final_project")

    return wandb.run.id
