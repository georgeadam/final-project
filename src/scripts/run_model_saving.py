import logging
import os

import hydra
import wandb
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

from settings import ROOT_DIR
from src.callbacks import callbacks
from src.data import data_modules
from src.inferers import inferers
from src.models import models
from src.modules import modules
from src.trackers import trackers
from src.trainers import trainers
from src.utils.hydra import get_wandb_run
from src.utils.wandb import WANDB_API_KEY

os.chdir(ROOT_DIR)
config_path = os.path.join(ROOT_DIR, "configs")
OmegaConf.register_new_resolver("wandb_run", get_wandb_run)


@hydra.main(config_path=config_path, config_name="model_saving")
def main(args: DictConfig):
    logging.info("\n" + OmegaConf.to_yaml(args))
    logging.info("Saving to: {}".format(os.getcwd()))

    cfg = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)

    # Initial training
    wandb.login(key=WANDB_API_KEY)
    wandb_logger = WandbLogger(project="final_project")
    wandb_logger.experiment.config.update(cfg)

    inferer = inferers.create(args.inferer.name, **args.inferer.params)
    prediction_tracker = trackers.create("prediction")
    metric_tracker = trackers.create("metric_multiclass")

    seed_everything(args.experiment.seed)
    data_module = data_modules.create(args.data_module.name, **args.data_module.params)

    initial_fit(args, data_module, inferer, metric_tracker, prediction_tracker, wandb_logger)

    wandb_logger.log_table("predictions", dataframe=prediction_tracker.get_table())
    wandb_logger.log_table("metrics", dataframe=metric_tracker.get_table())


def initial_fit(args, data_module, inferer, metric_tracker, prediction_tracker, wandb_logger):
    model = models.create(args.model.name, data_dimension=data_module.data_dimension,
                          num_classes=data_module.num_classes, **args.model.params)
    module = modules.create(args.module.name, model=model, **args.module.params)
    callbacks_list = [callbacks.create(value.name, **value.params) for key, value in args.callback.items()]
    trainer = trainers.create(args.trainer.name, update_num=0, callbacks=callbacks_list, logger=wandb_logger,
                              **args.trainer.params)
    trainer.fit(module, train_dataloaders=data_module.train_dataloader(0),
                val_dataloaders=data_module.val_dataloader(0))

    prediction_tracker.track(model, data_module, inferer, "eval", 0)
    metric_tracker.track(model, data_module, inferer, "eval", 0)
    wandb_logger.log_metrics({"eval/loss": metric_tracker.get_most_recent("loss"),
                              "eval/acc": metric_tracker.get_most_recent("acc")})


if __name__ == "__main__":
    main()
