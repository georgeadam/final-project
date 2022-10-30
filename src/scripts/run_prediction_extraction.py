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
from src.models import models
from src.modules import modules
from src.threshold_selectors import threshold_selectors
from src.trackers import trackers
from src.trainers import trainers

os.chdir(ROOT_DIR)
config_path = os.path.join(ROOT_DIR, "configs")


@hydra.main(config_path=config_path, config_name="prediction_extraction")
def main(args: DictConfig):
    logging.info("\n" + OmegaConf.to_yaml(args))
    logging.info("Saving to: {}".format(os.getcwd()))

    cfg = OmegaConf.to_container(
        args, resolve=True, throw_on_missing=True
    )

    # Initial training
    wandb.login(key="604640cf55056fd18bf07355ea2757e21a0c8d17")
    wandb_logger = WandbLogger(project="final_project")
    wandb_logger.experiment.config.update(cfg)

    prediction_tracker = trackers.create("prediction")
    metric_tracker = trackers.create("metric")

    seed_everything(args.experiment.seed)
    data_module = data_modules.create(args.data_module.name, **args.data_module.params)
    threshold_selector = threshold_selectors.create(args.threshold_selector.name,
                                                    **args.threshold_selector.params)

    callbacks_list = [callbacks.create(value.name, **value.params) for key, value in args.callback.items()]
    initial_fit(args, callbacks_list, data_module, metric_tracker, prediction_tracker,
                threshold_selector, wandb_logger)

    wandb_logger.log_table("predictions", dataframe=prediction_tracker.get_table())
    wandb_logger.log_table("metrics", dataframe=metric_tracker.get_table())


def initial_fit(args, callbacks_list, data_module, metric_tracker, prediction_tracker, threshold_selector,
                wandb_logger):
    model = models.create(args.model.name, data_dimension=data_module.data_dimension, **args.model.params)
    module = modules.create(args.module.name, model=model, **args.module.params)
    trainer = trainers.create(args.trainer.name, callbacks=callbacks_list, logger=wandb_logger,
                              **args.trainer.params)
    trainer.fit(module, train_dataloaders=data_module.train_dataloader(0),
                val_dataloaders=data_module.val_dataloader(0))

    threshold_selector.select_threshold(module, data_module, trainer, 0)
    probs, preds, y, _ = trainer.make_predictions(module, dataloaders=data_module.current_update_batch_dataloader(1))
    prediction_tracker.track(probs, preds, y, "update", 0)
    metric_tracker.track(probs, preds, y, "update", 0)

    probs, preds, y, _ = trainer.make_predictions(module, dataloaders=data_module.eval_dataloader(0))
    metric_tracker.track(probs, preds, y, "eval", 0)
    wandb_logger.log_metrics({"eval/loss": metric_tracker.get_most_recent("loss"),
                              "eval/aupr": metric_tracker.get_most_recent("aupr"),
                              "eval/auc": metric_tracker.get_most_recent("auc")})


if __name__ == "__main__":
    main()
