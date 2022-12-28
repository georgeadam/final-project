import copy
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
from src.label_corruptors import label_corruptors
from src.models import models
from src.modules import modules
from src.threshold_selectors import threshold_selectors
from src.trackers import trackers
from src.trainers import trainers
from src.utils.hydra import get_wandb_run

os.chdir(ROOT_DIR)
config_path = os.path.join(ROOT_DIR, "configs")
OmegaConf.register_new_resolver("wandb_run", get_wandb_run)


@hydra.main(config_path=config_path, config_name="update_pipeline")
def main(args: DictConfig):
    logging.info("\n" + OmegaConf.to_yaml(args))
    logging.info("Saving to: {}".format(os.getcwd()))

    cfg = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)

    # Initial training
    wandb.login(key="604640cf55056fd18bf07355ea2757e21a0c8d17")
    wandb_logger = WandbLogger(project="final_project", prefix="initial")
    wandb_logger.experiment.config.update(cfg)

    inferer = inferers.create(args.inferer.name, **args.inferer.params)
    prediction_tracker = trackers.create("prediction")
    metric_tracker = trackers.create("metric")
    noise_tracker = trackers.create("noise")

    seed_everything(args.experiment.seed)
    data_module = data_modules.create(args.data_module.name, **args.data_module.params)
    threshold_selector = threshold_selectors.create(args.threshold_selector.name, **args.threshold_selector.params)

    model = initial_fit(args, data_module, inferer, metric_tracker,
                        prediction_tracker, threshold_selector, wandb_logger)


    update_model(args, data_module, inferer, metric_tracker, model, noise_tracker,
                 prediction_tracker, threshold_selector)

    wandb_logger = WandbLogger(project="final_project")
    wandb_logger.log_metrics({"eval_final/loss": metric_tracker.get_most_recent("loss"),
                              "eval_final/aupr": metric_tracker.get_most_recent("aupr"),
                              "eval_final/auc": metric_tracker.get_most_recent("auc")})

    if args.experiment.save_predictions:
        wandb_logger.log_table("predictions", dataframe=prediction_tracker.get_table())

    wandb_logger.log_table("metrics", dataframe=metric_tracker.get_table())
    wandb_logger.log_table("noisy_labels", dataframe=noise_tracker.get_table())


def initial_fit(args, data_module, inferer, metric_tracker, prediction_tracker, threshold_selector, wandb_logger):
    model = models.create(args.model.name, data_dimension=data_module.data_dimension, **args.model.params)
    module = modules.create(args.original_module.name, model=model, **args.original_module.params)

    callbacks_list = [callbacks.create(value.name, **value.params) for key, value in args.original_callback.items()]
    trainer = trainers.create(args.original_trainer.name, update_num=0, callbacks=callbacks_list, logger=wandb_logger,
                              **args.original_trainer.params)
    trainer.fit(module, train_dataloaders=data_module.train_dataloader(0),
                val_dataloaders=data_module.val_dataloader(0))

    threshold_selector.select_threshold(model, data_module, inferer, 0)
    prediction_tracker.track(model, data_module, inferer, "eval", 0)
    metric_tracker.track(model, data_module, inferer, "eval", 0)
    wandb_logger.log_metrics({"eval_original/loss": metric_tracker.get_most_recent("loss"),
                              "eval_original/aupr": metric_tracker.get_most_recent("aupr"),
                              "eval_original/auc": metric_tracker.get_most_recent("auc")})
    return model


def update_model(args, data_module, inferer, metric_tracker, model, noise_tracker,
                 prediction_tracker, threshold_selector):
    label_corruptor = label_corruptors.create(args.label_corruptor.name, noise_tracker=noise_tracker,
                                              **args.label_corruptor.params)

    for update_num in range(1, data_module.num_updates + 1):
        original_model = copy.deepcopy(model)
        callbacks_list = [callbacks.create(value.name, **value.params) for key, value in args.update_callback.items()]

        metric_tracker.track(model, data_module, inferer, "update-clean", update_num)
        label_corruptor.corrupt(model, data_module, inferer, update_num)
        metric_tracker.track(model, data_module, inferer, "update-noisy", update_num)
        data_module.update_transforms(update_num)

        if not model.warm_start:
            model = models.create(args.model.name, data_dimension=data_module.data_dimension, **args.model.params)

        wandb_logger = WandbLogger(project="final_project", prefix="update-{}".format(update_num))
        trainer = trainers.create(args.update_trainer.name, update_num=update_num, callbacks=callbacks_list,
                                  logger=wandb_logger, **args.update_trainer.params)
        module = modules.create(args.update_module.name, model=model, original_model=original_model,
                                **args.update_module.params)
        trainer.fit(module, train_dataloaders=data_module.train_dataloader(update_num),
                    val_dataloaders=data_module.val_dataloader(update_num))

        threshold_selector.select_threshold(model, data_module, inferer, update_num)
        prediction_tracker.track(model, data_module, inferer, "eval", update_num)
        metric_tracker.track(model, data_module, inferer, "eval", update_num)


if __name__ == "__main__":
    main()
