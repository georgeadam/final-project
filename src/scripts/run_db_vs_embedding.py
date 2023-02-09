import copy
import logging
import os

import hydra
import pandas as pd
import wandb
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

from settings import ROOT_DIR
from src.callbacks import callbacks
from src.data import data_modules
from src.modules import modules
from src.trainers import trainers
from src.utils.hydra import get_wandb_run
from src.utils.load import find_last_checkpoint_for_update, get_checkpoints, load_model
from src.utils.subgroup import compute_metrics_per_subgroup

os.chdir(ROOT_DIR)
config_path = os.path.join(ROOT_DIR, "configs")
OmegaConf.register_new_resolver("wandb_run", get_wandb_run)


@hydra.main(config_path=config_path, config_name="db_vs_embedding")
def main(args: DictConfig):
    logging.info("\n" + OmegaConf.to_yaml(args))
    logging.info("Saving to: {}".format(os.getcwd()))

    cfg = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)

    wandb.login(key="604640cf55056fd18bf07355ea2757e21a0c8d17")
    wandb_logger = WandbLogger(project="final_project")
    wandb_logger.experiment.config.update(cfg)

    counts = pd.read_csv(os.path.join(ROOT_DIR, args.counts_path))

    clean_run = get_clean_run(**args.params)
    clean_config = get_config(clean_run)
    data_module = get_data_module(clean_config)

    # Compute baseline metrics for the clean model, before and after updates
    clean_initial_model = get_model("initial", clean_config, clean_run, data_module)
    clean_initial_metrics = compute_metrics_per_subgroup(clean_initial_model, data_module, counts)
    del clean_initial_model

    clean_final_model = get_model("final", clean_config, clean_run, data_module)
    clean_final_metrics = compute_metrics_per_subgroup(clean_final_model, data_module, counts)

    # Compute baseline metrics for the noisy model after updates
    noisy_run = get_run(**args.params)
    noisy_config = get_config(noisy_run)
    noisy_final_model = get_model("final", noisy_config, noisy_run, data_module)
    noisy_final_metrics = compute_metrics_per_subgroup(noisy_final_model, data_module, counts)

    # Sanity check: fine-tune all layers of updated noisy model on clean data
    noisy_fine_tune_model = fine_tune_model(noisy_final_model, noisy_config, data_module,
                                            data_module.num_updates)
    noisy_sanity_check_metrics = compute_metrics_per_subgroup(noisy_fine_tune_model, data_module, counts)

    # Fine tune embedding layers of updated noisy model on clean data
    noisy_final_model = get_model("final", noisy_config, noisy_run, data_module)
    noisy_final_model.freeze_classification_layer()
    noisy_fine_tune_model = fine_tune_model(noisy_final_model, noisy_config, data_module,
                                            data_module.num_updates)
    noisy_embedding_fine_tune_metrics = compute_metrics_per_subgroup(noisy_fine_tune_model, data_module, counts)

    # Fine tune classification layer of updated noisy model on clean data
    noisy_final_model = get_model("final", noisy_config, noisy_run, data_module)
    noisy_final_model.freeze_embedding_layers()
    noisy_fine_tune_model = fine_tune_model(noisy_final_model, noisy_config, data_module,
                                            data_module.num_updates)
    noisy_classification_fine_tune_metrics = compute_metrics_per_subgroup(noisy_fine_tune_model, data_module, counts)

    # Fine tune classification layer of initial clean model on clean data
    del clean_final_model
    del noisy_final_model
    clean_initial_model = get_model("initial", clean_config, clean_run, data_module)
    clean_initial_model.freeze_embedding_layers()
    clean_fine_tune_model = fine_tune_model(clean_initial_model, clean_config, data_module,
                                            data_module.num_updates)
    clean_classification_fine_tune_metrics = compute_metrics_per_subgroup(clean_fine_tune_model, data_module, counts)

    clean_initial_metrics.to_csv("clean_initial_metrics.csv")
    clean_final_metrics.to_csv("clean_final_metrics.csv")
    noisy_final_metrics.to_csv("noisy_final_metrics.csv")
    noisy_sanity_check_metrics.to_csv("noisy_sanity_check_metrics.csv")
    noisy_embedding_fine_tune_metrics.to_csv("noisy_embedding_fine_tune_metrics.csv")
    noisy_classification_fine_tune_metrics.to_csv("noisy_classification_fine_tune_metrics.csv")
    clean_classification_fine_tune_metrics.to_csv("clean_classification_fine_tune_metrics.csv")


def get_data_module(config):
    return data_modules.create(config.data_module.name, **config.data_module.params)


def get_config(run):
    return OmegaConf.create(run.config)


def get_clean_run(**params):
    params = copy.deepcopy(params)
    params["label_corruptor"] = "clean"

    run = get_run(**params)

    return run


def get_model(update_num, config, run, data_module):
    checkpoints_dir = os.path.join(ROOT_DIR, "results/{}/{}/updates_with_checkpoints/{}/checkpoints".format(
        config.data_module.name, config.model.name, run.id))
    checkpoints = get_checkpoints(checkpoints_dir)

    if update_num == "initial":
        update_num = 0
    else:
        update_num = data_module.num_updates

    checkpoint = find_last_checkpoint_for_update(checkpoints, update_num)
    model = load_model(checkpoints_dir, checkpoint, config, data_module)

    return model


def get_run(batch_size, data_module, early_stopping, feeder, label_corruptor, model, num_updates, sample_limit, seed, warm_start):
    api = wandb.Api(timeout=6000, api_key="604640cf55056fd18bf07355ea2757e21a0c8d17")
    runs = api.runs('georgeadam/final_project', {"$and": [
        {"config.original_callback.early_stopping.params.patience": early_stopping,
            "config.update_callback.early_stopping.params.patience": early_stopping,
            "config.experiment_name.value": "updates_with_checkpoints", "config.experiment.seed": seed,
            "config.label_corruptor.name": label_corruptor,
            "config.label_corruptor.params.sample_limit": {"$in": [None, sample_limit]},
            "config.model.params.warm_start": warm_start,
            "config.data_module.name": data_module,
            "config.model.name": model,
            "config.data_module.params.batch_size": batch_size,
            "config.data_module.params.feeder_args.params.num_updates": num_updates,
            "config.data_module.params.feeder_args.name": feeder,
            "config.update_trainer.params.enable_checkpointing": True}]})

    runs_processed = 0

    for run in runs:
        runs_processed += 1

        if runs_processed > 1:
            print("Found more than one unique run!")

    return run


def fine_tune_model(model, config, data_module, update_num):
    wandb.login(key="604640cf55056fd18bf07355ea2757e21a0c8d17")
    wandb_logger = WandbLogger(project="final_project_notebooks")

    args = config
    args.update_trainer.params.enable_progress_bar = False

    seed_everything(0)

    callbacks_list = [callbacks.create("early_stopping", patience=5)]
    trainer = trainers.create(args.update_trainer.name, update_num=update_num, callbacks=callbacks_list,
                              logger=wandb_logger, **args.update_trainer.params)
    data_module.update_transforms(update_num)

    module = modules.create(args.update_module.name, model=model, **args.update_module.params)
    trainer.fit(module, train_dataloaders=data_module.train_dataloader(update_num),
                val_dataloaders=data_module.val_dataloader(update_num))

    return model


if __name__ == "__main__":
    main()
