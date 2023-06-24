import copy
import logging
import os

import hydra
import numpy as np
import pandas as pd
import wandb
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger

from settings import ROOT_DIR
from src.data import data_modules
from src.inferers import Embedding
from src.utils.embedding import feature_space_linear_cka, frobenius_similarity, r2_similarity
from src.utils.hydra import get_wandb_run
from src.utils.load import find_last_checkpoint_for_update, get_checkpoints, load_model
from src.utils.subgroup import generate_subgroup_indices
from src.utils.wandb import WANDB_API_KEY

os.chdir(ROOT_DIR)
config_path = os.path.join(ROOT_DIR, "configs")
OmegaConf.register_new_resolver("wandb_run", get_wandb_run)


@hydra.main(config_path=config_path, config_name="embedding_similarity")
def main(args: DictConfig):
    logging.info("\n" + OmegaConf.to_yaml(args))
    logging.info("Saving to: {}".format(os.getcwd()))

    cfg = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)

    wandb.login(key=WANDB_API_KEY)
    wandb_logger = WandbLogger(project="final_project")
    wandb_logger.experiment.config.update(cfg)

    counts = pd.read_csv(os.path.join(ROOT_DIR, args.counts_path))

    clean_run = get_clean_run(**args.params)
    clean_config = get_config(clean_run)
    data_module = get_data_module(clean_config)

    # Compute baseline metrics for the noisy model after updates
    noisy_run = get_noisy_run(**args.params)
    noisy_config = get_config(noisy_run)

    matched_embedding_similarities = get_all_matched_embedding_similarities(clean_config, clean_run,
                                                                            noisy_config, noisy_run,
                                                                            data_module, counts)

    initial_embedding_similarities = get_all_initial_embedding_similarities(clean_config, clean_run,
                                                                            noisy_config, noisy_run,
                                                                            data_module, counts)

    matched_embedding_similarities.to_csv("matched_embedding_similarities.csv")
    initial_embedding_similarities.to_csv("initial_embedding_similarities.csv")


def get_all_matched_embedding_similarities(clean_config, clean_run, noisy_config, noisy_run, data_module, counts):
    all_similarities = []

    for update_num in range(data_module.data_feeder.num_updates):
        update_similarity = get_embedding_similarities_for_update(update_num, clean_config, clean_run,
                                                                  update_num, noisy_config, noisy_run,
                                                                  data_module, counts)
        update_similarity["update_num"] = update_num
        all_similarities.append(update_similarity)

    return pd.concat(all_similarities)


def get_all_initial_embedding_similarities(clean_config, clean_run, noisy_config, noisy_run, data_module, counts):
    all_similarities = []

    for update_num in range(data_module.data_feeder.num_updates):
        update_similarity = get_embedding_similarities_for_update(0, clean_config, clean_run,
                                                                  update_num, noisy_config, noisy_run,
                                                                  data_module, counts)
        update_similarity["update_num"] = update_num
        all_similarities.append(update_similarity)

    return pd.concat(all_similarities)


def get_embedding_similarities_for_update(clean_update_num, clean_config, clean_run,
                                          noisy_update_num, noisy_config, noisy_run,
                                          data_module, counts):
    similarity_fns = {"r2": r2_similarity, "cka_linear": feature_space_linear_cka, "frobenius": frobenius_similarity}

    inferer = Embedding()

    clean_model = get_model(clean_update_num, clean_config, clean_run, data_module)
    clean_embeddings = inferer.make_predictions(clean_model, data_module.eval_dataloader(0))

    noisy_model = get_model(noisy_update_num, noisy_config, noisy_run, data_module)
    noisy_embeddings = inferer.make_predictions(noisy_model, data_module.eval_dataloader(0))

    subgroup_similarities = []

    for similarity_fn_name, similarity_fn in similarity_fns.items():
        subgroup_similarity = compute_subgroup_similarity(clean_embeddings, noisy_embeddings, counts,
                                                          data_module.data_feeder.indices_test, similarity_fn)
        subgroup_similarity["method"] = similarity_fn_name
        subgroup_similarities.append(subgroup_similarity)

    return pd.concat(subgroup_similarities)


def compute_subgroup_similarity(clean_embeddings, noisy_embeddings, counts, indices, similarity_fn):
    similarity = {"subgroup": [], "similarity": []}
    subgroup_indices = generate_subgroup_indices(counts)

    for subgroup_name, subgroup_index in subgroup_indices:
        relevant_indices =  np.isin(indices, subgroup_index)
        similarity["subgroup"].append(subgroup_name)
        similarity["similarity"].append(similarity_fn(clean_embeddings[relevant_indices],
                                                      noisy_embeddings[relevant_indices]))

    similarity = pd.DataFrame(similarity)

    return similarity


def get_data_module(config):
    return data_modules.create(config.data_module.name, **config.data_module.params)


def get_config(run):
    return OmegaConf.create(run.config)


def get_clean_run(**params):
    params = copy.deepcopy(params)
    params["label_corruptor"] = "clean"
    params["seed"] = params["clean_seed"]

    del params["clean_seed"]
    del params["noisy_seed"]

    run = get_run(**params)

    return run


def get_noisy_run(**params):
    params = copy.deepcopy(params)
    params["seed"] = params["noisy_seed"]

    del params["clean_seed"]
    del params["noisy_seed"]

    run = get_run(**params)

    return run


def get_model(update_num, config, run, data_module):
    checkpoints_dir = os.path.join(ROOT_DIR, "results/{}/{}/updates_with_checkpoints/{}/checkpoints".format(
        config.data_module.name, config.model.name, run.id))
    checkpoints = get_checkpoints(checkpoints_dir)

    checkpoint = find_last_checkpoint_for_update(checkpoints, update_num)
    model = load_model(checkpoints_dir, checkpoint, config, data_module)

    return model


def get_run(batch_size, data_module, early_stopping, feeder, label_corruptor, max_steps, max_epochs, model,
            num_updates, sample_limit, seed, task, warm_start):
    api = wandb.Api(timeout=6000, api_key=WANDB_API_KEY)
    runs = api.runs('georgeadam/final_project', {"$and": [
        {
            "config.original_callback.early_stopping.params.patience": early_stopping,
            "config.update_callback.early_stopping.params.patience": early_stopping,
            "config.experiment_name.value": "updates_with_checkpoints",
            "config.experiment.seed": seed,
            "config.label_corruptor.name": label_corruptor,
            "config.label_corruptor.params.sample_limit": {"$in": [None, sample_limit, ""]},
            "config.model.name": model,
            "config.model.params.warm_start": warm_start,
            "config.data_module.name": data_module,
            "config.data_module.params.batch_size": batch_size,
            "config.data_module.params.feeder_args.params.num_updates": num_updates,
            "config.data_module.params.feeder_args.name": feeder,
            "config.data_module.task": {"$in": [None, task, ""]},
            "config.update_trainer.params.enable_checkpointing": True,
            "config.update_trainer.params.max_epochs": max_epochs,
            "config.update_trainer.params.max_steps": max_steps,
         }]})

    runs_processed = 0

    for run in runs:
        runs_processed += 1

        if runs_processed > 1:
            print("Found more than one unique run!")

    return run



if __name__ == "__main__":
    main()
