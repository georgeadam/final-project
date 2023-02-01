import os

import numpy as np
import pandas as pd
import yaml
from omegaconf import OmegaConf

from settings import ROOT_DIR
from src.inferers import inferers
from src.trackers import trackers


def compute_metrics_per_subgroup(model, data_module, counts):
    with open(os.path.join(ROOT_DIR, "configs/inferer/standard.yaml"), "r") as f:
        inferer_args = yaml.safe_load(f)
        inferer_args = OmegaConf.create(inferer_args)

    inferer = inferers.create(inferer_args.name, **inferer_args.params)
    prediction_tracker = trackers.create("prediction")
    prediction_tracker.track(model, data_module, inferer, "eval", 0)
    predictions = prediction_tracker.get_table()

    predictions = merge_predictions_and_counts(predictions, counts)
    subgroup_conditions = generate_subgroup_conditions(predictions)

    return compute_metrics_helper(predictions, subgroup_conditions)


def merge_predictions_and_counts(predictions, counts):
    return predictions.merge(counts, on="sample_idx")


def generate_subgroup_conditions(predictions):
    first = np.percentile(predictions["correct"], 33)
    second = np.percentile(predictions["correct"], 66)
    third = np.percentile(predictions["correct"], 100)
    subgroup_conditions = [("first_tercile", predictions["correct"] < first),
                           ("second_tercile", (predictions["correct"] >= first) & (predictions["correct"] < second)),
                           ("third_tercile", (predictions["correct"] >= second) & (predictions["correct"] <= third)),
                           ("all", predictions["correct"] > -1)]

    return subgroup_conditions


def compute_metrics_helper(predictions, subgroup_conditions):
    metrics = []

    for subgroup_condition in subgroup_conditions:
        difficulty = subgroup_condition[0]
        condition = subgroup_condition[1]

        sub_predictions = predictions.loc[condition]
        metric_tracker = trackers.create("metric_multiclass")
        metric_tracker.track_helper(sub_predictions["pred"], sub_predictions["y"], "eval", 0)

        temp_metrics = metric_tracker.get_table()
        temp_metrics["difficulty"] = "{} | n={}".format(difficulty, len(sub_predictions))

        metrics.append(temp_metrics)

    return pd.concat(metrics)
