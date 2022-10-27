import pandas as pd

from src.utils.metric import get_metrics
from .creation import trackers
from .tracker import TrackerInterface


class Metric(TrackerInterface):
    def __init__(self):
        self._metrics = {"fpr": [], "tpr": [], "fnr": [], "tnr": [], "precision": [], "recall": [], "f1": [], "auc": [],
                         "loss": [], "aupr": [], "fp_conf": [], "pos_conf": [], "fp_count": [], "total_samples": [],
                         "fp_prop": [], "acc": [], "youden": [], "update_num": [], "partition": []}

    def track(self, probs, preds, y, partition, update_num):
        metrics = get_metrics(probs, preds, y)

        for key, value in metrics.items():
            if key in self._metrics.keys():
                self._metrics[key].append(value)
            else:
                self._metrics[key] = [value]

        self._metrics["partition"].append(partition)
        self._metrics["update_num"].append(update_num)

    def get_table(self):
        return pd.DataFrame(self._metrics)


trackers.register_builder("metric", Metric)
