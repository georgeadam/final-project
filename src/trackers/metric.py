import pandas as pd

from src.utils.metric import get_metrics
from .creation import trackers
from .tracker import TrackerInterface


class Metric(TrackerInterface):
    def __init__(self):
        self._metrics = {"fpr": [], "tpr": [], "fnr": [], "tnr": [], "precision": [], "recall": [], "f1": [], "auc": [],
                         "loss": [], "aupr": [], "fp_conf": [], "pos_conf": [], "fp_count": [], "total_samples": [],
                         "acc": [], "youden": [], "update_num": [], "partition": []}

    def track(self, module, data_module, trainer, partition, update_num):
        probs, preds, y, indices = trainer.make_predictions(module,
                                                            dataloaders=data_module.get_dataloader_by_partition(
                                                                partition, update_num))

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

    def get_most_recent(self, rate):
        return self._metrics[rate][-1]


trackers.register_builder("metric", Metric)
