import pandas as pd

from src.utils.metric import get_metrics_multiclass
from .creation import trackers
from .tracker import TrackerInterface


class MetricMulticlass(TrackerInterface):
    def __init__(self):
        self._metrics = {"total_samples": [], "acc": [], "update_num": [], "partition": [], "loss": []}

    def track(self, model, data_module, inferer, partition, update_num):
        _, preds, y, indices = inferer.make_predictions(model,
                                                        dataloaders=data_module.get_dataloader_by_partition(partition,
                                                            update_num))

        self.track_helper(preds, y, partition, update_num)

    def track_helper(self, preds, y, partition, update_num):
        metrics = get_metrics_multiclass(preds, y)

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


trackers.register_builder("metric_multiclass", MetricMulticlass)
