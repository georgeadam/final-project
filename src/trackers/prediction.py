import pandas as pd

from .creation import trackers
from .tracker import TrackerInterface


class Prediction(TrackerInterface):
    def __init__(self):
        self._predictions = {"y": [], "prob": [], "pred": [], "update_num": [], "partition": [], "sample_idx": []}

    def track(self, module, data_module, trainer, partition, update_num):
        probs, preds, y, indices = trainer.make_predictions(module,
                                                            dataloaders=data_module.get_dataloader_by_partition(
                                                                partition, update_num))

        self.track_helper(probs, preds, y, indices, partition, update_num)

    def track_helper(self, probs, preds, y, indices, partition, update_num):
        self._predictions["prob"] += list(probs)
        self._predictions["pred"] += list(preds)
        self._predictions["y"] += list(y)
        self._predictions["partition"] += [partition] * len(y)
        self._predictions["update_num"] += [update_num] * len(y)
        self._predictions["sample_idx"] += list(indices)

    def get_table(self):
        return pd.DataFrame(self._predictions)


trackers.register_builder("prediction", Prediction)
