import pandas as pd

from .creation import trackers
from .tracker import TrackerInterface


class Prediction(TrackerInterface):
    def __init__(self):
        self._predictions = {"y": [], "prob": [], "pred": [], "update_num": [], "partition": [], "sample_idx": []}

    def track(self, probs, preds, y, partition, update_num):
        self._predictions["prob"] += list(probs)
        self._predictions["pred"] += list(preds)
        self._predictions["y"] += list(y)
        self._predictions["partition"] += [partition] * len(y)
        self._predictions["update_num"] += [update_num] * len(y)
        self._predictions["sample_idx"] += list(range(len(y)))

    def get_table(self):
        return pd.DataFrame(self._predictions)


trackers.register_builder("prediction", Prediction)
