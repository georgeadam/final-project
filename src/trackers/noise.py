import pandas as pd

from .creation import trackers
from .tracker import TrackerInterface


class Noise(TrackerInterface):
    def __init__(self):
        self._noisy_labels = {"update_num": [], "sample_idx": [], "type": []}

    def track(self, update_num, actual_indices, potential_indices):
        self.track_helper(update_num, actual_indices, potential_indices)

    def track_helper(self, update_num, actual_indices, potential_indices):
        self._noisy_labels["update_num"] += [update_num] * (len(actual_indices) + len(potential_indices))
        self._noisy_labels["sample_idx"] += actual_indices
        self._noisy_labels["sample_idx"] += potential_indices
        self._noisy_labels["type"] += ["actual"] * len(actual_indices)
        self._noisy_labels["type"] += ["potential"] * len(potential_indices)

    def get_table(self):
        return pd.DataFrame(self._noisy_labels)


trackers.register_builder("noise", Noise)