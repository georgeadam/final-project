import numpy as np

from .batch_fetchers import StaticBatchFetcher
from .creation import feeders
from .input_updaters import StaticInputUpdater
from .label_updaters import StaticLabelUpdater
from .static import Static


class CumulativeStatic(Static):
    def __init__(self, splitted_data, val_percentage, num_updates, random_state):
        super().__init__(splitted_data, val_percentage, num_updates, random_state)

        self._input_updater = StaticInputUpdater(num_updates)
        self._label_updater = StaticLabelUpdater(num_updates)
        self._batch_fetcher = StaticBatchFetcher(num_updates)

    def _get_all_data_for_split(self, update_num):
        x_train, y_train, indices_train = self.x_train, self.y_train, self.indices_train

        if update_num == 0:
            return x_train, y_train, indices_train

        samples_per_update = int(len(self.x_update) / self.num_updates)

        x_update = self.x_update[: update_num * samples_per_update]
        y_update = self.y_update[: update_num * samples_per_update]
        indices_update = self.indices_update[: update_num * samples_per_update]

        x = np.concatenate([x_train, x_update])
        y = np.concatenate([y_train, y_update])
        indices = np.concatenate([indices_train, indices_update])

        return x, y, indices


feeders.register_builder("cumulative_static", CumulativeStatic)
