import numpy as np
from sklearn.model_selection import train_test_split

from .creation import feeders
from .static import Static


class CumulativeStatic(Static):
    def __init__(self, splitted_data, val_percentage, num_updates, random_state):
        super().__init__(splitted_data, val_percentage, num_updates, random_state)

    def get_train_data(self, update_num):
        x, y, indices = self._get_all_cumulative_data(update_num)
        x, _, y, _, indices, _ = train_test_split(x, y, indices, test_size=self._val_percentage,
                                                  random_state=self._random_state)

        return x, y, indices

    def get_val_data(self, update_num):
        x, y, indices = self._get_all_cumulative_data(update_num)
        _, x, _, y, _, indices = train_test_split(x, y, indices, test_size=self._val_percentage,
                                                  random_state=self._random_state)

        return x, y, indices

    def _get_all_cumulative_data(self, update_num):
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
