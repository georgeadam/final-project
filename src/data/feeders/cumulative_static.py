import numpy as np
from sklearn.model_selection import train_test_split

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

    def _get_train_data(self, update_num):
        x_train, y_train, indices_train = self.x_train, self.y_train, self.indices_train

        x_train, _, y_train, _, indices_train, _ = train_test_split(x_train, y_train, indices_train,
                                                                    test_size=self._val_percentage,
                                                                    random_state=self._random_state)

        if update_num == 0:
            return x_train, y_train, indices_train

        x_update, y_update, indices_update = [], [], []

        for i in range(1, update_num +  1):
            x_temp, y_temp, indices_temp = self.get_current_update_batch(i)
            x_temp, _, y_temp, _, indices_temp, _ = train_test_split(x_temp, y_temp, indices_temp,
                                                                     test_size=self._val_percentage,
                                                                     random_state=self._random_state)

            x_update.append(x_temp)
            y_update.append(y_temp)
            indices_update.append(indices_temp)

        x_update = np.concatenate(x_update)
        y_update = np.concatenate(y_update)
        indices_update = np.concatenate(indices_update)

        x = np.concatenate([x_train, x_update])
        y = np.concatenate([y_train, y_update])
        indices = np.concatenate([indices_train, indices_update])

        return x, y, indices

    def _get_val_data(self, update_num):
        x_train, y_train, indices_train = self.x_train, self.y_train, self.indices_train

        _, x_train, _, y_train, _, indices_train = train_test_split(x_train, y_train, indices_train,
                                                                    test_size=self._val_percentage,
                                                                    random_state=self._random_state)

        if update_num == 0:
            return x_train, y_train, indices_train

        x_update, y_update, indices_update = [], [], []

        for i in range(1, update_num + 1):
            x_temp, y_temp, indices_temp = self.get_current_update_batch(i)
            _, x_temp, _, y_temp, _, indices_temp = train_test_split(x_temp, y_temp, indices_temp,
                                                                     test_size=self._val_percentage,
                                                                     random_state=self._random_state)

            x_update.append(x_temp)
            y_update.append(y_temp)
            indices_update.append(indices_temp)

        x_update = np.concatenate(x_update)
        y_update = np.concatenate(y_update)
        indices_update = np.concatenate(indices_update)

        x = np.concatenate([x_train, x_update])
        y = np.concatenate([y_train, y_update])
        indices = np.concatenate([indices_train, indices_update])

        return x, y, indices


feeders.register_builder("cumulative_static", CumulativeStatic)
