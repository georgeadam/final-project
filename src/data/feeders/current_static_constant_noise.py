import numpy as np
from sklearn.model_selection import train_test_split

from .batch_fetchers import StaticBatchFetcher
from .creation import feeders
from .input_updaters import StaticInputUpdater
from .label_updaters import StaticLabelUpdater
from .static import Static


class CurrentStaticConstantNoise(Static):
    def __init__(self, splitted_data, val_percentage, num_updates, random_state):
        super().__init__(splitted_data, val_percentage, num_updates, random_state)

        self._input_updater = StaticInputUpdater(num_updates)
        self._label_updater = StaticLabelUpdater(num_updates)
        self._batch_fetcher = StaticBatchFetcher(num_updates)

        self.noisy_indices = None

    def _get_train_data(self, update_num):
        if update_num == 0:
            x_train, y_train, indices_train = self.x_train, self.y_train, self.indices_train

            x_train, _, y_train, _, indices_train, _ = train_test_split(x_train, y_train, indices_train,
                                                                        test_size=self._val_percentage,
                                                                        random_state=self._random_state)

            return x_train, y_train, indices_train
        else:
            x_update, y_update, indices_update = self.get_current_update_batch(update_num)
            x_update_noisy, y_update_noisy, indices_update_noisy = self.get_current_update_batch(1)
            x_update_noisy = x_update_noisy[self.noisy_indices]
            y_update_noisy = y_update_noisy[self.noisy_indices]
            indices_update_noisy = indices_update_noisy[self.noisy_indices]

            clean_indices = np.setdiff1d(np.arange(len(x_update)), self.noisy_indices)
            x_update = np.concatenate([x_update[clean_indices], x_update_noisy])
            y_update = np.concatenate([y_update[clean_indices], y_update_noisy])
            indices_update = np.concatenate([indices_update[clean_indices], indices_update_noisy])

            x_update, _, y_update, _, indices_update, _ = train_test_split(x_update, y_update, indices_update,
                                                                           test_size=self._val_percentage,
                                                                           random_state=self._random_state)

            return x_update, y_update, indices_update

    def _get_val_data(self, update_num):
        if update_num == 0:
            x_train, y_train, indices_train = self.x_train, self.y_train, self.indices_train

            _, x_val, _, y_val, _, indices_val = train_test_split(x_train, y_train, indices_train,
                                                                  test_size=self._val_percentage,
                                                                  random_state=self._random_state)

            return x_val, y_val, indices_val
        else:
            x_update, y_update, indices_update = self.get_current_update_batch(update_num)
            x_update_noisy, y_update_noisy, indices_update_noisy = self.get_current_update_batch(1)
            x_update_noisy = x_update_noisy[self.noisy_indices]
            y_update_noisy = y_update_noisy[self.noisy_indices]
            indices_update_noisy = indices_update_noisy[self.noisy_indices]

            clean_indices = np.setdiff1d(np.arange(len(x_update)), self.noisy_indices)
            x_update = np.concatenate([x_update[clean_indices], x_update_noisy])
            y_update = np.concatenate([y_update[clean_indices], y_update_noisy])
            indices_update = np.concatenate([indices_update[clean_indices], indices_update_noisy])

            _, x_update, _, y_update, _, indices_update = train_test_split(x_update, y_update, indices_update,
                                                                           test_size=self._val_percentage,
                                                                           random_state=self._random_state)

            return x_update, y_update, indices_update

    def overwrite_current_update_labels(self, new_labels, indices, update_num):
        if self.noisy_indices is None:
            self.noisy_indices = indices
            self._label_updater.overwrite_update_labels(self.y_update, new_labels, indices, update_num)


feeders.register_builder("current_static_constant_noise", CurrentStaticConstantNoise)
