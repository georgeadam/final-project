from sklearn.model_selection import train_test_split

from .creation import feeders
from .static import Static


class CurrentStatic(Static):
    def __init__(self, x, y, indices, n_train, n_update, n_test, val_percentage, num_updates, random_state):
        super().__init__(x, y, indices, n_train, n_update, n_test, val_percentage, num_updates, random_state)

    def get_train_data(self, update_num):
        if update_num == 0:
            x, _, y, _, indices, _ = train_test_split(self.x_train, self.y_train, self.indices_train,
                                                      test_size=self._val_percentage, random_state=self._random_state)
        else:
            x, y, indices = self.get_current_update_batch(update_num)
            x, _, y, _, indices, _ = train_test_split(x, y, indices, test_size=self._val_percentage,
                                                      random_state=self._random_state)

        return x, y, indices

    def get_val_data(self, update_num):
        if update_num == 0:
            _, x, _, y, _, indices = train_test_split(self.x_train, self.y_train, self.indices_train,
                                                      test_size=self._val_percentage, random_state=self._random_state)
        else:
            x, y, indices = self.get_current_update_batch(update_num)
            _, x, _, y, _, indices = train_test_split(x, y, indices, test_size=self._val_percentage,
                                                      random_state=self._random_state)

        return x, y, indices


feeders.register_builder("current_static", CurrentStatic)
