from .batch_fetchers import StaticBatchFetcher
from .creation import feeders
from .input_updaters import StaticInputUpdater
from .label_updaters import StaticLabelUpdater
from .static import Static


class CurrentStatic(Static):
    def __init__(self, splitted_data, val_percentage, num_updates, random_state):
        super().__init__(splitted_data, val_percentage, num_updates, random_state)

        self._input_updater = StaticInputUpdater(num_updates)
        self._label_updater = StaticLabelUpdater(num_updates)
        self._batch_fetcher = StaticBatchFetcher(num_updates)

    def _get_all_data_for_split(self, update_num):
        if update_num == 0:
            return self.x_train, self.y_train, self.indices_train
        else:
            return self.get_current_update_batch(update_num)


feeders.register_builder("current_static", CurrentStatic)
