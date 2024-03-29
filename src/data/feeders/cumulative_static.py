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
        return self.get_train_cumulative_data(update_num)

    def _get_val_data(self, update_num):
        return self.get_val_cumulative_data(update_num)


feeders.register_builder("cumulative_static", CumulativeStatic)
