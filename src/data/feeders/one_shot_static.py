from .batch_fetchers import NullBatchFetcher
from .creation import feeders
from .input_updaters import NullInputUpdater
from .label_updaters import NullLabelUpdater
from .static import Static


class OneShotStatic(Static):
    def __init__(self, splitted_data, val_percentage, random_state):
        super().__init__(splitted_data, val_percentage, 0, random_state)

        self._input_updater = NullInputUpdater()
        self._label_updater = NullLabelUpdater()
        self._batch_fetcher = NullBatchFetcher()

    def _get_all_data_for_split(self, update_num):
        return self.x_train, self.y_train, self.indices_train


feeders.register_builder("one_shot_static", OneShotStatic)
