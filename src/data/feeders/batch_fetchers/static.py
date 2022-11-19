from .batch_fetcher import BatchFetcherInterface


class Static(BatchFetcherInterface):
    def __init__(self, num_updates):
        self._num_updates = num_updates

    def fetch(self, x_update, y_update, indices_update, update_num):
        samples_per_update = int(len(x_update) / self._num_updates)
        x_update = x_update[(update_num - 1) * samples_per_update: update_num * samples_per_update]
        y_update = y_update[(update_num - 1) * samples_per_update: update_num * samples_per_update]
        indices_update = indices_update[(update_num - 1) * samples_per_update: update_num * samples_per_update]

        return x_update, y_update, indices_update
