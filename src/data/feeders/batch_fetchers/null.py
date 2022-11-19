from .batch_fetcher import BatchFetcherInterface


class Null(BatchFetcherInterface):
    def fetch(self, x_update, y_update, indices_update, update_num):
        return None, None, None
