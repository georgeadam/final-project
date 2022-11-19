import abc


class BatchFetcherInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fetch(self, x_update, y_update, indices_update, update_num):
        raise NotImplementedError
