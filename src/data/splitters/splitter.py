import abc


class SplitterInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def split_data(self, x, y, indices):
        raise NotImplementedError
