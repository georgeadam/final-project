import abc


class TrackerInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def track(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def track_helper(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def get_table(self):
        raise NotImplementedError
