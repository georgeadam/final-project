import abc


class TrackerInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def track(self, module, data_module, trainer, partition, update_num):
        raise NotImplementedError

    @abc.abstractmethod
    def track_helper(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def get_table(self):
        raise NotImplementedError
