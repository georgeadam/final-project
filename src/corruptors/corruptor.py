import abc


class CorruptorInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def corrupt(self, module, data_module, trainer, update_num):
        raise NotImplementedError
