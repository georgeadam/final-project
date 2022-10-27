from .corruptor import CorruptorInterface
from .creation import corruptors


class Clean(CorruptorInterface):
    def __init__(self):
        pass

    def corrupt(self, module, data_module, trainer, update_num):
        pass


corruptors.register_builder("clean", Clean)
