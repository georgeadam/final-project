from .label_corruptor import LabelCorruptorInterface
from .creation import label_corruptors


class Clean(LabelCorruptorInterface):
    def __init__(self):
        pass

    def corrupt(self, module, data_module, trainer, update_num):
        pass


label_corruptors.register_builder("clean", Clean)
