from .creation import label_corruptors
from .label_corruptor import LabelCorruptor


class Clean(LabelCorruptor):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def corrupt(self, module, data_module, trainer, update_num):
        pass


label_corruptors.register_builder("clean", Clean)
