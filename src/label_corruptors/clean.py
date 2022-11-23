from .creation import label_corruptors
from .label_corruptor import LabelCorruptor


class Clean(LabelCorruptor):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def corrupt_helper(self, y, **kwargs):
        return y

    def get_actual_indices(self, *args, **kwargs):
        return []

    def get_potential_indices(self, *args, **kwargs):
        return []


label_corruptors.register_builder("clean", Clean)
