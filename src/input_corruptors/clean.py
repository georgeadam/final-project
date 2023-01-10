from .creation import input_corruptors
from .input_corruptor import InputCorruptor


class Clean(InputCorruptor):
    def __init__(self, noise_tracker, applicator_args, sample_limit, seed):
        super().__init__(noise_tracker, applicator_args, sample_limit, seed)

    def get_actual_indices(self, x, sample_indices):
        return []

    def get_potential_indices(self, x, sample_indices):
        return []

    def get_corruption_indices(self, x, sample_indices):
        return []

    def get_relevant_indices(self, x, sample_indices):
        return []


input_corruptors.register_builder("clean", Clean)
