import numpy as np

from .creation import input_corruptors
from .input_corruptor import InputCorruptor


class UniformNoise(InputCorruptor):
    def __init__(self, noise_tracker, applicator_args, sample_limit, seed):
        super().__init__(noise_tracker, applicator_args, sample_limit, seed)

    def get_actual_indices(self, x, sample_indices):
        corruption_indices = self.get_corruption_indices(x)

        return list(sample_indices[corruption_indices])

    def get_potential_indices(self, x, sample_indices):
        potential_indices = self.get_relevant_indices(x)

        return list(sample_indices[potential_indices])

    def get_corruption_indices(self, x, **kwargs):
        indices = self.get_relevant_indices(x)
        indices = self.subset_indices(indices, self.sample_limit)

        return indices

    def get_relevant_indices(self, x):
        return np.arange(len(x))


input_corruptors.register_builder("uniform_noise", UniformNoise)
