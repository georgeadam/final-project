import abc
import copy

import numpy as np
import torchvision.transforms as transforms

from .applicators import applicators


class InputCorruptor:
    def __init__(self, noise_tracker, applicator_args, sample_limit=float("inf"), seed=0):
        self.noise_tracker = noise_tracker
        self.sample_limit = sample_limit
        self.seed = seed
        self.applicator = applicators.create(applicator_args.name, **applicator_args.params)

        self.transform = transforms.Compose([transforms.ToTensor()])

    def corrupt(self, data_module, inferer, update_num):
        update_batch_dataloader = data_module.current_update_batch_dataloader(update_num)
        update_batch_dataloader.dataset.transform = self.transform
        x, sample_indices = inferer.make_predictions(dataloaders=update_batch_dataloader)

        new_x, relative_indices = self.corrupt_helper(x, sample_indices=sample_indices)
        actual_indices = self.get_actual_indices(x, sample_indices=sample_indices)
        potential_indices = self.get_potential_indices(x, sample_indices=sample_indices)
        self.noise_tracker.track(update_num, actual_indices, potential_indices)
        data_module.overwrite_current_update_inputs(new_x, relative_indices, update_num)

    def corrupt_train(self, data_module, inferer):
        train_dataloader = data_module.train_initial_dataloader()
        train_dataloader.dataset.transform = self.transform
        x, sample_indices = inferer.make_predictions(dataloaders=train_dataloader)

        new_x, relative_indices = self.corrupt_helper(x, sample_indices=sample_indices)
        actual_indices = self.get_actual_indices(x, sample_indices=sample_indices)
        potential_indices = self.get_potential_indices(x, sample_indices=sample_indices)
        self.noise_tracker.track(0, actual_indices, potential_indices)
        data_module.overwrite_train_inputs(new_x, relative_indices)

    def corrupt_helper(self, x, sample_indices):
        x = copy.deepcopy(x)
        corruption_indices = self.get_corruption_indices(x=x, sample_indices=sample_indices)
        x = self.applicator.apply_corruption(x[corruption_indices])

        return x, corruption_indices

    @abc.abstractmethod
    def get_actual_indices(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def get_potential_indices(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def get_corruption_indices(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def get_relevant_indices(self, *args, **kwargs):
        raise NotImplementedError

    def subset_indices(self, indices, sample_limit):
        if sample_limit is not None and len(indices) > sample_limit:
            random_state = np.random.RandomState(self.seed)
            indices = random_state.choice(indices, min(len(indices), sample_limit), replace=False)

        return indices
