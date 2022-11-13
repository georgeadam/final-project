from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    def __init__(self, x, y, indices, transform, target_transform):
        self.x = x
        self.y = y
        self.indices = indices
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.x)
