from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    def __init__(self, x, y, transform):
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]

        if self.transform is not None:
            x = self.transform(x)

        return x, y
