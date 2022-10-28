from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    def __init__(self, x, y, transform, target_transform):
        self.x = x
        self.y = y
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y

    def __len__(self):
        return len(self.x)