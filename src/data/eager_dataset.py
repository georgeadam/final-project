from .dataset import Dataset


class EagerDataset(Dataset):
    def __init__(self, x, y, indices, transform, target_transform):
        super().__init__(x, y, indices, transform, target_transform)

    def __getitem__(self, index):
        x, y, index = self.x[index], self.y[index], self.indices[index]

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y, index
