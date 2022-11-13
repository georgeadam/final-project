from PIL import Image

from .dataset import Dataset


class LazyDataset(Dataset):
    def __init__(self, x, y, indices, transform, target_transform):
        super().__init__(x, y, indices, transform, target_transform)

    def __getitem__(self, index):
        x = Image.open(self.x[index]).convert('RGB')
        y, index = self.y[index], self.indices[index]

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y, index
