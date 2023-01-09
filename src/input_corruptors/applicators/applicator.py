import abc

from torchvision import transforms

class Applicator:
    def __init__(self, severity):
        self.severity = severity
        self.transform = transforms.ToPILImage()

    @abc.abstractmethod
    def apply_corruption(self, x):
        raise NotImplementedError
