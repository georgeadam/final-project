from .creation import trainers
from pytorch_lightning import Trainer


class PyTorch(Trainer):
    def __init__(self, update_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_num = update_num


trainers.register_builder("pytorch", PyTorch)

