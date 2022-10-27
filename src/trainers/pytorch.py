from .creation import trainers
from pytorch_lightning import Trainer


trainers.register_builder("pytorch", Trainer)

