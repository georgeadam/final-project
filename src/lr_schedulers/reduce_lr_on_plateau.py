from torch.optim.lr_scheduler import ReduceLROnPlateau

from .creation import lr_schedulers


def reduce_lr_on_plateau_wrapper(optimizer, **kwargs):
    return ReduceLROnPlateau(optimizer, **kwargs)


lr_schedulers.register_builder("reduce_lr_on_plateau", reduce_lr_on_plateau_wrapper)
