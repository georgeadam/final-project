from torch.optim import SGD

from .creation import optimizers


def sgd_wrapper(parameters, lr, momentum, weight_decay, **args):
    return SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)


optimizers.register_builder("sgd", sgd_wrapper)
