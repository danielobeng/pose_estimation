from typing import Generator, Union

import torch.optim as optim
from torch import nn


def get_optimizer(
    optim_name: str,
    params: Generator[nn.parameter.Parameter, None, None],
    lr: float = 0.001,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
) -> optim:
    """
    Gets a torch optimizer function to be used during model training.

    Parameters
    ----------
    params:
        model parameters

    lr:
        model learning rate

    momentum:
        some optimisers take momentum as a parameter

    weight_decay:
        some optimisers take weight_decay as a parameter

    Returns
    -------

    """
    if optim_name == "AdamW":
        return optim.AdamW(params, lr, weight_decay=weight_decay)
    elif optim_name == "SGD":
        return optim.SGD(params, lr)
    else:
        raise ValueError(f"Optimizer of type {optim} not supported.")


def get_scheduler(stype: Union[str, None], **kwargs) -> optim.lr_scheduler:
    """
    Gets a torch scheduler function to be used during model training.

    Parameters
    ----------
    stype:


    Returns
    -------

    """
    if stype == None:
        return None
    if stype == "one_cycle":
        cb = optim.lr_scheduler.OneCycleLR
    if stype == "step_lr":
        cb = optim.lr_scheduler.StepLR
    return cb(**kwargs)
