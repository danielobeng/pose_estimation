import torch.optim as optim


def get_optimizer(params, lr=0.001, momentum=0.9, weight_decay=0.0005):
    # return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optim.AdamW(params, lr, weight_decay=weight_decay)


def get_scheduler(optimizer, lr, num_epochs, steps_per_epoch, step_size=3, gamma=0.01):
    # return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return optim.lr_scheduler.OneCycleLR(
        optimizer, epochs=num_epochs, steps_per_epoch=steps_per_epoch, max_lr=lr
    )
