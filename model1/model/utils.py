import os
import shutil
import numpy as np
import torch
from glob import glob
from tqdm import tqdm


def train_one_epoch(
    model,
    optimizer,
    dl,
    device,
    epoch,
):
    model.train()
    # do metric logging

    print_logs()
    if opt.resume:
        ckpt = load_checkpoint(opt_path_to_checkpoint)


def print_logs():
    return None


class MetricLogger:
    def __init__(self) -> None:
        pass


def save_checkpoint(state, save_path: str, is_best: bool = False, max_keep: int = None):
    """
    Saves torch model to checkpoint file.

    Parameters
    ----------
        state (torch model state):
            State of a torch Neural Network

        save_path (str):
            Destination path for saving checkpoint

        is_best (bool):
            If ``True`` creates additional copy
            ``best_model.ckpt``

        max_keep (int):
            Specifies the max amount of checkpoints to keep
    """
    # save checkpoint
    torch.save(state, save_path)

    # deal with max_keep
    save_dir = os.path.dirname(save_path)
    list_path = os.path.join(save_dir, "latest_checkpoint.txt")

    save_path = os.path.basename(save_path)
    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
            ckpt_list = [save_path + "\n"] + ckpt_list
    else:
        ckpt_list = [save_path + "\n"]

    if max_keep is not None:
        for ckpt in ckpt_list[max_keep:]:
            ckpt = os.path.join(save_dir, ckpt[:-1])
            if os.path.exists(ckpt):
                os.remove(ckpt)
        ckpt_list[max_keep:] = []

    with open(list_path, "w") as f:
        f.writelines(ckpt_list)

    # copy best
    if is_best:
        shutil.copyfile(save_path, os.path.join(save_dir, "best_model.ckpt"))

    tqdm.write(f"Checkpoint saved for epoch - {save_path}")


def load_checkpoint(ckpt_dir_or_file: str, map_location=None):
    """
    Loads torch model from checkpoint file.

    Parameters
    ----------
        ckpt_dir_or_file (str):
            Path to checkpoint directory or filename

        map_location:
            Can be used to directly load to specific device
    """

    list_of_checkpoints = glob("*.ckpt")
    # latest_checkpoint_file = max(list_of_checkpoints, key=os.path.getctime)

    # if ckpt_dir_or_file == "latest":
    #     ckpt_path = latest_checkpoint_file

    # else:
    ckpt_path = ckpt_dir_or_file
    ckpt = torch.load(ckpt_path, map_location=map_location)
    logging.info(" [*] Loading checkpoint from %s succeed!" % ckpt_path)
    return ckpt


def ensure_dir(dir_name: str):
    """Creates folder if not exists."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
