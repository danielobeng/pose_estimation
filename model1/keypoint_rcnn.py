import argparse
import logging
import os
from functools import partial
from pathlib import Path
from typing import Union

import torch
from torch import profiler

import utils
import wandb
from model.dataset import get_data
from model.hooks import activation_hook, gradients_hook
from model.model import CustomKeypointRCNN
from model.ops import get_optimizer, get_scheduler
from utils import get_module_by_name, optimize_num_workers

logging.basicConfig(filename="keypoint_rcnn.log", encoding="utf-8", level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logging.getLogger().addHandler(console)

ROOT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
EPOCHS = 1
BATCH_SIZE = 1
LR = 25e-4
NUM_CLASSES = 2
NUM_WORKERS = 3
NUM_KEYPOINTS = 23
MOMENTUM = 0.95
W_DECAY = 0.0005
GAMMA = 0.1
LR_SCHED_STEP = 1
None

pin_memory = True

BASE_DIR = Path(os.path.dirname(os.path.realpath(__file__))).parent

if __name__ == "__main__":
    # argparse for additional flags for experiment
    parser = argparse.ArgumentParser(description="Train a keypoint estimator.")
    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
        help=f"Spcifies learing rate for optimizer (default: {LR})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=f"If set resumes training from provided checkpoint (default: {False})",
    )
    parser.add_argument(
        "--path-to-checkpoint",
        type=str,
        default="latest",
        help=f'Path to checkpoint to resume training (default: "latest")',
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for data loaders (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Number of workers for data loader. (default: {NUM_WORKERS})",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=NUM_CLASSES,
        help=f"Number of classes of dataset. (default: {NUM_CLASSES})",
    )
    parser.add_argument(
        "--num-keypoints",
        type=int,
        default=NUM_KEYPOINTS,
        help=f"Number of labelled keypoints per person. (default: {NUM_KEYPOINTS})",
    )
    parser.add_argument(
        "--momentum",
        type=int,
        default=MOMENTUM,
        help=f"Specify momentum for the learning rate (default: {MOMENTUM})",
    )
    parser.add_argument(
        "--w-decay",
        type=int,
        default=W_DECAY,
        help=f"Speficy weight decay for the learning rate (default: {W_DECAY})",
    )
    parser.add_argument(
        "--gamma",
        type=int,
        default=GAMMA,
        help=f"Spedicy learning rate gamma value (default: {GAMMA})",
    )
    parser.add_argument(
        "--lr-scheduler-step",
        type=int,
        default=LR_SCHED_STEP,
        help=f"Specify step size for learning rate scheduler (default: {LR_SCHED_STEP})",
    )
    parser.add_argument(
        "--train",
        "-t",
        action="store_true",
        help=f"Run in training mode. (default: {False})",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=f"Run in testing mode. (default: {False})",
    )
    parser.add_argument(
        "--pred-file",
        default=None,
        help=f"Path to pre-computed predictions if running --test. (default: {None})",
    )
    parser.add_argument(
        "--model-weights",
        default=None,
        help=f"Path to model weights if running --test. (default: {None})",
    )
    parser.add_argument(
        "--dummy-data",
        action="store_true",
        default=False,
        help=f"Load a small only a small amount of dummy data from the dataset (default: {False})",
    )
    parser.add_argument(
        "--log",
        "-l",
        action="store_true",
        default=False,
        help=f"Log trianind data with Weights & Biases. Requires login. (default: {False})",
    )
    parser.add_argument(
        "--optimizer",
        choices=["AdamW", "SGD"],
        default="AdamW",
        help=f"Optimizer used during training. (default: AdamW)",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=[None, "one_cycle", "step_lr"],
        default="one_cycle",
        help=f"Optimizer used during training. (default: one_cycle)",
    )
    parser.add_argument(
        "--optimize-workers",
        action="store_true",
        default=False,
        help=f"Loop from num_workers=1-20 to determine optimal num_workers. (default: {False})",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=False,
        help=f"Choose device on which to train, test, and load data (default: {False})",
    )
    parser.add_argument(
        "--overfit",
        action="store_true",
        default=False,
        help=f"Overfit model by running on a small dummy dataset over a large number of epochs (default: {False})",
    )

    opt = parser.parse_args()

    assert opt.train or opt.test, AssertionError(
        "Either --train or --test must be passed."
    )

    wandb_notes = ""
    wandb_tags = [
        "",
    ]

    wandb_config = dict(
        architecture="resnet",
        dataset_id="peds-0001",
        learning_rate=opt.lr,
        momentum=opt.momentum,
        num_workers=opt.momentum,
        num_keypoints=opt.num_keypoints,
        batch_size=opt.batch_size,
        num_epochs=opt.epochs,
        num_classes=opt.num_classes,
        w_decay=opt.w_decay,
        gamma=opt.gamma,
        lr_scheduler_step=opt.lr_scheduler_step,
        optimizer=opt.optimizer,
    )

    data_paths = {
        17: {
            "train": {
                "img_path": f"{BASE_DIR}/data/train2017",
                "ann_path": f"{BASE_DIR}/data/annotation_trainval2017/annotations/person_keypoints_train2017.json",
            },
            "test": {
                "img_path": f"{BASE_DIR}/data/val2017",
                "ann_path": f"{BASE_DIR}/data/annotation_trainval2017/annotations/person_keypoints_val2017.json",
            },
        },
        23: {
            "train": {
                "img_path": f"{BASE_DIR}/data/train2017",
                "ann_path": f"{BASE_DIR}/data/person_keypoints_train2017_foot_v1.json",
            },
            "test": {
                "img_path": f"{BASE_DIR}/data/val2017",
                "ann_path": f"{BASE_DIR}/data/person_keypoints_val2017_foot_v1.json",
            },
        },
    }

    img_path, ann_path = data_paths[opt.num_keypoints][
        "train" if opt.train else "test"
    ].values()

    model = CustomKeypointRCNN(
        opt.num_classes,
        num_keypoints=opt.num_keypoints,
    )
    model.device = (
        torch.device("cuda")
        if (torch.cuda.is_available() & opt.device == False)
        else torch.device("cpu")
    )
    model.to(model.device)
    model.opt = opt

    if opt.optimize_workers:
        num_workers = optimize_num_workers()

    if opt.train:
        if opt.overfit:
            opt.dummy_data = True
            opt.epochs = 200
            wandb_notes = ""
            wandb_tags = "overfit test"
        (
            train_data,
            train_loader,
            valid_data,
            valid_loader,
            valid_data_coco_indices,
        ) = get_data(
            img_path=img_path,
            ann_path=ann_path,
            num_keypoins=opt.num_keypoints,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            dummy_data=opt.dummy_data,
            transform=False,
            pin_memory=pin_memory,
            is_test=opt.test,
        )
        params = [p for p in model.parameters() if p.requires_grad]
        params = model.parameters()

        optimizer = get_optimizer(
            opt.optimizer,
            params,
            lr=opt.lr,
            momentum=opt.momentum,
            weight_decay=opt.w_decay,
        )
        lr_scheduler = get_scheduler(
            opt.lr_scheduler,
            optimizer=optimizer,
            max_lr=opt.lr,
            epochs=opt.epochs,
            # in the case of gradient accum, need a way to set batch size correctly
            steps_per_epoch=len(train_data),
            pct_start=0.475,
            anneal_strategy="cos",
            cycle_momentum=True,
            base_momentum=0.75,
            max_momentum=opt.momentum,
            div_factor=10,
            final_div_factor=1e4,
        )

        wandb.init(
            mode="online" if opt.log else "disabled",
            project="23-keypoint-prediction",
            notes=wandb_notes,
            tags=wandb_tags,
            config=wandb_config,
            settings=wandb.Settings(code_dir="."),
        )

        model.register_buffer("valid_data_coco_indices", valid_data_coco_indices[0])

        model._train(
            num_epochs=opt.epochs,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dl=train_loader,
            valid_dl=valid_loader,
        )

        # if opt.dummy_data:
        #     for i in train_loader:
        #         image_id = i[1][0]["image_id"].item()
        #         extra_zero = lambda: 0 if len(str(image_id)) == 5 else ""

        model._test(
            test_ds=valid_data,
            test_dl=valid_loader,
        )

    if opt.test:
        model_weights = (
            ROOT_DIR
            # / "checkpoints/model_checkpoint_keypoints_23_epoch_4_2022-09-02-1408.ckpt"
            / "checkpoints/model_checkpoint_keypoints_17_epoch_0_2022-11-23-1959.ckpt"
        )
        (test_data, test_loader) = get_data(
            img_path=img_path,
            ann_path=ann_path,
            num_keypoins=opt.num_keypoints,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            dummy_data=opt.dummy_data,
            transform=False,
            pin_memory=pin_memory,
            is_test=opt.test,
        )

        state_dict = torch.load(model_weights)
        model.load_state_dict(state_dict["net"])

        assert model_weights != None

        with profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:

            model._test(pred_file=opt.pred_file, test_dl=test_loader, test_ds=test_data)

            prof.export_chrome_trace("trace.json")

        # utils.plot_worst_images(
        #     model, valid_data.dataset, "preds_2022-09-02-2217.json_sorted.txt"
        # )
