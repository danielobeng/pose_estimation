#%%
import os
import argparse
import torch

import utils

from model.model import CustomKeypointRCNN
from model.ops import get_optimizer, get_scheduler

from model.dataset import get_data

from torch.utils.tensorboard import SummaryWriter

# from engine import train_one_epoch, evaluate
# from torchvision._internally_replaced_utils import load_state_dict_from_url

# import GPUtil
import wandb

# training_runs = os.scandir("training_runs/keypoint_rcnn")
# latest_run = max(training_runs, key=os.path.getctime)
# num = int(latest_run.name.split("_")[1])
# current_run = f"run_{num +1}"

# writer = SummaryWriter(
#     f"training_runs/{os.path.splitext(os.path.basename(__file__))[0]}/{current_run}"
# )

torch.cuda.empty_cache()

# TODO
# EXPERIMENTS (GOAL - BETTER MODEL)
# ARCHITECHTURE -
# RESNET100, XRESNET, CONVNEXT
# - cf timm library
# - look at how fastai replaces models
# OPTIMISERS
# - SGD
# - ADAM
# -ADAMW
# Weight decay - Jeremy says 0.1 is best
# LR finder
# - the medium blog that says better to set an dynamic lr depending on the latyer
# instead of freezing layers, so small lr in early layers is interesting
# - do fastai fine tune method (cf live coding notes)
# - do fit one cycle and scheduling
# one cycle learning rates
# TRAINING
# Gradient accumulation - make callback
# initialisation cf fastai layerwise sequentuial unit varaiance
# DATA AUGMENTATIONS
# - Test time augmentation
# - look at fastai data augmentations
# - make sure normalising
# ERROR ANALYSIS
# - look at notes full stack ml and deeplearning ai
# visualise the most incorrect images to get some insight into why
# Metrics
# - look at how detectron measure accuracy
# - implement more metrics to look at https://nanonets.com/blog/human-pose-estimation-2d-guide/#:~:text=Object%20Keypoint%20Similarity%20(OKS)%20based%20mAP%3A&text=To%20put%20it%20simply%2C%20OKS,the%20scale%20of%20the%20person.


# - refactor code
# copy faster rcnn, gneeralised rcnn roi head files locally and use modifications instead of changing the actual torch packages
# try different hyperparm settings
# - use half precision points - compare speed - CANNOT BE DONE ON 1080Ti
# - get cocoanalze working and all plotting - half complete
# Use facebook's implementation of oks metric and cocoeval https://github.com/facebookresearch/DensePose/blob/main/detectron/datasets/densepose_cocoeval.py
# compare a training run with Adam, AdamW and SGD and SGD with momentum in weights and b - might also need to tune hyperparams or use fastai ones - good to experiment though
# understand and experiments with the loss function of keypoint rcnn
# use/check out torchvision's visualisation stuff for plotting keypoints
# try out methods from cs230 niotes kecture 7 interpretability of nnets
#  WHY DOES CALLING BACKWARD AFTER ZERO GRAD MAKE SUCH A DIFFERENCE?????
# Augmentations
# write own fit function, look at whe fit onec cycle does fastai
# OTHER
# - understand overwrite_eps(model, 0.0)
# - understand frozenbatchnorm vs batchnorm
# - add video capability (cf detectron)
# - test_dl method for things to do at testing
# - save transformations done when saving model also
# - try out weights and biases - DONE
# - understand OKS, AP, AR metrics - DONE
# - train on feet + body - DONE
# - get eval working - DONE
# - get visualisation working - DONE
# - see if you can fine tune on feet - DONE
# calculate some sort of errir rate DONE

# NOTE
# - when fine tuning, don't set learning rate too high you you risk catapulting
# your weights out of the pre-trained loss valley - use someing around 1e-4 or 1e-5
# bc break = backward compatability break
# always create a test to make sure your data is loaded correctly eg keypoints are
# actually correct
# Keep in mind that if we resize the images, then we have to rescale the keypoint coordinates as well. This will ensure that they match for the resized images.
#  Forgetting to rescale the keypoints will make our model learn all the wrong coordinates for facial features of the images.
# Lesson: when changing layers, still initialise them, this was the different between training working in 1 epoch and not working even over 5 epochs
#         nn.init.kaiming_normal_(
#             self.roi_heads.keypoint_predictor.kps_score_lowres.weight,
#             mode="fan_out",
#             nonlinearity="relu",
#         )
#         nn.init.constant_(self.roi_heads.keypoint_predictor.kps_score_lowres.bias, 0)
# - tuning OKS values here https://cocodataset.org/#keypoints-eval doesn't seem like a
# clean way to do things

# useful thing
#     if not os.path.exists("keypoint.py"):
#         import subprocess

#         subprocess.run(
#             [
#                 "wget",
#                 "https://raw.githubusercontent.com/facebookresearch/maskrcnn-benchmark/e0a525a0139baf7086117b7ed3fd318a4878d71c/maskrcnn_benchmark/structures/keypoint.py",
#             ],
#         )


# Computation time: If you freeze all the layers but the last 5 ones, you only need to backpropagate the gradient and update the weights of the last 5 layers. In contrast to backpropagating and updating the weights all the layers of the network, this means a huge decrease in computation time. For this reason, if you unfreeze all the network, this will allow you to see the data fewer epochs than if you were to update only the last layers weights'.
# Accuracy: Of course, by not updating the weights of most of the network your are only optimizing in a subset of the feature space. If your dataset is similar to any subset of the imagenet dataset, this should not matter a lot, but, if it is very different from imagenet, then freezing will mean a decrease in accuracy. If you have enough computation time, unfreezing everything will allow you to optimize in the whole feature space, allowing you to find better optima.
# To wrap up, I think that the main point is to check if your images are comparable to the ones in imagenet. In this case, I would not unfreeze many layers. Otherwise, unfreeze everything but get ready to wait for a long training time.

# Also, I would not recommend unfreezing all layers if you have any new/untrained layers in your model. These untrained layers will have large gradients in the first few epocs, and your model will train as if initialized by random(and not pre-trained) weights.


# in_features = model.roi_heads.box_predictor.cls_score.in_features
# anchor_generator = AnchorGenerator(
#     sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
# )

# # let's define what are the feature maps that we will
# # use to perform the region of interest cropping, as well as
# # the size of the crop after rescaling.
# # if your backbone returns a Tensor, featmap_names is expected to
# # be [0]. More generally, the backbone should return an
# # OrderedDict[Tensor], and in featmap_names you can choose which
# # feature maps to use.
# roi_pooler = torchvision.ops.MultiScaleRoIAlign(
#     featmap_names=[0], output_size=7, sampling_ratio=2
# )

# Reproducibility
# torch.backends.cudnn.benchmark = True
# torch.manual_seed(1)
# torch.cuda.manual_seed(1)

# Optimisations for faster training iterations

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# This will allow the cuda backend to optimize your graph during its first execution.
# However, be aware that if you change the network input/output tensor size the graph will be optimized
#  each time a change occurs. This can lead to very slow runtime and out of memory errors.
# Only set this flag if your input and output have always the same shape.
# Usually, this results in an improvement of about 20%.
# torch.backends.cudnn.benchmark = True

EPOCHS = 1
BATCH_SIZE = 1
LR = 1e-4
NUM_CLASSES = 2
NUM_WORKERS = 4
NUM_KEYPOINTS = 23
MOMENTUM = 0.95
W_DECAY = 0.0005
GAMMA = 0.1
LR_SCHED_STEP = 1

if __name__ == "__main__":
    # argparse for additional flags for experiment
    parser = argparse.ArgumentParser(description="Train a keypoint estimator.")
    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
        help=f"Spcifies learing rate for optimizer. (default: {LR})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set resumes training from provided checkpoint. (default: None)",
    )
    parser.add_argument(
        "--path-to-checkpoint",
        type=str,
        default="latest",
        # help=f'Path to checkpoint to resume training. (default: "{default}")',
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        # help=f"Number of training epochs. (default: {default})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        # help=f"Batch size for data loaders. (default: {default})",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        # help=f"Number of workers for data loader. (default: {default})",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=NUM_CLASSES,
        # help=f"Number of classes of dataset. (default: {default})",
    )
    parser.add_argument(
        "--num-keypoints",
        type=int,
        default=NUM_KEYPOINTS,
        # help=f"Number of classes of dataset. (default: {default})",
    )
    parser.add_argument(
        "--momentum",
        type=int,
        default=MOMENTUM,
        # help=f"Number of classes of dataset. (default: {default})",
    )
    parser.add_argument(
        "--wdecay",
        type=int,
        default=W_DECAY,
        # help=f"Number of classes of dataset. (default: {default})",
    )
    parser.add_argument(
        "--gamma",
        type=int,
        default=GAMMA,
        # help=f"Number of classes of dataset. (default: {default})",
    )
    parser.add_argument(
        "--lr-scheduler-step",
        type=int,
        default=LR_SCHED_STEP,
        # help=f"Number of classes of dataset. (default: {default})",
    )

    parser.add_argument(
        "--test-mode",
        action="store_true",
        # help=f"Run in testing mode. (default: {default})",
    )
    parser.add_argument(
        "--dummy-data",
        action="store_true",
        default=False
        # help=f"Run in testing mode. (default: {default})",
    )
    parser.add_argument("-n")

    opt = parser.parse_args()

    batch_size = opt.batch_size
    lr = opt.lr
    num_epochs = opt.epochs
    num_classes = (
        opt.num_classes
    )  # background and person classes is default number of classes required
    num_keypoints = opt.num_keypoints
    num_workers = opt.num_workers
    test_mode = opt.test_mode
    dummy_data = opt.dummy_data
    momentum = opt.momentum
    w_decay = opt.wdecay
    gamma = opt.gamma
    lr_scheduler_step = opt.lr_scheduler_step
    # notes = opt.notes

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    config = dict(
        architecture="resnet",
        dataset_id="peds-0001",
        learning_rate=lr,
        momentum=momentum,
        num_workers=num_workers,
        num_keypoints=num_keypoints,
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_classes=num_classes,
        w_decay=w_decay,
        gamma=gamma,
        lr_scheduler_step=lr_scheduler_step,
    )

    train_data, train_loader, valid_data, valid_loader = get_data(
        num_keypoins=num_keypoints,
        batch_size=batch_size,
        num_workers=num_workers,
        dummy_data=dummy_data,
    )
    # GPUtil.showUtilization()

    # utils.plot_example(dataset)
    # ex = next(iter(train_data))[0][0]
    # img_grid = torchvision.utils.make_grid(ex)

    model = CustomKeypointRCNN(
        num_classes,
        num_keypoints=num_keypoints,
        # writer=writer,
    )
    model.to(device)

    # params = [p for p in model.parameters() if p.requires_grad]
    params = model.parameters()
    optimizer = get_optimizer(params, lr=lr, momentum=momentum, weight_decay=w_decay)
    lr_scheduler = get_scheduler(
        optimizer,
        lr,
        num_epochs,
        steps_per_epoch=len(train_data),
        step_size=lr_scheduler_step,
        gamma=gamma,
    )

    # base_path = os.path.splitext(os.path.basename(__file__))[0]

    # need to sort out paths to not care from where code is run
    if test_mode:
        # state_dict = torch.load(
        #     "checkpoints/model_checkpoint_keypoints_23_epoch_3_2022-08-08-1634.ckpt"
        #     # "checkpoints/model_checkpoint_keypoints_23_epoch_2_2022-08-12-1227.ckpt"
        # )
        # model.load_state_dict(state_dict["net"])

        model._test(valid_loader, device)
        # model.eval()
        utils.plot_keypoints(
            model, f"{os.path.dirname(os.path.realpath(__file__))}/runner2.jpg"
        )

    else:
        wandb.init(
            project="23-keypoint-prediction",
            notes="adam w one cycle no gradient accum",
            tags=[
                "baseline",
            ],
            config=config,
        )

        model._train(
            opt,
            num_epochs=num_epochs,
            optimizer=optimizer,
            train_dl=train_loader,
            valid_dl=valid_loader,
            device=device,
            lr_scheduler=lr_scheduler,
        )
