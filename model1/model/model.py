import json
import logging
import os
from datetime import datetime

import torch
import torch.nn as nn
from pycocotools.coco import COCO
from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.models.convnext import convnext_small
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.backbone_utils import (
    _resnet_fpn_extractor,
    _validate_trainable_layers,
)
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.resnet import resnet50, resnet152, resnext50_32x4d
from tqdm import tqdm

import torchvision
from torchvision.models.detection import KeypointRCNN
from torchvision.ops import misc as misc_nn_ops

if __name__ != "model1.model.model":
    from model.utils import save_checkpoint, load_checkpoint

    from cocoanalyze.pycocotools.coco import COCO
    from cocoanalyze.pycocotools.cocoanalyze import COCOanalyze

import gc
import time
from bisect import bisect_left
from operator import itemgetter
from typing import Optional, Union

from torch.profiler import ProfilerActivity, profile, record_function
from torchviz import make_dot

import wandb
from error_analysis.error_analysis import run_coco_analyze

# "keypointrcnn_resnet50_fpn_coco": "https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-fc266e95.pth",
#     "keypointrcnn_resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",


class CustomKeypointRCNN(KeypointRCNN):
    """
    A class to represent the custom KeypointRCNN model.

    Attributes
    ----------
    num_classes:
        Specifies the number of classes used in the model.
    num_keypoints:
        Specifies the number of keypoints used in the model.
    mixed_precision:
        [WIP] Run the model in mixed precision training

    Methods
    -------
    _train():
        Prints the person's name and age.
    _test()
    """

    def __init__(
        self,
        num_classes: int,
        num_keypoints: int,
        mixed_precision: bool = False,
        **kwargs,
    ) -> None:

        """
        Contsruct a custom KeypointRCNN model.

        Parameters
        ----------
        num_classes:
            Specifies the number of classes used in the model.
        num_keypoints:
            Specifies the number of keypoints used in the model.
        mixed_precision:
            [WIP] Run the model in mixed precision training

        Returns
        -------
        None
        """

        self.init_timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
        self.num_keypoints = num_keypoints

        # self.log_batch = False

        trainable_backbone_layers = _validate_trainable_layers(True, None, 5, 3)
        backbone = self._load_backbone("resnet50")
        backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)

        super().__init__(
            backbone, num_classes=num_classes, num_keypoints=self.num_keypoints
        )
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # self.device = torch.device("cpu")
        self.to(self.device)
        # key = "keypointrcnn_resnet50_fpn_coco"
        # model_urls = {
        #     "keypointrcnn_resnet50_fpn_coco": "https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-fc266e95.pth",
        #     "keypointrcnn_resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
        # }
        # # key = "keypointrcnn_resnet152"
        # # model_urls = {
        # #     "keypointrcnn_resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
        # # }

        # state_dict = load_state_dict_from_url(model_urls[key], progress=True)
        # self.load_state_dict(state_dict)

        overwrite_eps(self, 0.0)

        self.frozen_layers = dict()
        for name, param in self.named_parameters():
            if "body.layer1" in name or "body.conv1" in name:
                param.requires_grad = False
            self.frozen_layers[name] = param.requires_grad
            logging.debug(name)
            logging.debug(param.requires_grad)

        logging.debug("Init Layers:")

        self.apply(self._init_weights)
        # with open("init test kaiming only", "w") as f:
        #     for n, b in self.named_children():
        #         for na, ba in b.named_parameters():
        #             f.write(f"{na} - > {ba[0]}\n")
        # import sys

        # sys.exit()

    @staticmethod
    def _init_weights(m):
        for name, mod in m.named_modules():
            if any(word in name for word in ["roi", "rpn"]):
                if (
                    isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear))
                    and "backbone" not in name
                ):
                    logging.debug(name)
                    nn.init.kaiming_normal_(
                        mod.weight, mode="fan_in", nonlinearity="relu"
                    )
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)

    @staticmethod
    def _load_backbone(net: str):
        if net == "resnext50":
            backbone = resnext50_32x4d(
                pretrained=True,
                progress=True,
                norm_layer=misc_nn_ops.FrozenBatchNorm2d,
            )
        elif net == "resnet152":
            backbone = resnet152(
                pretrained=True,
                progress=True,
                norm_layer=misc_nn_ops.FrozenBatchNorm2d,
            )
        elif net == "resnet50":
            backbone = resnet50(
                pretrained=True,
                progress=True,
                norm_layer=misc_nn_ops.FrozenBatchNorm2d,
            )
        elif net == "convnext":
            backbone = convnext_small(
                pretrained=True,
                progress=True,
                norm_layer=misc_nn_ops.FrozenBatchNorm2d,
            )
        else:
            raise ValueError(f"{net} is not a supported backbone model.")

        return backbone

    def _train(
        self,
        # sort this opt out
        # opt,
        num_epochs=1,
        train_dl=None,
        valid_dl=None,
        optimizer=None,
        lr_scheduler=None,
        save_interval=1,
        train_method_cb=None,
        test=False,
        vis_data=False,
    ) -> None:
        """
        Run the traininng loop on the model.

        Parameters
        ----------
        # opt,
        num_epochs=1,
        train_dl=None,
        valid_dl=None,
        optimizer=None,
        lr_scheduler=None,
        save_interval=1,
        train_method_cb=None,

        Returns
        -------
        None
        """

        self.start_epoch = 0
        self.epochs = num_epochs
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.optimizer = optimizer
        self.scheduler = lr_scheduler

        # TODO need to resume with same training and valid data
        if self.opt.resume:
            raise NotImplementedError(
                "Resume function currently not fully implemented - does not resume with same trianing and validation sets."
            )
            ckpt = load_checkpoint(self.opt.path_to_checkpoint)
            self.load_state_dict(ckpt["net"])
            self.start_epoch = ckpt["epoch"] + 1
            self.epochs = self.start_epoch + self.epochs
            # start_n_iter = ckpt["n_iter"]
            self.optimizer.load_state_dict(ckpt["optim"])
            logging.info("last checkpoint restored")

        loop_epoch = tqdm(
            range(self.start_epoch, self.epochs),
            unit="EPOCH",
            desc="Epoch number",
            position=0,
        )

        avg_valid_losses, avg_train_losses = [], []

        wandb.watch(self, log="all")
        tbl = wandb.Table(columns=["train_loss_avg", "valid_loss_avg"])

        for epoch in loop_epoch:
            self.current_epoch = epoch
            loop_epoch.write(f"|EPOCH {str(epoch)}|")
            self.loss_history = []

            train_loop_batch = tqdm(
                enumerate(self.train_dl),
                desc="Batch",
                leave=False,
                total=len(self.train_dl),
            )
            step = 0
            total_train_loss = 0

            self.train()

            for i, (image_batch, targets) in train_loop_batch:
                step += 1
                # TODO - crude implementation of checking if keypoints are 0 and skipping
                # only checks 1 item in batch also
                if (targets[0].get("keypoints") == 0).all():
                    continue

                # TODO: find better place for target to device - should be done when data is transformed?
                image_batch = list(image.to(self.device) for image in image_batch)
                targets = [
                    {k: v.to(self.device) for k, v in t.items()} for t in targets
                ]

                if vis_data:
                    visualize_data(image_batch, targets)
                output_dict = self(image_batch, targets)

                train_loss = sum(loss for loss in output_dict.values())
                total_train_loss += train_loss.item()

                output_dict = {k + f" epoch {epoch}": v for k, v in output_dict.items()}
                self.output = output_dict

                loss_log = {
                    f"train loss sum": train_loss,
                    "epoch #": self.current_epoch,
                }
                loss_log.update(output_dict)

                # my_custom_data[0].append(i),
                # my_custom_data[1].append(train_loss.item()),
                # wandb.log(
                #     {
                #         "custom_data_table": wandb.Table(
                #             data=my_custom_data, columns=["batch", "loss"]
                #         )
                #     }
                # )
                # TODO use this to make custom charts
                # Set up data to log in custom charts
                #   data = []
                #   for i in range(100):
                #     data.append([i, random.random() + math.log(1 + i) + offset + random.random()])

                #   # Create a table with the columns to plot
                #   table = wandb.Table(data=data, columns=["step", "height"])

                #   # Use the table to populate various custom charts
                #   line_plot = wandb.plot.line(table, x='step', y='height', title='Line Plot')
                #   histogram = wandb.plot.histogram(table, value='height', title='Histogram')
                #   scatter = wandb.plot.scatter(table, x='step', y='height', title='Scatter Plot')

                #   # Log custom tables, which will show up in customizable charts in the UI
                #   wandb.log({'line_1': line_plot,
                #              'histogram_1': histogram,
                #              'scatter_1': scatter})

                self.loss_history.append(train_loss.item())
                # self.log_batch = False
                # TODO something that allows logging to work with overfit batch which is amller than 100
                if i % 100 == 0:
                    # self.log_batch = True
                    train_loop_batch.set_postfix(loss=train_loss.item())
                    wandb.log(loss_log, step=(self.current_epoch + 1) * i)
                    wandb.log(
                        {"learning_rate": self.optimizer.param_groups[0]["lr"]},
                        step=i,
                    )
                    if int(output_dict.get(f"loss_keypoint epoch {epoch}").item()) == 0:
                        # logging.debug(output_dict.get(f"loss_keypoint epoch {epoch}"))
                        # logging.debug("\n\nOutput Dict ")
                        # logging.debug(str(output_dict))
                        # logging.debug("\n\nTargets")
                        # logging.debug(str(targets))
                        wandb_images = [wandb.Image(im) for im in image_batch]
                        wandb.log({"0 Error Image": wandb_images})

                self.grad_accum(i, train_loss, image_batch)

            valid_loop_batch = tqdm(
                enumerate(self.valid_dl),
                desc="Image",
                leave=False,
                total=len(self.valid_dl),
            )

            # refactor this as very similar to train loop
            total_valid_loss = 0
            valid_error = None
            # prof.export_chrome_trace("trace.json")

            # self.eval()
            # for i, (images, targets) in valid_loop_batch:
            #     images = list(image.to(self.device) for image in images)
            #     targets = [
            #         {k: v.to(self.device) for k, v in t.items()} for t in targets
            #     ]
            # with torch.inference_mode():
            #     outputs = self(images, targets)
            #     output_preds = outputs[0]
            #     output_losses = outputs[1]

            #     valid_loss = sum(loss for loss in output_losses.values())
            #     total_valid_loss += valid_loss.item()

            #     batch_mae = torch.abs(
            #         [t["keypoints"][..., :2] for t in output_preds][0]
            #         - [t["keypoints"][..., :2] for t in targets][0]
            #     ).mean()

            #     if valid_error == None:
            #         valid_error = batch_mae.unsqueeze(0)

            #     else:
            #         valid_error = torch.cat((valid_error, batch_mae.unsqueeze(0)))

            #     if i % 100 == 0:
            #         valid_error = valid_error.mean()
            #         print("MAE LOG")
            #         wandb.log({"valid_error (MAE)": valid_error})
            #         valid_error = None

            # with torch.inference_mode():
            #     avg_train_loss = total_train_loss / len(self.train_dl)
            #     avg_train_losses.append(avg_train_loss)

            #     avg_valid_loss = total_valid_loss / len(self.valid_dl)
            #     avg_valid_losses.append(avg_valid_loss)

            if epoch % save_interval == 0:
                cpkt = {
                    "net": self.state_dict(),
                    "epoch": epoch,
                    "optim": optimizer.state_dict(),
                    # "valid_data_coco_indices": valid_data_coco_indices,
                    "frozen_layers": self.frozen_layers,
                }
                ckpt_path = f"checkpoints/model_checkpoint_keypoints_{self.num_keypoints}_epoch_{epoch}_{datetime.now().strftime('%Y-%m-%d-%H%M')}.ckpt"
                # if not test or len(self.train.dl) <= 64:
                #     save_checkpoint(cpkt, ckpt_path)

            # tbl.add_data(avg_train_loss, avg_valid_loss)
            wandb.log({"Avg Loss per Epoch TEST": tbl})
            wandb.log(
                {
                    "epoch_losses": wandb.plot.line_series(
                        xs=range(self.epochs),
                        ys=[avg_train_losses, avg_valid_losses],
                        keys=["train_loss_avg", "valid_loss_avg"],
                        title="Avg Loss per Epoch test",
                        xname="epoch",
                    )
                }
            )

        save_checkpoint(cpkt, ckpt_path)

    def grad_accum(
        self,
        i: int,
        train_loss: torch.tensor,
        image_batch: torch.tensor,
        accum_steps: int = 64,
    ) -> None:
        """
        Perform Gradient accumulation when training the model.

        Parameters
        ----------

        Returns
        -------
        None
        """

        train_loss /= len(image_batch)
        train_loss.backward()

        if ((i + 1) % accum_steps) == 0 or ((i + 1) == len(image_batch)):
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()
        else:
            if self.scheduler:
                self.scheduler.step()

    def _test(
        self,
        pred_file: Union[str, os.PathLike] = None,
        test_ds=None,
        test_dl=None,
    ) -> None:
        """
        - Make predictions on the validation set and save in a file.
        - Run an error analysis on the prediction results.

        Parameters
        ----------
        pred_file:
            Path to prediciton file.

        test_ds:
            The validation dataset on which to run the tests.

        test_dl:
            The validation DataLoader on which to run the tests.

        Returns
        -------
        None
        """

        logging.info("Running Evaluation...")
        self.eval()
        self.to(self.device)

        with torch.inference_mode():
            if pred_file == None:
                pred_file = f"preds_{self.init_timestamp}.json"

            if type(test_ds) is torch.utils.data.dataset.Subset:
                test_ds = test_ds.dataset

            if pred_file or test_dl:
                self._make_predictions(pred_file, test_ds, test_dl)
            else:
                raise ValueError("Either pred_file or test_dl must be given.")

            logging.debug("ANNPATH")
            logging.debug(test_ds.ann_path)

            run_coco_analyze(pred_file, test_ds.ann_path)

    def _make_predictions(
        self,
        pred_file: Union[str, os.PathLike],
        test_ds: torch.utils.data.dataset,
        test_dl: torch.utils.data.DataLoader,
    ) -> None:
        """
        Convenience function for quick testing very specific dataset
        To be removed
        """

        """
        Parameters
        ----------

        Returns
        -------
        None
        """
        preds = []
        worst_to_best_preds = SortedCollection(key=itemgetter(1))

        for example in tqdm(
            test_dl,
            desc="Images",
            # total=len(all_imgs),
            # position=0,
            leave=False,
        ):

            image_tensor = example[0][0].cuda()
            image_id = example[1][0]["image_id"]

            anns_id_sorted = sorted(
                test_ds.cocoanns.loadAnns(test_ds.annids), key=lambda item: item["id"]
            )
            ids = [i["id"] for i in anns_id_sorted]
            img_ids = [i["image_id"] for i in anns_id_sorted]

            if image_id in img_ids:
                with torch.inference_mode():
                    # indexing the second 0 here probs should not be done if multiple people detected
                    self.cpu()
                    out = self([image_tensor.cpu()])
                res = [[dict(zip(o.keys(), i)) for i in zip(*o.values())] for o in out]
                res = [
                    [{k: v.flatten().tolist() for k, v in i.items()} for i in r]
                    for r in res
                ]
                res = [[dict(i, **{"image_id": image_id}) for i in r] for r in res]
                res = [[dict(i, **{"category_id": 1}) for i in r] for r in res]

                for person in res:
                    for r in person:
                        r["score"] = r.pop("scores")[0]
                        r["labels"] = r["labels"][0]
                        r["image_id"] = r["image_id"].item()
                        # save all results also but have a way to extract only ones in validation set too
                        idx = [
                            (i, v)
                            for i, v in enumerate(anns_id_sorted)
                            if v["image_id"] == image_id
                        ][0][0]
                        kp_score = (
                            r["image_id"],
                            sum(r["keypoints_scores"]),
                            anns_id_sorted[idx]["keypoints"],
                        )
                        worst_to_best_preds.insert(kp_score)
                    preds.extend(person)

        with open(pred_file, "w") as f:
            json.dump(preds, f)

        with open(f"{pred_file}_sorted.txt", "w") as f:
            for l in worst_to_best_preds:
                f.write(str(l) + "\n")


def collate_fn(batch):
    return tuple(zip(*batch))


class SortedCollection(object):
    def __init__(self, iterable=(), key=None):
        """
        Parameters
        ----------

        Returns
        -------
        None
        """
        self._given_key = key
        key = (lambda x: x) if key is None else key
        decorated = sorted((key(item), item) for item in iterable)
        self._keys = [k for k, item in decorated]
        self._items = [item for k, item in decorated]
        self._key = key

    def __getitem__(self, i):
        return self._items[i]

    def __iter__(self):
        return iter(self._items)

    def __repr__(self):
        return repr(self._items)

    def insert(self, item):
        "Insert a new item.  If equal keys are found, add to the left"
        """
        Parameters
        ----------

        Returns
        -------
        None
        """
        k = self._key(item)
        i = bisect_left(self._keys, k)
        self._keys.insert(i, k)
        self._items.insert(i, item)


def visualize_data(image, labels):
    image = image[0].cpu()
    labels = labels
    from matplotlib import pyplot as plt

    plt.figure(figsize=(20, 15))
    result_image = image.permute(1, 2, 0)
    n = 5
    colors = iter(["green", "blue", "orange", "cyan"])
    for i in range(len(labels)):
        x = labels[i]["keypoints"][0, :, 0].cpu().detach()
        y = labels[i]["keypoints"][0, :, 1].cpu().detach()

        plt.scatter(x, y, color=next(colors), s=1)
    plt.imshow(result_image)
    plt.show()
