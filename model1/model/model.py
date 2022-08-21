import torch
import torch.nn as nn

import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import KeypointRCNN
from torchvision._internally_replaced_utils import load_state_dict_from_url

from torchvision.ops import misc as misc_nn_ops
from torchvision.models.resnet import resnet50
from torchvision.models.detection.backbone_utils import (
    _resnet_fpn_extractor,
    _validate_trainable_layers,
)
from torchvision.models.detection._utils import overwrite_eps
from pycocotools.coco import COCO


if __name__ != "model1.model.model":
    # from engine import train_one_epoch, evaluate
    from model.utils import save_checkpoint, load_checkpoint

    # from pycocotools.cocoanalyze import COCOanalyze
    from cocoanalyze.pycocotools.coco import COCO
    from cocoanalyze.pycocotools.cocoanalyze import COCOanalyze


from datetime import datetime


import json
from tqdm import tqdm
from PIL import Image
import os
import wandb

from torch.profiler import profile, record_function, ProfilerActivity


import torch, time, gc

# Timing utilities
start_time = None


def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()


def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print(
        "Max memory used by tensors = {} bytes".format(
            torch.cuda.max_memory_allocated()
        )
    )


class CustomKeypointRCNN(KeypointRCNN):
    def __init__(
        self,
        num_classes,
        num_keypoints=17,
        mixed_precision=False,
        **kwargs,
    ) -> None:

        trainable_backbone_layers = _validate_trainable_layers(True, None, 5, 3)
        backbone = resnet50(
            pretrained=True,
            progress=True,
            norm_layer=misc_nn_ops.FrozenBatchNorm2d,
        )
        backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)

        super().__init__(backbone, num_classes=num_classes, num_keypoints=17)
        self.num_keypoints = num_keypoints
        self.mixed_precision = mixed_precision

        # if writer:
        #     self.writer = writer

        key = "keypointrcnn_resnet50_fpn_coco"
        model_urls = {
            "keypointrcnn_resnet50_fpn_coco": "https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-fc266e95.pth",
        }
        state_dict = load_state_dict_from_url(model_urls[key], progress=True)
        self.load_state_dict(state_dict)

        overwrite_eps(self, 0.0)

        # self.roi_heads.keypoint_predictor.kps_score_lowres = nn.Sequential(
        #     self.roi_heads.keypoint_predictor.kps_score_lowres,
        #     # nn.Identity()
        #     nn.Conv2d(17, 124, kernel_size=(3, 3), stride=(1, 1)),
        #     nn.ConvTranspose2d(
        #         124, num_keypoints, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
        #     ),
        # # )

        for name, param in self.named_parameters():
            if "roi" not in name:
                param.requires_grad = False

        if num_keypoints == 23:
            self.roi_heads.keypoint_predictor.kps_score_lowres = nn.ConvTranspose2d(
                512,
                num_keypoints,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
            )
            nn.init.kaiming_normal_(
                self.roi_heads.keypoint_predictor.kps_score_lowres.weight,
                mode="fan_out",
                nonlinearity="relu",
            )
            nn.init.constant_(
                self.roi_heads.keypoint_predictor.kps_score_lowres.bias, 0
            )
        # print(self.backbone.body.conv1.weight)
        # print(self.roi_heads.box_head.fc6.weight)

        # print(
        #     self.roi_heads.keypoint_predictor.kps_score_lowres.get_parameter(
        #         "kernel_size"
        #     )
        # )

    def _train(
        self,
        # sort this opt out
        opt,
        num_epochs=1,
        train_dl=None,
        valid_dl=None,
        optimizer=None,
        lr_scheduler=None,
        device=None,
        save_interval=1,
    ):
        # assert data_loader!= None:
        #     raise AssertionError('Data Loader should not be None')

        self.start_epoch = 0
        self.epochs = num_epochs
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.optimizer = optimizer
        self.scheduler = lr_scheduler

        if opt.resume:
            ckpt = load_checkpoint(
                opt.path_to_checkpoint
            )  # custom method for loading last checkpoint
            self.load_state_dict(ckpt["net"])
            self.start_epoch = ckpt["epoch"]
            # start_n_iter = ckpt["n_iter"]
            self.optimiser.load_state_dict(ckpt["optim"])
            print("last checkpoint restored")

        loop_epoch = tqdm(
            range(self.start_epoch, self.epochs),
            unit="EPOCH",
            desc="Epoch number",
            position=0,
        )

        scaler = torch.cuda.amp.GradScaler()
        avg_valid_losses = []
        avg_train_losses = []

        for epoch in loop_epoch:
            # train_one_epoch(
            #     self,
            #     optimizer,
            #     data_loader,
            #     device,
            #     epoch,
            #     print_freq=10,
            #     writer=self.writer,
            # )
            # evaluate(self, data_loader, device=device)

            loop_epoch.write(f"|EPOCH {str(epoch)}|")

            self.loss_history = []

            train_loop_batch = tqdm(
                enumerate(self.train_dl),
                desc="Batch",
                leave=False,
                total=len(self.train_dl),
            )
            step = 0
            ex_count = 0
            total_train_loss = 0

            self.train()

            start_timer()
            for i, (image_batch, targets) in train_loop_batch:
                step += 1

                ex_count += len(image_batch)

                # TODO: find better place for target to device
                image_batch = list(image.to(device) for image in image_batch)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # image_batch[0] = image_batch[0]
                # with torch.autocast(device_type="cuda", dtype=torch.float16):

                # for t in targets[0]:
                #     if targets[0][t].dtype == torch.float32:
                #         targets[0][t] = targets[0][t].type(torch.float16)
                # with profile(
                #     activities=[ProfilerActivity.CUDA],
                #     # on_trace_ready=torch.profiler.tensorboard_trace_handler(
                #     #     "./log/keypoint_rcnn"
                #     # ),
                #     record_shapes=True,
                #     with_stack=True,
                #     use_cuda=True,
                #     profile_memory=True,
                # ) as prof:
                #     with record_function("OUTPUT DICT"):
                output_dict = self(image_batch, targets)

                # for output in output_dict.values():
                #     print(output.dtype)
                #     assert output.dtype is torch.float16

                train_loss = sum(loss for loss in output_dict.values())
                total_train_loss += train_loss.item()

                # assert loss.dtype is torch.float32

                # c = nn.CrossEntropyLoss()
                # loss = c(output, targets)
                output_dict = {k + f" epoch {epoch}": v for k, v in output_dict.items()}

                loss_log = {
                    f"train loss sum epoch {epoch}": train_loss,
                }
                loss_log.update(output_dict)

                self.loss_history.append(train_loss.item())
                if i % 100 == 0:
                    train_loop_batch.set_postfix(loss=train_loss.item())
                    wandb.log(loss_log)

                # if ex_count > 64:

                self.optimizer.zero_grad()
                # scaler.scale(loss).backward()
                train_loss.backward()

                self.optimizer.step()
                ex_count = 0

                if self.scheduler:
                    self.scheduler.step()

            # _, pred = torch.max(output_dict.data, dim=1)
            # correct += (pred == targets).sum().item()
            # accuracy = correct / self.batch_size
            # if self.writer:
            #     self.writer.add_scalar(
            #         f"Loss/train Epoch {epoch}", loss.item(), step
            #     )
            # self.writer.add_scalar(f"batch accuracy {epoch} ", accuracy, step)

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

            # print(
            #     prof.key_averages(group_by_stack_n=5).table(
            #         row_limit=10, sort_by="cuda_memory_usage"
            #     )
            # )

            self.eval()
            for i, (images, targets) in valid_loop_batch:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                with torch.no_grad():
                    outputs = self(images, targets)
                    output_preds = outputs[0]
                    output_losses = outputs[1]

                    valid_loss = sum(loss for loss in output_losses.values())
                    total_valid_loss += valid_loss.item()

                    batch_mae = torch.abs(
                        [t["keypoints"][..., :2] for t in output_preds][0]
                        - [t["keypoints"][..., :2] for t in targets][0]
                    ).mean()

                    if valid_error == None:
                        valid_error = batch_mae.unsqueeze(0)

                    else:
                        valid_error = torch.cat((valid_error, batch_mae.unsqueeze(0)))

                    if i % 100 == 0:
                        valid_error = valid_error.mean()
                        wandb.log({"valid_error (MAE)": valid_error})
                        valid_error = None

            with torch.no_grad():
                avg_train_loss = total_train_loss / len(self.train_dl)
                avg_train_losses.append(avg_train_loss)

                avg_valid_loss = total_valid_loss / len(self.valid_dl)
                avg_valid_losses.append(avg_valid_loss)

            wandb.log({"learning_rate": self.scheduler.get_last_lr()[0]})

            if epoch % save_interval == 0:
                cpkt = {
                    "net": self.state_dict(),
                    "epoch": epoch,
                    # "n_iter": n_iter,
                    "optim": optimizer.state_dict(),
                }
                timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
                ckpt_path = f"checkpoints/model_checkpoint_keypoints_{self.num_keypoints}_epoch_{epoch}_{timestamp}.ckpt"
                save_checkpoint(cpkt, ckpt_path)
                tqdm.write(f"Checkpoint saved for epoch - {ckpt_path}")

        end_timer_and_print("Mixed precision:")

        wandb.log(
            {
                "epoch_losses": wandb.plot.line_series(
                    xs=range(self.epochs),
                    ys=[avg_train_losses, avg_valid_losses],
                    keys=["train_loss_avg", "valid_loss_avg"],
                    title="Loss per Epoch",
                    xname="epoch",
                )
            }
        )

        # update the learning rate
        # torch.save(model.state_dict(), "checkpoints/transfer_learn_added_final_layer.pt")
        # torch.save(model.state_dict(), "checkpoints/transfer_learn_final_layer.pt")
        # model.load_state_dict(torch.load("checkpoints/transfer_learn_final_layer_only.pt"))

    def _test(self, data_loader_test=None, device=None):

        self.eval()
        # self._make_predictions()

        # preds = json.load(open("preds.json", "rb"))
        # preds = [d for d in preds if d["image_id"] in imgs_info]

        self._run_coco_analyze()

    def _run_coco_analyze(self):

        # annFile = "/media/10TB/coco_kp_dataset/data2.json"
        annFile = "/media/10TB/coco_kp_dataset/annotations_trainval2017/annotations/person_keypoints_val2017_fixed.json"
        resFile = "preds.json"
        print("{:10}[{}]".format("annFile:", annFile))
        print("{:10}[{}]".format("resFile:", resFile))

        gt_data = json.load(open(annFile, "rb"))
        imgs_info = {
            i["id"]: {"id": i["id"], "width": i["width"], "height": i["height"]}
            for i in gt_data["images"]
        }

        team_dts = json.load(
            open(
                resFile,
                "rb",
            )
        )
        team_dts = [d for d in team_dts if d["image_id"] in imgs_info]

        print(
            f"Loaded [{len(team_dts)}] instances in [{len(imgs_info)}] images.".format()
        )

        coco_gt = COCO(annFile)

        coco_dt = coco_gt.loadRes(team_dts)

        coco_analyze = COCOanalyze(coco_gt, coco_dt, "keypoints")

        coco_analyze.evaluate(verbose=True, makeplots=True, savedir=".")

    def _make_predictions(self):
        """
        Convenience function for quick testing very specific dataset
        To be removed
        """

        image_path = "/media/10TB/coco_kp_dataset/val2017"
        preds = []

        all_imgs = os.scandir(image_path)
        # all_imgs = os.listdir(image_path)
        for image_file in tqdm(
            all_imgs,
            desc="Images",
            # total=len(all_imgs),
            # position=0,
            leave=False,
        ):
            image = Image.open(image_path + "/" + image_file.name)
            image_tensor = torchvision.transforms.functional.to_tensor(image).cuda()
            image_id = image_file.name.strip("0").replace(".jpg", "")
            image_id = int(image_id)
            self.eval()

            out = self([image_tensor])[0]
            res = [dict(zip(out.keys(), i)) for i in zip(*out.values())]
            res = [{k: v.flatten().tolist() for k, v in i.items()} for i in res]
            res = [dict(i, **{"image_id": image_id}) for i in res]
            res = [dict(i, **{"category_id": 1}) for i in res]

            for r in res:
                r["score"] = r.pop("scores")
                r["score"] = r["score"][0]
                r["labels"] = r["labels"][0]

            preds.extend(res)

        with open("preds.json", "w") as f:
            json.dump(preds, f)
