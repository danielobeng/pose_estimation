#%%
import json
import os
import itertools
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt
from matplotlib import patches
from collections import namedtuple
from pycocotools.coco import COCO
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torchvision.transforms as T
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from engine import train_one_epoch, evaluate
import utils


class FootKeypointsDataset(Dataset):
    """
    Custom COCO dataset with foot annotations from https://cmu-perceptual-computing-lab.github.io/foot_keypoint_dataset/
    """

    def __init__(
        self, dataset_dir=None, json_file=None, transforms=None,
    ):
        """
        Args:
            dataset_dir (str): Path to the image dataset
            json_file (str): Path to json file containing COCO format annotations 
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dataset_dir = dataset_dir
        self.json_file = json_file
        # self.transforms = transforms
        self.transforms = None

        self.cocoanns = COCO(self.json_file)
        self.annids = self.cocoanns.getAnnIds()
        self.anns = self.cocoanns.loadAnns(self.annids)

        self.ids = list(sorted(self.cocoanns.imgs.keys()))

        with open(self.json_file, "r") as f:
            self.label_file = json.load(f)

        # annots = self.label_file["annotations"]

        # self.keypoints = [l["keypoints"] for l in annots]
        # keypoints_x = torch.tensor(
        #     [[k[l] for l in range(len(k)) if l % 3 == 0] for k in self.keypoints]
        # )
        # keypoints_y = torch.tensor(
        #     [[k[l] for l in range(len(k)) if l % 3 == 1] for k in self.keypoints]
        # )
        # keypoints_v = torch.tensor(
        #     [[k[l] for l in range(len(k)) if l % 3 == 2] for k in self.keypoints]
        # )
        # self.keypoints = torch.stack((keypoints_x, keypoints_y, keypoints_v)).permute(
        #     1, 2, 0
        # )

        # self.img_ids = [l["image_id"] for l in annots]
        # self.img_list = [
        #     f'{self.dataset_dir}/{"0" * (12 - len(str(i)))}{i}.jpg'
        #     for i in self.img_ids
        # ]

    def __len__(self) -> int:
        return len(self.ids)
        # return len(self.anns)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        path = self.cocoanns.loadImgs(img_id)[0]["file_name"]
        # path = f'{"0" * (12 - len(str(img_id)))}{img_id}.jpg'
        img = Image.open(os.path.join(self.dataset_dir, path)).convert("RGB")

        # img = Image.open(self.img_list[idx]).convert("RGB")
        img = T.transforms.ToTensor()(img)
        # plt.imshow(img)
        # plt.show()
        # keypoints = self.keypoints[idx]
        # keypoints = torch.as_tensor(keypoints, dtype=torch.unint8)
        keypoints = torch.as_tensor(self.anns[idx]["keypoints"], dtype=torch.float)
        keypoints = torch.reshape(keypoints, (1, 17, 3))
        masks = self.cocoanns.annToMask(self.anns[idx])
        masks = torch.as_tensor(masks, dtype=torch.uint8).unsqueeze(1)
        masks = masks.permute(1, 0, 2)
        # print(keypoints.shape)

        obj_ids = np.unique(masks)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        bbox = self.anns[idx]["bbox"]
        num_objs = len(obj_ids)
        # for i in range(num_objs):
        # pos = np.where(masks[i])
        # xmin = bbox[0]
        # ymin = bbox[1]
        # xmax = xmin + bbox[2]
        # ymax = ymin + bbox[3]
        # boxes.append([xmin, ymin, xmax, ymax])
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = xmin + bbox[2]
        ymax = ymin + bbox[3]
        boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # print(boxes)
        # print(boxes.shape)

        # there is only 1 class - person, which is 1
        # num_objs = len(self.img_list)
        labels = torch.ones((1), dtype=torch.int64)

        image_id = self.anns[idx]["image_id"]
        image_id = torch.tensor(image_id)

        area = torch.tensor(boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # area = (
        #     torch.tensor(boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # ).to(self.device)
        # suppose all instances are not crowd
        # iscrowd = torch.zeros((1,), dtype=torch.int64)
        # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        # area = torch.as_tensor([self.anns[idx]["area"]], dtype=torch.float32)
        iscrowd = torch.as_tensor([self.anns[idx]["iscrowd"]], dtype=torch.int64)
        num_keypoints = torch.tensor(self.anns[idx]["num_keypoints"], dtype=torch.int64)
        target = {}
        target["labels"] = labels
        target["keypoints"] = keypoints
        target["boxes"] = boxes
        # target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["num_keypoints"] = num_keypoints
        # target["id"] = self.ids[idx]

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


# dataset = FootKeypointsDataset(
#     "COCO_images/val2017", "COCO_annotations/person_keypoints_val2017.json",
# )
# print(dataset)
# train_dl = DataLoader(d , batch_size=100)


# load a model pre-trained pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 2  # 1 class (person) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# load a pre-trained model for classification and return
# only the features
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# FasterRCNN needs to know the number of
# output channels in a backbone. For mobilenet_v2, it's 1280
# so we need to add it here
backbone.out_channels = 1280

# let's make the RPN generate 5 x 3 anchors per spatial
# location, with 5 different sizes and 3 different aspect
# ratios. We have a Tuple[Tuple[int]] because each feature
# map could potentially have different sizes and
# aspect ratios
anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
)

# let's define what are the feature maps that we will
# use to perform the region of interest cropping, as well as
# the size of the crop after rescaling.
# if your backbone returns a Tensor, featmap_names is expected to
# be [0]. More generally, the backbone should return an
# OrderedDict[Tensor], and in featmap_names you can choose which
# feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=[0], output_size=7, sampling_ratio=2
)

# put the pieces together inside a FasterRCNN model
model = FasterRCNN(
    backbone,
    num_classes=2,
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler,
)


class CustomCocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self, root, annFile, transform=None, target_transform=None, transforms=None
    ):
        super().__init__(root, annFile, transform, transforms=None)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]["file_name"]

        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def get_model_keypoints(num_classes):
    # keypoint_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    #     featmap_names=["0"], output_size=14, sampling_ratio=2
    # )
    # model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
    #     pretrained=False, keypoint_roi_pool=keypoint_roi_pooler
    # )
    # num_kp = 6
    num_kp = 17
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False)
    out = nn.ConvTranspose2d(
        512, num_kp, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)
    )
    model.roi_heads.keypoint_predictor.kps_score_lowres = out

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = FootKeypointsDataset(
        # "COCO_images/train2017",
        # "COCO_annotations/person_keypoints_train2017.json",
        # "COCO_annotations/person_keypoints_val2017.json",
        "/media/10TB/coco_kp_dataset/val2017",
        "/media/10TB/coco_kp_dataset/annotations_trainval2017/annotations/person_keypoints_val2017.json",
        # "COCO_images/val2017",
        # "COCO_annotations/person_keypoints_val2017_foot_v1.json",
        get_transform(train=True),
    )
    dataset_test = FootKeypointsDataset(
        # "COCO_images/train2017",
        # "COCO_annotations/person_keypoints_train2017.json",
        # "COCO_images/val2017",
        # "COCO_annotations/person_keypoints_val2017_foot_v1.json",
        "/media/10TB/coco_kp_dataset/val2017",
        "/media/10TB/coco_kp_dataset/annotations_trainval2017/annotations/person_keypoints_val2017.json",
        # "COCO_annotations/person_keypoints_val2017.json",
        get_transform(train=False),
    )

    # dataset = CustomCocoDetection(
    #     # "COCO_images/val2017",
    #     # "COCO_annotations/person_keypoints_val2017.json",
    #     "COCO_images/train2017",
    #     "COCO_annotations/person_keypoints_train2017.json",
    #     transform=transforms.ToTensor(),
    # )
    # dataset_test = CustomCocoDetection(
    #     # "COCO_images/val2017",
    #     # "COCO_annotations/person_keypoints_val2017.json",
    #     "COCO_images/train2017",
    #     "COCO_annotations/person_keypoints_train2017.json",
    #     transform=transforms.ToTensor(),
    # )

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=1, collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    # get the model using our helper function
    # model = get_model_instance_segmentation(num_classes)
    model = get_model_keypoints(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 1

    # fit(num_epochs, model, loss, optimizer, data_loader, data_loader_test)

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    # torch.save(model.state_dict(), "keypoints_test1_model_weights.pt")
    torch.save(model, "models/entire_model3")


# def to_tensor(img):
#     # img = Image.open(self.img_list[idx]).convert("RGB")
#     img = transforms.ToTensor()(img)
#     return img, target


def test_model(img_path, model, kp=True):
    model.eval()
    img = Image.open(img_path)
    img = transforms.ToTensor()(img).to("cuda")
    pred = model([img])
    if kp:
        print(pred)
        print(pred[0].shape)
        # plt.scatter([])

        plt.imshow(img.permute(1, 2, 0).detach().cpu().numpy())
    else:
        img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
        pred_img = Image.fromarray(pred[0]["masks"][0, 0].mul(255).byte().cpu().numpy())
        return img, pred_img


main()
# model = torch.load("models/entire_model3")
# test_model("runner2.jpg", model)

# %%
# # model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
# # model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False)
# # weights = torch.load(
# #     # "keypoints_test1_model_weights.pt", map_location=torch.device("cpu")
# #     "test1_model_weights.pt",
# #     map_location=torch.device("cpu"),
# # )
# # model.load_state_dict(weights)
# model = torch.load("entire_model2", map_location=torch.device("cpu"))
# # keypoint_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
# #     featmap_names=["0"], output_size=14, sampling_ratio=2
# # )
# # model.roi_heads.keypoint_roi_pool = keypoint_roi_pooler
# model.eval()
# img = Image.open("runner2.jpg")
# img = transforms.ToTensor()(img)
# pred = model([img])
# kp = pred[0]["keypoints"]
# plt.scatter(kp[0][:, 0].detach().numpy(), kp[0][:, 1].detach().numpy())
# plt.imshow(img.permute(1, 2, 0))
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
# print(kp.shape)
# print(kp)
# plt.show()

# model([torch.rand(3, 300, 400), torch.rand(3, 300, 400)])

# %%


# %%
# if not exists('keypoint.py'):
#   !wget https://raw.githubusercontent.com/facebookresearch/maskrcnn-benchmark/e0a525a0139baf7086117b7ed3fd318a4878d71c/maskrcnn_benchmark/structures/keypoint.py

from keypoint import PersonKeypoints
import cv2


def vis_keypoints(img, kps, kp_thresh=2, alpha=0.7):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
    """
    dataset_keypoints = PersonKeypoints.NAMES
    kp_lines = PersonKeypoints.CONNECTIONS

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw mid shoulder / mid hip first for better visualization.
    mid_shoulder = (
        kps[:2, dataset_keypoints.index("right_shoulder")]
        + kps[:2, dataset_keypoints.index("left_shoulder")]
    ) / 2.0
    sc_mid_shoulder = np.minimum(
        kps[2, dataset_keypoints.index("right_shoulder")],
        kps[2, dataset_keypoints.index("left_shoulder")],
    )
    mid_hip = (
        kps[:2, dataset_keypoints.index("right_hip")]
        + kps[:2, dataset_keypoints.index("left_hip")]
    ) / 2.0
    sc_mid_hip = np.minimum(
        kps[2, dataset_keypoints.index("right_hip")],
        kps[2, dataset_keypoints.index("left_hip")],
    )
    nose_idx = dataset_keypoints.index("nose")
    if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
        cv2.line(
            kp_mask,
            tuple(mid_shoulder),
            tuple(kps[:2, nose_idx]),
            color=colors[len(kp_lines)],
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
        cv2.line(
            kp_mask,
            tuple(mid_shoulder),
            tuple(mid_hip),
            color=colors[len(kp_lines) + 1],
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    # Draw the keypoints.
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = kps[0, i1], kps[1, i1]
        p2 = kps[0, i2], kps[1, i2]
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2, color=colors[l], thickness=2, lineType=cv2.LINE_AA
            )
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask,
                p1,
                radius=3,
                color=colors[l],
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask,
                p2,
                radius=3,
                color=colors[l],
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def overlay_keypoints(image, kps, scores):
    kps = torch.cat((kps[:, :, 0:2], scores[:, :, None]), dim=2).cpu().numpy()
    for region in kps:
        image = vis_keypoints(image, region.transpose((1, 0)))
    return image


# %%
def plot_keypoints(model, image_file):
    image = Image.open(image_file)
    image_tensor = torchvision.transforms.functional.to_tensor(image).cuda()
    out = model([image_tensor])[0]
    result_image = np.array(image.copy())
    result_image = overlay_keypoints(
        result_image,
        out["keypoints"].cpu().detach(),
        out["keypoints_scores"].cpu().detach(),
    )
    plt.figure(figsize=(20, 15))
    plt.imshow(result_image)


# %%
