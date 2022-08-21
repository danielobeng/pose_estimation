from pycocotools.coco import COCO
from sklearn import datasets
import torch
from torch.utils.data import Dataset
from PIL import Image

import json
import numpy as np

import os
import torchvision.transforms as T


class FootKeypointsDataset(Dataset):
    """
    Custom COCO dataset with foot annotations from https://cmu-perceptual-computing-lab.github.io/foot_keypoint_dataset/
    """

    def __init__(
        self, dataset_dir=None, json_file=None, transforms=None, num_keypoints=17
    ):
        """
        Args:
            dataset_dir (str): Path to the image dataset
            json_file (str): Path to json file containing COCO format annotations 
        """
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dataset_dir = dataset_dir
        self.json_file = json_file
        # self.transforms = transforms
        self.transforms = None
        self.num_keypoints = 17

        self.cocoanns = COCO(self.json_file)
        self.annids = self.cocoanns.getAnnIds()
        self.anns = self.cocoanns.loadAnns(self.annids)

        # self.ids = list(sorted(self.cocoanns.imgs.keys()))
        self.ids = list(sorted(self.cocoanns.anns.keys()))
        self.cocoanns.getImgIds()

        with open(self.json_file, "r") as f:
            self.label_file = json.load(f)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        """
        boxes:
        Bounding boxes for objects
        In coco format, bbox = [xmin, ymin, width, height]
        In pytorch, the input should be [xmin, ymin, xmax, ymax]
        """
        id = self.ids[idx]
        img_id = self.cocoanns.anns[id]["image_id"]
        path = self.cocoanns.loadImgs(img_id)[0]["file_name"]

        img = Image.open(os.path.join(self.dataset_dir, path)).convert("RGB")
        img = T.transforms.ToTensor()(img)

        keypoints = torch.as_tensor(
            self.cocoanns.anns[id]["keypoints"], dtype=torch.float
        )
        # TODO the 6 needs to be a variable dependant on number of keypoitns in dataset
        # maybe some sort of check that it matches and warning if not
        # assert self.num_keypoints ==
        num_ann_keypoints = keypoints.shape[0] // 3
        keypoints = torch.reshape(keypoints, (1, num_ann_keypoints, 3))

        boxes = []
        bbox = self.cocoanns.anns[id]["bbox"]
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = xmin + bbox[2]
        ymax = ymin + bbox[3]
        boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # there is only ever 1 class - person, which is 1 hence the hard coding
        labels = torch.ones((1), dtype=torch.int64)

        area = torch.as_tensor(boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.as_tensor(
            [self.cocoanns.anns[id]["iscrowd"]], dtype=torch.int64
        )
        num_keypoints = torch.tensor(
            self.cocoanns.anns[id]["num_keypoints"], dtype=torch.int64
        )
        target = {}
        target["id"] = torch.as_tensor(id)
        target["image_id"] = torch.as_tensor(img_id)
        target["labels"] = labels
        target["keypoints"] = keypoints
        target["boxes"] = boxes
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["num_keypoints"] = num_keypoints

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # img = img.half()
        return img, target

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)


def get_data(num_keypoins=17, batch_size=1, num_workers=8, dummy_data=None):
    # NOTE not sure where to include this info but the categories section of the defaul foot train dataset
    # is not wrapped in a list like original train dataset so COCO throws error, needs to be wrapped in a list first
    data = FootKeypointsDataset(
        "/media/10TB/coco_kp_dataset/train2017",
        "/media/10TB/coco_kp_dataset/person_keypoints_train2017_foot_v1.json",
        get_transform(train=True),
        num_keypoints=num_keypoins,
    )

    indices = torch.randperm(len(data)).tolist()
    train_data = torch.utils.data.Subset(data, indices[:-500])
    valid_data = torch.utils.data.Subset(data, indices[-500:])

    if dummy_data:
        train_data = torch.utils.data.Subset(data, indices[:10])
        valid_data = torch.utils.data.Subset(data, indices[-10:])

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn,
    )

    return train_data, train_loader, valid_data, valid_loader


def collate_fn(batch):
    return tuple(zip(*batch))


def get_transform(train):
    # These transformations are required when using pre-trained pytorch models to ensure
    # transfer learning works as requires the same pre-processing
    transforms = []
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        transforms.extend(
            [
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize,
            ]
        )
    # if valid:
    #     transforms.extend([T.Scale(256), T.CenterCrop(224), T.ToTensor, normalize])

    return T.Compose(transforms)

