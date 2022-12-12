import json
import os
from functools import partial
from typing import Union

import albumentations as A
import numpy as np
import torch
import torchvision.transforms as T
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pycocotools.coco import COCO
from sklearn import datasets
from torch.utils.data import Dataset


class FootKeypointsDataset(Dataset):
    """
    A class to represent custom COCO dataset with foot annotations from https://cmu-perceptual-computing-lab.github.io/foot_keypoint_dataset/.

    Attributes
    ----------
    img_path:
        Specifies the number of classes used in the model.
    num_keypoints:
        Specifies the number of keypoints used in the model.
    mixed_precision:
        [WIP] Run the model in mixed precision training

    Methods
    -------
    _train():

    _test():

    """

    def __init__(
        self,
        img_path: Union[str, os.PathLike] = None,
        ann_path: Union[str, os.PathLike] = None,
        transforms=None,
        num_keypoints: int = 17,
    ) -> None:
        """
        Construct the Dataset object for keypoint annotations

        Parameters
        ----------
            img_path (str):
                Path to the image dataset

            ann_path (str):
                Path to json file containing COCO format annotations

            transforms:
                dataset transforms

            num_keypoints:
                number of keypoints labelled in the dataset

        Returns
        -------
        None
        """
        self.img_path = img_path
        self.ann_path = ann_path
        self.transforms = transforms
        self.num_keypoints = num_keypoints

        self.cocoanns = COCO(self.ann_path)
        self.annids = self.cocoanns.getAnnIds()
        self.anns = self.cocoanns.loadAnns(self.annids)
        self.ids = list(sorted(self.cocoanns.anns.keys()))

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Union[torch.tensor, torch.tensor]:
        """
        Retrieve indexed the training example from the dataset

        Parameters
        ----------
        idx:
            index number to be retrieved from the dataset

        Returns
        -------
        image tensor
        label tensor for the given index


        Note
        ----
        boxes:
        Bounding boxes for objects
        In coco format, bbox = [xmin, ymin, width, height]
        In pytorch, the input should be [xmin, ymin, xmax, ymax]
        """
        id = self.ids[idx]
        img_id = self.cocoanns.anns[id]["image_id"]
        path = self.cocoanns.loadImgs(img_id)[0]["file_name"]
        img = Image.open(os.path.join(self.img_path, path)).convert("RGB")

        # TODO put all this in a custom transform class
        keypoints = torch.as_tensor(
            self.cocoanns.anns[id]["keypoints"], dtype=torch.float
        )
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

        # there is only ever 1 class (person), which is class 1
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

        if self.transforms:
            img, target["keypoints"][..., :2] = get_transforms(img, target)
        else:
            img = T.transforms.ToTensor()(img)
        self.img, self.target = img, target
        return img, target

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        body += [f"Images Path: {self.img_path}"]
        body += [f"Annotation Path: {self.ann_path}"]
        if hasattr(self, "transforms") and self.transforms is not None:
            body += ["Transforms:" + repr(self.transforms)]
        lines = [head] + [" " + line for line in body]
        return "\n".join(lines)


def get_data(
    img_path,
    ann_path,
    num_keypoins: int = 23,
    batch_size: int = 1,
    num_workers: int = 8,
    dummy_data=None,
    transform: bool = False,
    pin_memory: bool = False,
    is_test: bool = False,
):
    # TODO put all this in a custom transform class
    # NOTE not sure where to include this info but the categories section of the defaul foot train dataset
    # is not wrapped in a list like original train dataset so COCO throws error, needs to be wrapped in a list first

    data = FootKeypointsDataset(
        img_path,
        ann_path,
        transforms=get_transforms if transform else False,
        num_keypoints=num_keypoins,
    )

    indices = torch.randperm(len(data)).tolist()
    num_valid = int(len(data) * 0.05)

    make_loader_p = partial(
        make_loader,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    if is_test:
        test_data = torch.utils.data.Subset(data, indices[:])
        test_loader = make_loader_p(test_data, batch_size * 2, shuffle=True)
        return test_data, test_loader

    else:
        train_data = torch.utils.data.Subset(data, indices[:-num_valid])
        valid_data = torch.utils.data.Subset(data, indices[-num_valid:])
        if dummy_data:
            train_data = torch.utils.data.Subset(train_data.dataset, indices[:11])
            valid_data = torch.utils.data.Subset(valid_data.dataset, indices[:11])

        valid_data_coco_indices = (
            torch.tensor(valid_data.dataset.annids)[valid_data.indices],
        )

        train_loader = make_loader_p(train_data, batch_size, shuffle=True)
        valid_loader = make_loader_p(train_data, batch_size * 2, shuffle=True)

        return (
            train_data,
            train_loader,
            valid_data,
            valid_loader,
            valid_data_coco_indices,
        )


def collate_fn(batch):
    return tuple(zip(*batch))


def make_loader(dataset, batch_size, shuffle, num_workers, collate_fn, pin_memory):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    return loader


def get_transforms(img, targets, train=False):
    """
    Construct and return the transformations to be performed on the dataset object

    Parameters
    ----------

    Returns
    -------
    """
    img = np.array(img)
    keypoints = targets.get("keypoints")
    keypoints = keypoints[0, :, :2]
    keypoints = torch.split(keypoints, 1)
    keypoints = [k.squeeze(0) for k in keypoints]
    transforms = A.Compose(
        [
            # random crop removes keypoints that are cropped out
            # reducing dimension along the axis of n keypoints
            # A.RandomCrop(height=360, width=360, p=0.9),
            # A.RandomSizedCrop(min_max_height=(256, 1025), height=512, width=512, p=0.9),
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.RandomRotate90(p=0.5),
            A.OneOf([A.HueSaturationValue(p=0.9), A.RGBShift(p=0.9)], p=1),
            A.RandomBrightnessContrast(p=0.5),
            A.ToGray(p=0.6),
            # A.Normalize(mean=[0,0,0])
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy"),
    )

    transformed = transforms(image=img, keypoints=keypoints)
    transformed["image"] = T.transforms.ToTensor()(transformed["image"])
    transformed["keypoints"] = torch.tensor(transformed.get("keypoints"))

    return transformed["image"], transformed["keypoints"]
