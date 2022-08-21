from model1.model.model import CustomKeypointRCNN

from matplotlib import pyplot as plt
import numpy as np

from typing import Callable, List, Tuple
import json
from PIL import Image
from tqdm import tqdm

import torchvision
import torch
import os
import cv2


class KeypointPredictor:
    def __init__(self) -> None:
        self.model = CustomKeypointRCNN(num_classes=2, num_keypoints=23)
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to("cuda")
        self.model.eval()

        self.video, self.image = None, None

        self._load_model()

    @property
    # specify the args and return type of callable properly
    def results(self, cb: Callable[[int], int] = None) -> dict:
        # do generic stuff that image and video need
        if self.image:
            reults = self.plot_keypoints(self.image, connect_skeleton)
        # if self.video:
        results = self.video_plot(self.video)
        # else:
        #     ValueError

        # if cb and hasattr(
        #     cb,
        # ):
        #     results = cb()
        # else:
        #     results = None

        return results

    def video_plot(self, path):
        vid = cv2.VideoCapture(path)

        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vid.get(cv2.CAP_PROP_FPS)
        num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(self.video)
        out_name = basename.split(".")
        print(out_name)
        output_file = cv2.VideoWriter(
            filename=f"{out_name[0]}_keypoints.mlv",
            # some installation of opencv may not support x264 (due to its license),
            # you can try other format (e.g. MPEG)
            fourcc=cv2.VideoWriter_fourcc(*"MPEG"),
            # fourcc=cv2.VideoWriter_fourcc(*"x264"),
            fps=float(fps),
            frameSize=(width, height),
            isColor=True,
        )
        frames = self._gen_video_frames(vid)

        for frame in frames:

            image = Image.fromarray(frame)
            image = torchvision.transforms.functional.to_tensor(frame).cuda()
            image.unsqueeze_(0)

            out = self.model(image)[0]

            self.overlay_keypoints(out, frame)

            output_file.write(frame)
            cv2.namedWindow("keypoint_viz", cv2.WINDOW_NORMAL)
            cv2.imshow("keypoint_viz", frame)
            if cv2.waitKey(1) == 27:
                break  # esc key

        cv2.destroyAllWindows()
        return None

    def overlay_keypoints(self, out, frame):
        if out["keypoints"].shape[0] > 0:
            best = torch.argmax(out["keypoints_scores"].sum(dim=1))
            x = out["keypoints"][best, :, 0].cpu().detach()
            y = out["keypoints"][best, :, 1].cpu().detach()

            frame_mask = frame.copy()
            alpha = 0.5

            for i, kp in enumerate(x):
                cv2.circle(
                    frame,
                    (int(x[i].item()), int(y[i].item())),
                    2,
                    (0, 255, 255),
                    -1,
                )

            if connect_skeleton:
                for connection in connect_skeleton:
                    start_pt_x = int(x[connection[0]].item())
                    start_pt_y = int(y[connection[0]].item())

                    end_pt_x = int(x[connection[1]].item())
                    end_pt_y = int(y[connection[1]].item())

                    cv2.line(
                        frame_mask,
                        (start_pt_x, start_pt_y),
                        (end_pt_x, end_pt_y),
                        (0, 165, 255),
                        2,
                    )
                cv2.addWeighted(frame_mask, alpha, frame, 1 - alpha, 0, frame)

    def _gen_video_frames(self, vid):
        while vid.isOpened():
            success, frame = vid.read()
            if success:
                yield frame
            else:
                break

    def _load_model(self):
        print("Loading Model...")

        state_dict = torch.load(
            "/home/dan/Project_APE/krcnn_experiments/model1/checkpoints/model_checkpoint_keypoints_23_epoch_3_2022-08-08-1634.ckpt"
        )
        self.model.load_state_dict(state_dict["net"])

    def plot_keypoints(
        self,
        image_file: str,
        connect_skeleton: List[Tuple[int, int]] = True,
        save_path: str = None,
    ) -> None:
        """
        Args:

        Params:

        Returns: None

        Skeleton connections:
            nose -> left_eye -> left_ear. (0, 1), (1, 3)
            nose -> right_eye -> right_ear. (0, 2), (2, 4)
            nose -> left_shoulder -> left_elbow -> left_wrist. (0, 5), (5, 7), (7, 9)
            nose -> right_shoulder -> right_elbow -> right_wrist. (0, 6), (6, 8), (8, 10)
            left_shoulder -> left_hip -> left_knee -> left_ankle. (5, 11), (11, 13), (13, 15)
            right_shoulder -> right_hip -> right_knee -> right_ankle. (6, 12), (12, 14), (14, 16)
            left_shoulder -> right_shoulder
            left_hip -> right_hip
            left_small_toe -> left_heel
            left_big_toe -> left_heel
            left_heel -> left_ankle
            right_small_toe -> right_heel
            right_big_toe -> right_heel
            right_heel -> right_ankle
        """
        image = Image.open(image_file)
        image_tensor = torchvision.transforms.functional.to_tensor(image).cuda()
        image_tensor.unsqueeze_(0)
        out = self.model(image_tensor)[0]
        # detect_threshold = 0.75
        # idx = torch.where(scores > detect_threshold)
        # keypoints = kpts[idx]

        result_image = np.array(image.copy())
        print(out)
        print(out["keypoints_scores"].shape)
        print(out["keypoints_scores"].sum(dim=1))
        best = torch.argmax(out["keypoints_scores"].sum(dim=1))
        print(best)
        x = out["keypoints"][best, :, 0].cpu().detach()
        y = out["keypoints"][best, :, 1].cpu().detach()

        plt.figure(figsize=(20, 15))
        plt.scatter(x, y, c="yellow")
        plt.imshow(result_image)

        if connect_skeleton:
            for connection in connect_skeleton:
                start_pt_x = x[connection[0]]
                start_pt_y = y[connection[0]]

                end_pt_x = x[connection[1]]
                end_pt_y = y[connection[1]]

                plt.plot((start_pt_x, end_pt_x), (start_pt_y, end_pt_y), color="orange")
        plt.show()
        if save_path:
            plt.savefig(save_path)


connect_skeleton = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (11, 12),
    (5, 11),
    (6, 12),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
    (17, 19),
    (18, 19),
    (19, 15),
    (20, 22),
    (21, 22),
    (22, 16),
]

coco_keypoints = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_big toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
]
