import argparse
from pathlib import Path
import cv2
from inference.predictor import KeypointPredictor

# TODO
# - output raw keypoint data into file
# - refactoring eg image and video processing should use same function per frame


def get_args():
    parser = argparse.ArgumentParser(
        description="Demo for pose estimation application."
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help=f"Specify a video path file.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help=f"Specify a image path file.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=f"Specify a image path file.",
    )

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = get_args()

    # assert opt.image or opt.video

    predictor = KeypointPredictor()
    predictor.image, predictor.video = opt.image, opt.video

    # do image visualisation + analysis
    results = predictor.results

    # save output

    # do video visualisation + analysis
    # save output


# def get_results(cb=None):

# def get_files(path, extensions=None, recurse=False, include=None):
#     path = Path(path)
#     extensions = setify(extensions)
#     extensions = {e.lower() for e in extensions}
#     if recurse:
#         res = []
#         for i, (p, d, f) in enumerate(
#             os.walk(path)
#         ):  # returns (dirpath, dirnames, filenames)
#             if include is not None and i == 0:
#                 d[:] = [o for o in d if o in include]
#             else:
#                 d[:] = [o for o in d if not o.startswith(".")]
#             res += _get_files(p, f, extensions)
#         return res
#     else:
#         f = [o.name for o in os.scandir(path) if o.is_file()]
#         return _get_files(path, f, extensions)
