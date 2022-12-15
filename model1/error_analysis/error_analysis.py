import json
import logging
import os
from typing import Union

from cocoanalyze.pycocotools.coco import COCO
from cocoanalyze.pycocotools.cocoanalyze import COCOanalyze


def run_coco_analyze(
    resFile: Union[str, os.PathLike] = None, annFile: Union[str, os.PathLike] = None
) -> None:

    """
    Perfom eorror analysis on the results file from the keypoin model's predictions by comparing the results to the original annotations.

    Parameters
    ----------
    resFile:
        Results file path - contains the predictions

    annFile:
        Annotation file - contains the ground truth labels for comparison to the results file.

    Returns
    -------
    None
    """
    if annFile == None:
        ValueError("annFile (annotation file) must be specified.")

    print(f"{'annFile':10}[{annFile}]")
    print(f"{'resFile':10}[{resFile}]")

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

    # gt = [i["image_id"] for i in gt_data["annotations"]]
    accepted_img_ids = [i["image_id"] for i in team_dts]

    logging.info(f"Loaded [{len(team_dts)}] instances in [{len(imgs_info)}] images.")

    coco_gt = COCO(annFile, accepted_img_ids)
    # coco_gt = COCO(gt_data, acc)

    coco_dt = coco_gt.loadRes(team_dts)

    coco_analyze = COCOanalyze(coco_gt, coco_dt, "keypoints")

    coco_analyze.evaluate(verbose=True, makeplots=False, savedir="error_analysis")

    coco_analyze.params.oksThrs = [
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
    ]

    # set OKS threshold required to match a detection to a ground truth
    coco_analyze.params.oksLocThrs = 0.1

    # set KS threshold limits defining jitter errors
    coco_analyze.params.jitterKsThrs = [0.5, 0.85]

    # set the localization errors to analyze and in what order
    # note: different order will show different progressive improvement
    # to study impact of single error type, study in isolation
    coco_analyze.params.err_types = ["miss", "swap", "inversion", "jitter"]

    # # area ranges for evaluation
    # # 'all' range is union of medium and large
    # coco_analyze.params.areaRng = [
    #     [32**2, 1e5**2]
    # ]  # [96 ** 2, 1e5 ** 2],[32 ** 2, 96 ** 2]
    # coco_analyze.params.areaRngLbl = ["all"]  # 'large','medium'

    # coco_analyze.params.maxDets = [20]

    # input arguments:
    #  - check_kpts   : analyze keypoint localization errors for detections with a match (default: True)
    #                 : default errors types are ['jitter','inversion','swap','miss']
    #  - check_scores : analyze optimal score (maximizing oks over all matches) for every detection (default: True)
    #  - check_bkgd   : analyze background false positives and false negatives (default: True)

    coco_analyze.analyze(check_kpts=True, check_scores=True, check_bckgd=True)
    coco_analyze.summarize(makeplots=True, savedir="error_analysis")

    for stat in coco_analyze.stats:
        print(stat)

    corrected_dts = coco_analyze.corrected_dts["all"]

    i = 23
    # info on keypoint detection localization error
    # logging.info(f"good: {corrected_dts[i]['good']}")
    # logging.info(f"miss: {corrected_dts[i]['miss']}")
    # logging.info(f"swap: {corrected_dts[i]['swap']}")
    # logging.info(f"inv.: {corrected_dts[i]['inversion']}")
    # logging.info(f"jit.: {corrected_dts[i]['jitter']}\n")

    # corrected keypoint locations
    logging.info(f"predicted keypoints:\n {corrected_dts[i]['keypoints']}")
    logging.info(f"corrected keypoints:\n {corrected_dts[i]['opt_keypoints']}\n")

    # optimal detection score
    logging.info(f"original score: {corrected_dts[i]['score']}")
    logging.info(f"optimal score:  {corrected_dts[i]['opt_score']}\n")

    ## after summarize() has been called the following variables are available

    # list of the false positive detections and missed ground-truth annotations
    false_pos_dts = coco_analyze.false_pos_dts
    false_neg_gts = coco_analyze.false_neg_gts
    for oks in coco_analyze.params.oksThrs:
        logging.info(
            "Oks:[%.2f] - Num.FP:[%d] - Num.FN:[%d]"
            % (
                oks,
                len(false_pos_dts["all", str(oks)]),
                len(false_neg_gts["all", str(oks)]),
            )
        )
