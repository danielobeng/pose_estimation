import torch

import wandb


def activation_hook(name, inst, inp, out):
    with torch.no_grad():
        if isinstance(out, torch.Tensor):
            stats = {
                f"(activations) {name}_mean": out.mean().item(),
                f"(activations) {name}_std": out.std().item(),
                f"(activations) {name}_near_zero": (out <= 0.05).long().sum().item()
                / out.numel(),
            }
    return stats


def gradients_hook(name, grad_in, grad_out):
    with torch.no_grad():
        if isinstance(out, torch.Tensor) and model.log_batch:
            stats = {
                f"(gradients) {name}_mean": out.mean().item(),
                f"(gradients) {name}_std": out.std().item(),
                f"(gradients) {name}_near_zero": (out <= 0.05).long().sum().item()
                / out.numel(),
            }
    return stats

    # l = [
    #     module
    #     for module in model.named_modules()
    #     if not isinstance(module, torch.nn.Sequential)
    # ]

    # named_layers = dict(model.named_modules())

    # layers = [
    #     "backbone.body.conv1",
    #     "backbone.fpn.layer_blocks.3.0",
    #     "roi_heads.box_head.fc7",
    #     "roi_heads.box_predictor.cls_score",
    #     "roi_heads.box_predictor.bbox_pred",
    #     "roi_heads.keypoint_head.0",
    #     "roi_heads.keypoint_head.1",
    #     "roi_heads.keypoint_head.14",
    #     "roi_heads.keypoint_head.15",
    #     "roi_heads.keypoint_predictor.kps_score_lowres",
    # ]

    # hook_list = []
    # for l in layers:
    #     hook_list.append(
    #         get_module_by_name(model, l).register_forward_hook(
    #             partial(activation_hook, l)
    #         )
    #     )
    #     hook_list.append(
    #         get_module_by_name(model, l).register_backward_hook(
    #             partial(activation_hook, l)
    #         )
    #     )
