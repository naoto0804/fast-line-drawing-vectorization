# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch.nn as nn

from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.layers.non_local import init_nl_module
from .generalized_rcnn import GeneralizedRCNN
import torch
from maskrcnn_benchmark.layers import CoordConv2d
from torch.nn.parameter import Parameter

_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}


def build_detection_model(cfg):
    # Define and load the original model
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    model = meta_arch(cfg)
    dummy_checkpointer = DetectronCheckpointer(cfg, model)
    dummy_checkpointer.load(cfg.MODEL.WEIGHT)

    if cfg.MODEL.BACKBONE.COORDS:
        module_dict = {
            "input": {"parent": model.backbone.body.stem, "name": "conv1"},
            "rpn_input": {"parent": model.rpn.head, "name": "conv"}}
            # "rpn_input": {"parent": model.rpn.head, "name": "conv"}}
        # }
        for identifier in cfg.MODEL.BACKBONE.COORDS:
            if identifier not in module_dict.keys():
                continue
            parent_module = module_dict[identifier]["parent"]
            name = module_dict[identifier]["name"]
            old_conv = getattr(parent_module, name)
            out_ch, in_ch, h, w = old_conv.weight.shape

            new_weight = torch.cat([old_conv.weight,
                torch.zeros([out_ch, 2, h, w], dtype=torch.float32)], dim=1)
            kwargs = {"with_r": False}
            for key in ["in_channels", "out_channels", "kernel_size", "stride",
                        "padding", "dilation", "groups"]:
                kwargs[key] = getattr(old_conv, key)
            if old_conv.bias is None:
                kwargs["bias"] = False
            else:
                kwargs["bias"] = True

            # https://discuss.pytorch.org/t/how-can-i-modify-certain-layers-weight-and-bias/11638/3
            new_conv = CoordConv2d(**kwargs)
            new_conv.conv.state_dict()["weight"].copy_(new_weight)
            if old_conv.bias is not None:
                new_conv.conv.state_dict()["bias"].copy_(old_conv.bias.data)

            delattr(parent_module, name)
            setattr(parent_module, name, new_conv)
            print("Replace", old_conv, "to", new_conv)

    # insert non-local block just before the last block of res4 (layer3)
    # if cfg.MODEL.BACKBONE.NON_LOCAL != "":
    #     nl_block_type, _ = cfg.MODEL.BACKBONE.NON_LOCAL.split("_")
    #     layer3_list = list(model.backbone.body.layer3.children())
    #     in_ch = list(layer3_list[-1].children())[0].in_channels
    #     layer3_list.insert(
    #         len(layer3_list) - 1,
    #         init_nl_module(nl_block_type, in_ch, int(in_ch / 2)))
    #     model.backbone.body.layer3 = nn.Sequential(*layer3_list)

    return model
