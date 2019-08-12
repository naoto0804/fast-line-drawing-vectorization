# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from itertools import accumulate

import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d
from maskrcnn_benchmark.layers.attention import MultiHeadAttention, \
    MultiHeadAttentionConv
from maskrcnn_benchmark.modeling.poolers import Pooler
from ..box_head.roi_box_feature_extractors import \
    ResNet50Conv5ROIFeatureExtractor


class MaskRCNNFPNFeatureExtractor(nn.Module):

    def __init__(self, cfg):
        super(MaskRCNNFPNFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES,
            sampling_ratio=cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO,
        )
        self.pooler = pooler
        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS
        next_feature = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.blocks = []
        self.use_attn = False if cfg.MODEL.ROI_MASK_HEAD.ATTN == "" else True

        # Determine whether upsampling is necessary from the resolution
        # if cfg.MODEL.ROI_MASK_HEAD.RESOLUTION / (2.0 * resolution) == 2.0:
        #     use_upsample = True
        # else:
        #     use_upsample = False

        use_upsample = \
            True if (cfg.MODEL.ROI_MASK_HEAD.RESOLUTION / resolution) == 4.0 \
                else False
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "mask_fcn{}".format(layer_idx)
            # if layer_idx % 2 == 1 and use_upsample:
            #     module = ConvTranspose2d(next_feature, layer_features, 2, 2, 0)
            # else:
            #     module = Conv2d(next_feature, layer_features, 3, 1, 1)
            if layer_idx == 3 and use_upsample:
                module = ConvTranspose2d(next_feature, layer_features, 2, 2, 0)
            else:
                module = Conv2d(next_feature, layer_features, 3, 1, 1)

            # Caffe2 implementation uses MSRAFill, which in fact
            # corresponds to kaiming_normal_ in PyTorch
            nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                    nonlinearity="relu")
            nn.init.constant_(module.bias, 0)

            if self.use_attn and layer_idx in [2]:
                attn_name = "mask_attn{}".format(layer_idx)
                size = (layer_features, resolution, resolution)
                self.add_module(attn_name, RoIAttnModule(cfg, size))

            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        for layer_idx, layer_name in enumerate(self.blocks, 1):
            x = F.relu(getattr(self, layer_name)(x))
            if layer_idx == 2 and self.use_attn:
                layer_name = "mask_attn{}".format(layer_idx)
                x = getattr(self, layer_name)(x, proposals)
        return x


class RoIAttnModule(nn.Module):

    def __init__(self, cfg, size):
        super(RoIAttnModule, self).__init__()
        self.size = size
        self.attn_type = cfg.MODEL.ROI_MASK_HEAD.ATTN
        if self.attn_type == 'linear':
            self.attn = MultiHeadAttention(
                n_head=8, d_model=self.size, d_k=32, d_v=32)
        elif self.attn_type == 'conv':
            self.attn = MultiHeadAttentionConv(
                n_head=8, d_model=self.size, d_k=32, d_v=32)
        else:
            raise NotImplementedError

    def forward(self, x, proposals):
        n_rois = [len(p) for p in proposals]
        # n_batch = len(proposals)
        start_inds = [0] + list(accumulate(n_rois))[:-1]
        result_x_list = []
        for i, (start, step) in enumerate(zip(start_inds, n_rois)):
            h = x.narrow(0, start, step)
            if self.attn_type == 'linear':
                h = h.reshape((1, n_rois[i], -1))
                h, _ = self.attn(q=h, k=h, v=h)  # Self attention
                C, H, W = self.size
                h = h.reshape((-1, C, H, W))
            else:
                h, _ = self.attn(q=h, k=h, v=h)
            result_x_list.append(h)
        x = torch.cat(result_x_list, dim=0)

        return x


# class MaskRCNNFPNFeatureExtractorNL(nn.Module):
#
#     def __init__(self, cfg):
#         super(MaskRCNNFPNFeatureExtractorNL, self).__init__()
#
#         resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
#         pooler = Pooler(
#             output_size=(resolution, resolution),
#             scales=cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES,
#             sampling_ratio=cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO,
#         )
#         self.pooler = pooler
#         layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS
#         next_feature = cfg.MODEL.BACKBONE.OUT_CHANNELS
#         self.blocks = []
#         self.nl_blocks = []
#         self.use_nl = False
#
#         # Determine whether upsampling is necessary from the resolution
#         if cfg.MODEL.ROI_MASK_HEAD.RESOLUTION / (2.0 * resolution) == 4.0:
#             use_upsample = True
#         else:
#             use_upsample = False
#
#         for layer_idx, layer_features in enumerate(layers, 1):
#             layer_name = "mask_fcn{}".format(layer_idx)
#             if layer_idx % 2 == 1 and use_upsample:
#                 module = ConvTranspose2d(next_feature, layer_features, 2, 2, 0)
#             else:
#                 module = Conv2d(next_feature, layer_features, 3, 1, 1)
#             # Caffe2 implementation uses MSRAFill, which in fact
#             # corresponds to kaiming_normal_ in PyTorch
#             nn.init.kaiming_normal_(module.weight, mode="fan_out",
#                                     nonlinearity="relu")
#             nn.init.constant_(module.bias, 0)
#
#             nl_layer_name = "mask_fcn_nl{}".format(layer_idx)
#             nl_name = cfg.MODEL.ROI_MASK_HEAD.get('NON_LOCAL')
#             if layer_idx % 2 == 0 and nl_name != "":
#                 self.use_nl = True
#                 in_ch, mid_ch = next_feature, int(next_feature / 2)
#                 nl_type, n_nl_block = nl_name.split("_")
#                 assert int(n_nl_block) == 2
#                 self.nl_blocks.append(nl_layer_name)
#                 self.add_module(nl_layer_name,
#                                 init_nl_module(nl_type, in_ch, mid_ch))
#
#             self.add_module(layer_name, module)
#             next_feature = layer_features
#             self.blocks.append(layer_name)
#
#     def forward(self, x, proposals):
#         x = self.pooler(x, proposals)
#         for layer_idx, layer_name in enumerate(self.blocks, 1):
#             x = F.relu(getattr(self, layer_name)(x))
#             if layer_idx % 2 == 0 and self.use_nl:
#                 x = getattr(self, "mask_fcn_nl{}".format(layer_idx))(x)
#
#         return x


_ROI_MASK_FEATURE_EXTRACTORS = {
    "ResNet50Conv5ROIFeatureExtractor": ResNet50Conv5ROIFeatureExtractor,
    "MaskRCNNFPNFeatureExtractor": MaskRCNNFPNFeatureExtractor,
}


def make_roi_mask_feature_extractor(cfg):
    func = _ROI_MASK_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR]
    return func(cfg)
