# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
helper class that supports empty tensors on some nn functions.

Ideally, add support directly in PyTorch to empty tensors in
those functions.

This can be removed once https://github.com/pytorch/pytorch/issues/12013
is implemented
"""

import math

import torch
from torch.nn.modules.utils import _ntuple


class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class Conv2d(torch.nn.Conv2d):
    def forward(self, x):
        if x.numel() > 0:
            return super(Conv2d, self).forward(x)
        # get output shape

        output_shape = [
            (i + 2 * p - (di * (k - 1) + 1)) // d + 1
            for i, p, di, k, d in zip(
                x.shape[-2:], self.padding, self.dilation, self.kernel_size,
                self.stride
            )
        ]
        output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class ConvTranspose2d(torch.nn.ConvTranspose2d):
    def forward(self, x):
        if x.numel() > 0:
            return super(ConvTranspose2d, self).forward(x)
        # get output shape

        output_shape = [
            (i - 1) * d - 2 * p + (di * (k - 1) + 1) + op
            for i, p, di, k, d, op in zip(
                x.shape[-2:],
                self.padding,
                self.dilation,
                self.kernel_size,
                self.stride,
                self.output_padding,
            )
        ]
        output_shape = [x.shape[0], self.bias.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


def interpolate(
        input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    if input.numel() > 0:
        return torch.nn.functional.interpolate(
            input, size, scale_factor, mode, align_corners
        )

    def _check_size_scale_factor(dim):
        if size is None and scale_factor is None:
            raise ValueError("either size or scale_factor should be defined")
        if size is not None and scale_factor is not None:
            raise ValueError(
                "only one of size or scale_factor should be defined")
        if (
                scale_factor is not None
                and isinstance(scale_factor, tuple)
                and len(scale_factor) != dim
        ):
            raise ValueError(
                "scale_factor shape must match input shape. "
                "Input is {}D, scale_factor size is {}".format(dim, len(
                    scale_factor))
            )

    def _output_size(dim):
        _check_size_scale_factor(dim)
        if size is not None:
            return size
        scale_factors = _ntuple(dim)(scale_factor)
        # math.floor might return float in py2.7
        return [
            int(math.floor(input.size(i + 2) * scale_factors[i])) for i in
            range(dim)
        ]

    output_shape = tuple(_output_size(2))
    output_shape = input.shape[:-2] + output_shape
    return _NewEmptyTensorOp.apply(input, output_shape)


# Originally from:
# https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
class AddCoords(torch.nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        if input_tensor.numel() == 0:
            print("L119 of maskrcnn_benchmark.layers.misc")
            output_shape = list(input_tensor.shape)
            if self.with_r:
                output_shape[1] += 3
            else:
                output_shape[1] += 2
            return _NewEmptyTensorOp.apply(input_tensor, output_shape)
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_ch = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_ch = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        if x_dim < 2:
            xx_ch = torch.zeros(xx_ch.size()).fill_(0.5)
        else:
            xx_ch = xx_ch.float() / (x_dim - 1)

        if y_dim < 2:
            yy_ch = torch.zeros(yy_ch.size()).fill_(0.5)
        else:
            yy_ch = yy_ch.float() / (y_dim - 1)

        xx_ch = xx_ch * 2 - 1
        yy_ch = yy_ch * 2 - 1

        xx_ch = xx_ch.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_ch = yy_ch.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_ch.type_as(input_tensor),
            yy_ch.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(
                torch.pow(xx_ch.type_as(input_tensor) - 0.5, 2) + \
                torch.pow(yy_ch.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


# https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
class CoordConv2d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_channels += 2
        if with_r:
            in_channels += 1
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


def test_coord_conv2d():
    import torch.nn.functional as F
    for (h, w) in [(1, 1), (1, 3), (3, 3)]:
        basic_conv = Conv2d(3, 3, 3, padding=1)
        coord_conv = CoordConv2d(3, 3, 3, padding=1, bias=False)

        x = torch.randn(4, 3, h, w)
        x = basic_conv(x)

        coord_h = coord_conv(x)
        coord_loss = F.mse_loss(coord_h, torch.randn(coord_h.size()))
        with torch.autograd.detect_anomaly():
            coord_loss.backward()
        print(coord_loss)
