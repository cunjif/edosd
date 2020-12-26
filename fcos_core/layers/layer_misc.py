# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
import torch.nn.functional as F
from easydict import EasyDict
from fcos_core.config import cfg
from fcos_core.layers.batch_norm import FrozenBatchNorm2d
from fcos_core.layers import DFConv2d


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
                x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
            )
        ]
        output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)

def get_group_gn(dim, dim_per_gp, num_groups):
    """get number of groups used by GroupNorm, based on number of channels."""
    assert dim_per_gp == -1 or num_groups == -1, \
        "GroupNorm: can only specify G or C/G."

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0, \
            "dim: {}, dim_per_gp: {}".format(dim, dim_per_gp)
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0, \
            "dim: {}, num_groups: {}".format(dim, num_groups)
        group_gn = num_groups

    return group_gn

def group_norm(out_channels, affine=True, divisor=1):
    out_channels = out_channels // divisor
    dim_per_gp = cfg.MODEL.GROUP_NORM.DIM_PER_GP // divisor
    num_groups = cfg.MODEL.GROUP_NORM.NUM_GROUPS // divisor
    eps = cfg.MODEL.GROUP_NORM.EPSILON # default: 1e-5
    return torch.nn.GroupNorm(
        get_group_gn(out_channels, dim_per_gp, num_groups), 
        out_channels, 
        eps, 
        affine
    )

def conv_with_kaiming_uniform(use_gn=False, use_relu=False):
    def make_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1
    ):
        conv = Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=dilation * (kernel_size - 1) // 2, 
            dilation=dilation, 
            bias=False if use_gn else True,
            groups=groups
        )
        # Caffe2 implementation uses XavierFill, which in fact
        # corresponds to kaiming_uniform_ in PyTorch
        nn.init.kaiming_uniform_(conv.weight, a=1)
        if not use_gn:
            nn.init.constant_(conv.bias, 0)
        module = [conv,]
        if use_gn:
            module.append(group_norm(out_channels))
        if use_relu:
            module.append(nn.ReLU(inplace=True))
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv

    return make_conv


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class SACLikeConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super().__init__()

        self.switch = nn.Conv2d(
            in_channels,
            1,
            kernel_size=1,
            stride=1,
            bias=True
        )
        self.norm_switch = Hsigmoid()
        self.switch.weight.data.fill_(0)
        self.switch.bias.data.fill_(1)

        self.s_conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        padding = 2 * padding
        dilation = 2* dilation
        self.l_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        
        self.post_conv = nn.Conv2d(
            out_channels,
            out_channels,
            1
        )

        self._init_weight()

    def _init_weight(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 1e-6)
            elif isinstance(m, nn.Sequential):
                for sm in m.modules():
                    _init(sm)

        self.apply(_init)

    def forward(self, x):
        avg_x = torch.nn.functional.pad(x, pad=(2, 2, 2, 2), mode="reflect")
        avg_x = torch.nn.functional.avg_pool2d(avg_x, kernel_size=5, stride=1, padding=0)
        switch = self.norm_switch(self.switch(avg_x))

        out_s = self.s_conv(x)

        out_l = self.l_conv(x)

        print(F"switch: {switch}")
        out = switch * out_s + (1 - switch) * out_l
        
        avg_x = F.adaptive_avg_pool2d(out, output_size=1)
        avg_x = self.post_conv(avg_x)
        print(F"avg_x: {avg_x.flatten()}")
        out = out * avg_x.expand_as(out)

        return F.relu(out)


class ConvBottomBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBottomBlock, self).__init__()

        self.merge = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.ReLU()
        )
        self._init_weight()

    def _init_weight(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 1e-6)
            elif isinstance(m, nn.Sequential):
                for mm in m.modules():
                    if isinstance(mm, nn.Conv2d):
                        nn.init.kaiming_normal_(mm.weight, a=1)
                        if mm.bias is not None:
                            nn.init.constant_(mm.bias, 1e-6)
        self.apply(_init)

    def forward(self, features):
        x = self.merge(features)

        return x


class ConvBottomBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBottomBranch, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.norm = FrozenBatchNorm2d(out_channels)
        self.agg = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.appr = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                nn.init.kaiming_normal_(m.weight, a=1)

            elif isinstance(m, nn.Sequential):
                for mm in m.modules():
                    if isinstance(mm, nn.Conv2d):
                        if mm.bias is not None:
                            nn.init.constant_(mm.bias, 0)
                        nn.init.kaiming_normal_(mm.weight, a=1)

    def forward(self, x, de):
        p = self.conv(x)
        p = self.norm(p)
        p = F.relu(p)
        
        agg  = F.relu(self.agg(p))
        appr = F.relu(self.appr(p))

        if de is not None:
            agg = agg + de.agg
            appr = appr + de.appr

        return EasyDict({ "agg": agg, "appr": appr})


class _DecoupleBranch(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(_DecoupleBranch, self).__init__()

        self.conv = nn.Conv2d(in_channel, in_channel, 3, 2, 1)
        self.norm = FrozenBatchNorm2d(in_channel)

        self.agg_down = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.agg_conv = DFConv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=True)
        
        self.appr_down = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.appr_conv = DFConv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=True)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                nn.init.kaiming_normal_(m.weight, a=1)

    def forward(self, features):
        p = self.conv(features)
        p = self.norm(p)
        p = F.relu(p)
        
        agg  = self.agg_down(p)
        agg  = self.agg_conv(agg)

        appr = self.appr_down(p)
        appr = self.appr_conv(appr)

        return (p, EasyDict({ "agg": F.relu(agg), "appr": F.relu(appr)}))

class P6P7TopBlock(nn.Module):
    def __init__(self, in_channel, p6_channels, out_channel, p7_channels=None, use_p7=True):
        super(P6P7TopBlock, self).__init__()
        
        self.is_downed = in_channel != p6_channels
        if self.is_downed:
            self.down_channel = nn.Conv2d(in_channel, p6_channels, 1, 1)
        # add
        # else:
        #     self.down_channel = nn.Conv2d(in_channel, p6_channels, 3, 1, 1, bias=False)
        
        self.p6_decouple = _DecoupleBranch(p6_channels, out_channel)

        self.use_p7 = use_p7
        if use_p7:
            self.p7_down1 = nn.Conv2d(p6_channels, p7_channels, 3, 2, 1)
            # self.p7_down2 = nn.Conv2d(p6_channels, p7_channels, 1, 1)
            self.p7_decouple = _DecoupleBranch(p7_channels, out_channel)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                nn.init.kaiming_normal_(m.weight, a=1)

    def forward(self, x):
        if self.is_downed:
            x = self.down_channel(x)

        p6, p6decouple = self.p6_decouple(x)
        
        if self.use_p7:
            # p7 = self.p7_down1(x) + self.p7_down2(p6)
            p7 = self.p7_down1(x)
            # p7 = self.p7_down2(p6)
            _, p7decouple = self.p7_decouple(F.relu(p7))

            return [p6decouple, p7decouple]
        return [p6decouple,]
