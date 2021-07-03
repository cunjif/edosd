# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict

from torch import nn

from fcos_core.modeling import registry
# from fcos_core.modeling.make_layers import conv_with_kaiming_uniform
from fcos_core.modeling.backbone import fpn as fpn_module
from . import decouple_fpn 
from . import resnet
from fcos_core.modeling.backbone import mobilenet
from fcos_core.layers.layer_misc import (
    P6P7TopBlock, 
    ConvBottomBlock,
    conv_with_kaiming_uniform,
    ConvBottomBranch
)


@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    return model


@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("R-50-FPN-RETINANET")
@registry.BACKBONES.register("R-101-FPN-RETINANET")
def build_resnet_fpn_p3p7_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("R-50-FPN-RETINANET-DEFPN")
@registry.BACKBONES.register("R-101-FPN-RETINANET-DEFPN")
def build_resnet_fpn_p3p7_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    
    p6_channels = cfg.MODEL.EDOSD.Hidden_P6_CHANNELS
    p7_channels = cfg.MODEL.EDOSD.Hidden_P7_CHANNELS

    if not cfg.MODEL.RETINANET.USE_C5:
        p6_channels = in_channels_p6p7

    # stride 8
    fpn = decouple_fpn.DecoupleFPN(
    # fpn = decouple_fpn_no_merge.DecoupleFPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        num_outs=5,
        conv_block=conv_with_kaiming_uniform(cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU),
        # bottom_block = ConvBottomBlock(in_channels_stage2, out_channels),
        bottom_block = ConvBottomBranch(in_channels_stage2, out_channels),
        top_block=P6P7TopBlock(in_channels_p6p7, p6_channels, out_channel=out_channels, p7_channels=p7_channels),
        use_c5=cfg.MODEL.RETINANET.USE_C5
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("MNV2-FPN-RETINANET")
def build_mnv2_fpn_backbone(cfg):
    body = mobilenet.MobileNetV2(cfg)
    in_channels_stage2 = body.return_features_num_channels
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2[1],
            in_channels_stage2[2],
            in_channels_stage2[3],
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(out_channels, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
