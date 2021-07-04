# GIoU and Linear IoU are added by following
# https://github.com/yqyao/FCOS_PLUS/blob/master/maskrcnn_benchmark/layers/iou_loss.py.
import os
import torch
from torch import nn
import numpy as np
from functools import partial
import torch.nn.functional as F


class IOULoss(nn.Module):
    def __init__(self, iou_loss_type="iou"):
        super(IOULoss, self).__init__()
        self.iou_loss_type = iou_loss_type

    def calc_ious(self, pred, target):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        pred_cx = (pred_left + pred_right) / 2
        pred_cy = (pred_top + pred_bottom) / 2
        target_cx = (target_left + target_right) / 2
        target_cy = (target_left + target_bottom) / 2

        target_area = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(
            pred_right, target_right)
        inter_diag = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        outer_diag = g_w_intersect ** 2 + g_h_intersect ** 2
        ac_uion = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        gious = torch.clamp(gious, min=-1.0, max=1.0)
        dious = ious - inter_diag / outer_diag
        dious = torch.clamp(dious, min=-1.0, max=1.0)

        if self.iou_loss_type == "iou" or self.iou_loss_type == "linear_iou":
            return ious
        elif self.iou_loss_type == "giou":
            return gious
        elif self.iou_loss_type == "diou":
            return dious
        else:
            raise NotImplementedError
        
    def forward(self, ious, weight=None, reduction="sum"):
        if self.iou_loss_type == "iou":
            losses = -torch.log(ious)
        elif self.iou_loss_type == "linear_iou" or \
            self.iou_loss_type == "giou" or \
            self.iou_loss_type == "diou":
            losses = 1 - ious
        else:
            raise NotImplementedError
        
        if reduction == "sum":
            if weight is not None:
                losses = (losses * weight).sum() 
            else:
                losses = losses.sum()
        elif reduction == "origin":
            pass
        else:
            raise NotImplementedError
        return losses
