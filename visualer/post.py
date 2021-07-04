"""
This file contains specific functions for computing losses of FCOS
file
"""

import os
from numpy.core.fromnumeric import shape
import torch
import math
from torch.nn import functional as F
from torch import nn, poisson
from fcos_core.layers.iou_loss import IOULoss
from fcos_core.layers import SigmoidFocalLoss
from fcos_core.modeling.matcher import Matcher
from fcos_core.modeling.utils import cat
from fcos_core.structures.boxlist_ops import boxlist_iou
from fcos_core.structures.boxlist_ops import cat_boxlist


INF = 100000000


class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1

        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.center_sampling_radius = cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS
        # self.iou_loss_type = cfg.MODEL.FCOS.IOU_LOSS_TYPE
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS

        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1.0):
        '''
        This code is from
        https://github.com/yqyao/FCOS_PLUS/blob/0d20ba34ccc316650d8c30febb2eb40cb6eaae37/
        maskrcnn_benchmark/modeling/rpn/fcos/loss.py#L42
        '''
        num_gts = gt.shape[0]
        K = len(gt_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(
                xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0]
            )
            center_gt[beg:end, :, 1] = torch.where(
                ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1]
            )
            center_gt[beg:end, :, 2] = torch.where(
                xmax > gt[beg:end, :, 2],
                gt[beg:end, :, 2], xmax
            )
            center_gt[beg:end, :, 3] = torch.where(
                ymax > gt[beg:end, :, 3],
                gt[beg:end, :, 3], ymax
            )
            beg = end
        left = gt_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs[:, None]
        top = gt_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def asymetric_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys):
        num_gts = gt.shape[0]
        K = sum(num_points_per)
        gt = gt[None].expand(K, num_gts, 4)
        left_x = gt[..., 0]
        right_x = gt[..., 2]
        top_y = gt[..., 1]
        bottom_y = gt[..., 3]
        center_gt = gt.new_zeros(gt.shape)
        areas = (right_x - left_x) * (bottom_y - top_y)
        # no gt
        print(F"gt_xs is {gt_xs.shape}")
        print(F"gt shape: {gt.shape}")
        if ((left_x + right_x) / 2).sum() == 0:
            return left_x.new_zeros(K, dtype=torch.uint8)

        s_inds = areas > 40000
        l_inds = ~s_inds
        
        left_x[s_inds]   = (2*left_x[s_inds] + right_x[s_inds]) / 3
        right_x[s_inds]  = (left_x[s_inds] + 2*right_x[s_inds]) / 3
        top_y[s_inds]    = (2*top_y[s_inds] + bottom_y[s_inds]) / 3
        bottom_y[s_inds] = (top_y[s_inds] + 2*bottom_y[s_inds]) / 3
        # beg = 0
        # for level, n_p in enumerate(num_points_per):
        #     end = beg + n_p
        #     # stride = strides[level] * radius
        #     xmin = center_x[beg:end] - stride
        #     ymin = center_y[beg:end] - stride
        #     xmax = center_x[beg:end] + stride
        #     ymax = center_y[beg:end] + stride

        # exit()

    def prepare_targets(self, points, targets):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        self.num_points_per_level = num_points_per_level
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets, gt_inds, total_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)
            gt_inds[i] = torch.split(gt_inds[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        gt_inds_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )

            reg_targets_per_level = torch.cat([
                reg_targets_per_im[level]
                for reg_targets_per_im in reg_targets
            ], dim=0)

            gt_inds_level_first.append(
                torch.cat([gt_inds_per_im[level] for gt_inds_per_im in gt_inds], dim=0)
            )

            if self.norm_reg_targets:
                reg_targets_per_level = reg_targets_per_level / self.fpn_strides[level]
            reg_targets_level_first.append(reg_targets_per_level)

        # labels = [torch.cat(label, dim=0) for label in labels]

        return labels_level_first, reg_targets_level_first, gt_inds_level_first, total_targets

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        gt_inds = []
        total_target = 0
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            area = targets_per_im.area()
            labels_length = len(labels_per_im)

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            if self.center_sampling_radius > 0:
                is_in_boxes = self.get_sample_region(
                    bboxes,
                    self.fpn_strides,
                    self.num_points_per_level,
                    xs, ys,
                    radius=self.center_sampling_radius
                )
                # self.asymetric_sample_region(bboxes, self.fpn_strides, self.num_points_per_level, xs, ys)
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

            locations_to_gt_inds += (1 + total_target) # 1-based
            locations_to_gt_inds[locations_to_min_area == INF] = 0
            gt_inds.append(locations_to_gt_inds)
            total_target += labels_length

        return labels, reg_targets, gt_inds, total_target
    
    def calc_ious(self, pred, target, iou_type="iou"):
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

        if iou_type == "iou":
            return ious, ious
        elif iou_type == "giou":
            return gious, gious.clamp_min(0.0)
        elif iou_type == "diou":
            return dious, dious.clamp_min(0.0)
        else:
            raise NotImplementedError
    
    def __call__(self, locations, box_cls, box_regression, centerness, targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        labels, reg_targets, gt_inds, num_target = self.prepare_targets(locations, targets)

        box_cls_flatten_ = []
        box_regression_flatten_ = []
        centerness_flatten = []
        labels_flatten_ = []
        reg_targets_flatten_ = []
        gt_inds_flatten = []
        for l in range(len(labels)):
            box_cls_flatten_.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten_.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            labels_flatten_.append(labels[l].reshape(-1))
            reg_targets_flatten_.append(reg_targets[l].reshape(-1, 4))
            gt_inds_flatten.append(gt_inds[l].reshape(-1))
            centerness_flatten.append(centerness[l].reshape(-1))

        box_cls_flatten = torch.cat(box_cls_flatten_, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten_, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten_, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten_, dim=0)
        gt_inds_flatten = torch.cat(gt_inds_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        gt_inds_flatten = gt_inds_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        if pos_inds.numel() > 0:
            conf = sigmoid_focal_loss(box_cls_flatten, labels_flatten)
            _, ious = self.calc_ious(box_regression_flatten, reg_targets_flatten, iou_type="iou")

            return conf, ious, centerness_flatten.sigmoid()
        return None


def sigmoid_focal_loss(logits, targets):
    num_classes = logits.size(1)

    dtype = targets.dtype
    device = targets.device
    class_range = torch.arange(1, num_classes+1, dtype=dtype, device=device).unsqueeze(0)

    t = targets.unsqueeze(1)
    p = torch.sigmoid(logits)

    pos = p[t == class_range]

    return pos


def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator
