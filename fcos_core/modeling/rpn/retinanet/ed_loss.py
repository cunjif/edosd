"""
This file contains specific functions for computing losses on the RetinaNet
file
"""

import torch
from torch import nn
from torch.nn import functional as F

from ..utils import concat_box_prediction_layers

from fcos_core.layers import smooth_l1_loss
from fcos_core.layers import SigmoidFocalLoss
from fcos_core.modeling.matcher import Matcher
from fcos_core.modeling.utils import cat
from fcos_core.structures.boxlist_ops import boxlist_iou
from fcos_core.structures.boxlist_ops import cat_boxlist
from fcos_core.modeling.rpn.loss import RPNLossComputation


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


class RetinaNetLossComputation(RPNLossComputation):
    """
    This class computes the RetinaNet loss.
    """

    def __init__(self, proposal_matcher, box_coder,
                 generate_labels_func,
                 sigmoid_focal_loss,
                 bbox_reg_beta=0.11,
                 regress_norm=1.0,
                 fl_gamma=2,
                 fl_alpha=0.25,
                 num_classes=80):
        """
        Arguments:
            proposal_matcher (Matcher)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.box_coder = box_coder
        self.box_cls_loss_func = sigmoid_focal_loss
        self.bbox_reg_beta = bbox_reg_beta
        self.copied_fields = ['labels']
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['between_thresholds']
        self.regress_norm = regress_norm

        self.cls_weight_func = FocalLossFunc(
            fl_gamma,
            fl_alpha,
            num_classes
        )
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")

    def collect_bbox_weights(self, clses, labels, box_regrs, reg_targets, 
        anchors, assign_gt_inds, num_gts, eps=1e-12, gamma=0.1
    ):
        device = clses[0].device
        # if num_pos < 1:
        #     num_pos = 1

        with torch.no_grad():
            cls_losses = self.cls_weight_func(
                clses,
                labels
            )
            box_losses = self.compute_reg_loss(
                reg_targets,
                box_regrs,
                anchors
            )
            if len(cls_losses.shape) == 2:
                cls_losses = cls_losses.sum(dim=-1)
            if len(box_losses.shape) == 2:
                box_losses = box_losses.sum(dim=-1)

            losses = cls_losses + box_losses
            
            assert losses.size(0) == assign_gt_inds.size(0)
            
            label_sequence = torch.arange(1, num_gts+1, device=device)
            ious = self.compute_iou(box_regrs, reg_targets, anchors, iou_type="giou")

            weights = torch.zeros_like(losses)
            for gt in label_sequence:
                match = assign_gt_inds == gt
                if match.sum() > 0:
                    loss: torch.Tensor = losses[match]
                    iou = ious[match]
                    
                    high = loss.max()
                    low = loss.min()
                    ws = (loss - high) / (-F.relu(high - low) - eps) 
                    
                    ws[ws == ws.min()] = 1e-4

                    ws = torch.where(gamma - iou > 0, 
                        ws * iou + F.relu(gamma - iou),
                        ws)

                    weights[match] = ws

        return weights

    def compute_iou(self, pred, target, anchor, weight=None, iou_type="iou"):
        pred_boxes = self.box_coder.decode(pred.view(-1, 4), anchor.view(-1, 4))
        pred_x1 = pred_boxes[:, 0]
        pred_y1 = pred_boxes[:, 1]
        pred_x2 = pred_boxes[:, 2]
        pred_y2 = pred_boxes[:, 3]
        pred_x2 = torch.max(pred_x1, pred_x2)
        pred_y2 = torch.max(pred_y1, pred_y2)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

        gt_boxes = self.box_coder.decode(target.view(-1, 4), anchor.view(-1, 4))
        target_x1 = gt_boxes[:, 0]
        target_y1 = gt_boxes[:, 1]
        target_x2 = gt_boxes[:, 2]
        target_y2 = gt_boxes[:, 3]
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

        x1_intersect = torch.max(pred_x1, target_x1)
        y1_intersect = torch.max(pred_y1, target_y1)
        x2_intersect = torch.min(pred_x2, target_x2)
        y2_intersect = torch.min(pred_y2, target_y2)
        area_intersect = torch.zeros(pred_x1.size()).to(pred)
        mask = (y2_intersect > y1_intersect) * (x2_intersect > x1_intersect)
        area_intersect[mask] = (x2_intersect[mask] - x1_intersect[mask]) * (y2_intersect[mask] - y1_intersect[mask])

        x1_enclosing = torch.min(pred_x1, target_x1)
        y1_enclosing = torch.min(pred_y1, target_y1)
        x2_enclosing = torch.max(pred_x2, target_x2)
        y2_enclosing = torch.max(pred_y2, target_y2)
        area_enclosing = (x2_enclosing - x1_enclosing) * (y2_enclosing - y1_enclosing) + 1e-7

        area_union = pred_area + target_area - area_intersect + 1e-7
        ious = area_intersect / area_union
        gious = ious - (area_enclosing - area_union) / area_enclosing

        if iou_type == "iou":
            return ious
        elif iou_type == "giou":
            return gious
        else:
            raise NotImplementedError
    
    def GIoULoss(self, pred, target, anchor, weight=None):
        gious = self.compute_iou(pred, target, anchor, weight=weight, iou_type="giou")
        losses = 1 - gious

        if weight is not None and weight.sum() > 0:
            return (losses * weight)
        else:
            assert losses.numel() != 0
            return losses

    def compute_reg_loss(
        self, regression_targets, box_regression, all_anchors, labels, weights
    ):
        if 'iou' in self.reg_loss_type:
            reg_loss = self.GIoULoss(box_regression,
                                     regression_targets,
                                     all_anchors,
                                     weight=weights)
        elif self.reg_loss_type == 'smoothl1':
            reg_loss = smooth_l1_loss(box_regression,
                                      regression_targets,
                                      beta=self.bbox_reg_beta,
                                      size_average=False,
                                      sum=False)
            if weights is not None:
                reg_loss *= weights
        else:
            raise NotImplementedError
        return reg_loss[labels > 0].view(-1)
    
    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        num_gt = 0
        gt_inds = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image, self.copied_fields
            )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = self.generate_labels_func(matched_targets)
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard anchors that go out of the boundaries of the image
            if "not_visibility" in self.discard_cases:
                labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            if "between_thresholds" in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox
            )
        
            matched_idxs += num_gt
            num_gt += len(targets_per_image.get_field("labels"))
            labels.append(labels_per_image)
            gt_inds.append(matched_idxs)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets, gt_inds, num_gt

    def __call__(self, anchors, box_cls, box_regression, centerness, targets):
        """
        Arguments:
            anchors (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            retinanet_cls_loss (Tensor)
            retinanet_regression_loss (Tensor
        """
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets, gt_inds, num_gt = self.prepare_targets(anchors, targets)

        N = len(labels)
        box_cls, box_regression = \
                concat_box_prediction_layers(box_cls, box_regression)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        gt_inds = torch.cat(gt_inds, dim=0)
        pos_inds = torch.nonzero(labels > 0).squeeze(1)
        anchors = torch.cat([cat_boxlist(anchors_per_image).bbox
            for anchors_per_image in anchors], dim=0)
        centerness = [ip.permute(0, 2, 3, 1).reshape(N, -1, 1) for ip in centerness]
        centerness = torch.cat(centerness, dim=1).reshape(-1)
        centerness = centerness[pos_inds]

        num_gpus = get_num_gpus()
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)
        
        if pos_inds.numel() > 0:
            with torch.no_grad():
                box_weights = self.collect_bbox_weights(
                    box_cls[pos_inds],
                    labels[pos_inds].int(),
                    box_regression[pos_inds],
                    regression_targets[pos_inds],
                    anchors[pos_inds],
                    gt_inds[pos_inds],
                    num_gt
                )

            sum_centerness_targets_avg_per_gpu = \
                reduce_sum(box_weights.sum()).item() / float(num_gpus)

            gious = self.compute_iou(box_regression[pos_inds],
                    regression_targets[pos_inds],
                    anchors[pos_inds],
                    weight=self.collect_bbox_weights,
                    iou_type="giou")
            reg_loss = (gious * box_weights).sum() * 1.3
            centerness_loss = F.binary_cross_entropy_with_logits(
                centerness, 
                gious.clone().clamp(min=0).detach(),
                reduction="sum"
            ) / num_pos_avg_per_gpu * 0.5

            cls_loss = sigmoid_focal_loss(
                box_cls, 
                labels.int(),
                gamma=self.fl_gamma,
                alpha=self.fl_alpha,
                ious=gious
            ) / num_pos_avg_per_gpu
        else:
            cls_loss = self.box_cls_loss_func(
                box_cls,
                labels
            ) / num_pos_avg_per_gpu
            reg_loss = box_regression.sum()
            centerness_loss = centerness.sum()

        return cls_loss, reg_loss, centerness_loss


def sigmoid_focal_loss(logits, targets, gamma, alpha, ious, pos_thr=0.3):
    num_classes = logits.size(1)
    # print(F"max ious: {(ious == ious.max()).nonzero()}")
    # important_weight = scale_factor(ious)
    important_weight = F.relu((1. + ious) / 2. + 1e-6)

    dtype = targets.dtype
    device = targets.device
    class_range = torch.arange(1, num_classes+1, dtype=dtype, device=device).unsqueeze(0)

    t = targets.unsqueeze(1)
    p = torch.sigmoid(logits)

    pos = p[t == class_range]
    neg = p[((t != class_range) * (t >= 0))]

    term0 = (1 - pos)**(gamma / 3) * torch.log(pos).clamp_min(-100) 
    term1 = (important_weight * (term0.sum() / (important_weight * term0).sum())) * term0 * alpha
    # print((important_weight == important_weight.max()).nonzero())
    # t = important_weight * (term0.sum() / (important_weight * term0).sum())
    # print("t: ", (t == t.max()).nonzero())
    # exit()
    term2 = neg ** gamma * torch.log(1 - neg).clamp_min(-100) * (1-alpha)
    term3 = (ious.detach() < pos_thr).detach().float() * torch.log((1 - pos).clamp_min(-100))

    assert torch.isnan(term0).sum() == 0
    assert torch.isnan(term1).sum() == 0
    assert torch.isnan(term2).sum() == 0
    assert torch.isnan(term3).sum() == 0

    loss = -term1.sum() - term2.sum() - term3.sum()
    return loss


class FocalLossFunc(nn.Module):
    def __init__(self, gamma, alpha, num_classes):
        super(FocalLossFunc, self).__init__()
        
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes
    
    def forward(self, logits, targets):
        gamma = self.gamma
        alpha = self.alpha
        dtype = targets.dtype
        device = targets.device
        class_range = torch.arange(1, self.num_classes+1, dtype=dtype, device=device).unsqueeze(0)

        t = targets.unsqueeze(1)
        p = torch.sigmoid(logits)
        term1 = (1 - p) ** gamma * torch.log(p)
        term2 = p ** gamma * torch.log(1 - p)
        losses =  -(t == class_range).float() * term1 * alpha - ((t != class_range) * (t >= 0)).float() * term2 * (1 - alpha)
        return losses


def generate_retinanet_labels(matched_targets):
    labels_per_image = matched_targets.get_field("labels")
    return labels_per_image


def make_retinanet_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RETINANET.FG_IOU_THRESHOLD,
        cfg.MODEL.RETINANET.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )
    sigmoid_focal_loss = SigmoidFocalLoss(
        cfg.MODEL.RETINANET.LOSS_GAMMA,
        cfg.MODEL.RETINANET.LOSS_ALPHA
    )

    loss_evaluator = RetinaNetLossComputation(
        matcher,
        box_coder,
        generate_retinanet_labels,
        sigmoid_focal_loss,
        bbox_reg_beta = cfg.MODEL.RETINANET.BBOX_REG_BETA,
        regress_norm = cfg.MODEL.RETINANET.BBOX_REG_WEIGHT,
        fl_gamma=cfg.MODEL.RETINANET.LOSS_GAMMA,
        fl_alpha=cfg.MODEL.RETINANET.LOSS_ALPHA,
        num_classes=cfg.MODEL.RETINANET.NUM_CLASSES-1
    )
    return loss_evaluator
