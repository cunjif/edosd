"""可视化参数模块"""

import os
import torch
import matplotlib.pyplot as plt
import argparse
from random import choices
from fcos_core.config import cfg
from fcos_core.utils.checkpoint import DetectronCheckpointer
from visualer.network import GeneralizedRCNN
from PIL import Image
from torchvision.transforms import (
    Compose,
    Scale,
    ToTensor,
)
from fcos_core.data.transforms import Normalize
from fcos_core.data.transforms import Resize
from fcos_core.data.build import make_data_loader
import numpy as np
import cv2

torch.set_printoptions(precision=3, sci_mode=False, threshold=65536)


# _COLORS = ["red", "blue", "yellow", "orange", "purple", "green"]
_COLORS = ["purple",] * 5


def randomColors(k):
    colors = _COLORS[:k]
    return colors


def visual_core_edge_feats(dots, lvls):
    """visual special object's core feature response and \\
    edge feature response in special level feature maps  \\
    with its grouth truth box
    """
    plt.figure(figsize=(20,16))
    plt.xlabel("iou", fontsize=18)
    plt.ylabel("cls score", fontsize=16)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title("test-fcos")
    
    for (cls, iou), color in dots:
        # print(cls)
        # da = 1 / len(cls)
        # a = 1
        # for idx,(c,i) in enumerate(zip(cls, iou)):
        #     plt.scatter(y=c, x=i, c=color, alpha=a)
        #     # plt.annotate(F"{idx}", (i, c))
        #     if a-da != 0:
        #         a -= da
        plt.scatter(y=cls, x=iou, c=color, s=15)
    
    plt.legend(lvls)
    plt.show()


def calc_iou(bboxes, gt):
    bb_left   = bboxes[:, 1]
    bb_top    = bboxes[:, 0]
    bb_right  = bboxes[:, 3]
    bb_bottom = bboxes[:, 2]

    # gt = gt.expand_as(bboxes)
    # gt_left = gt[:, 0]
    # gt_top = gt[:, 1]
    # gt_right = gt[:, 2]
    # gt_bottom = gt[:, 3]

    gt_left, gt_top, gt_right, gt_bottom = gt

    # print(bboxes)

    bb_area = (bb_right - bb_left) * (bb_bottom - bb_top)

    gt_area = (gt_right - gt_left) * (gt_bottom - gt_top)

    inter_w = torch.min(bb_right, gt_right) - torch.max(bb_left, gt_left)
    inter_w[inter_w<0] = 0
    inter_h = torch.min(bb_bottom, gt_bottom) - torch.max(bb_top, gt_top)
    inter_h[inter_h<0] = 0

    # inter = (torch.min(bb_right, gt_right) - torch.max(bb_left, gt_left)) * (torch.min(bb_bottom, gt_bottom) - torch.max(bb_top, gt_top))
    inter = inter_w * inter_h
    
    union = gt_area + bb_area - inter
    iou = (inter) / (union+1)
    return iou


def show_heatmap(features, name, h, w, s):
    features = features[0]
    features = features.cpu().sigmoid().numpy()
    # h, w = features.shape[-2:]
    imgs = np.zeros((h*16+16, w*16+16, 3), dtype=np.uint8)
    
    r = []
    for idx,feat in enumerate(features):
        feat = feat * 255
        feat = feat.astype(np.uint8)
        feat = cv2.resize(feat, (w, h))
        img = cv2.applyColorMap(feat, cv2.COLORMAP_JET)
        hh = h+1
        ww = w+1
        f = np.zeros((hh,ww,3), dtype=np.uint8)
        f[0:h,0:w,:] = img
        r.append(f)
    h += 1
    w += 1
    for i in range(16):
        for j in range(16):
            imgs[i*h:(i+1)*h, w*j:w*(j+1), :] = r[i*16+j]

    cv2.imshow(F"{name}-{s}", imgs)
    cv2.waitKey(-1)


def visual():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--image", type=str)
    parser.add_argument("--bbox", type=str)
    parser.add_argument("--cat", type=int)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    network = GeneralizedRCNN(cfg)
    network.to(cfg.MODEL.DEVICE)
    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, network, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    # scale = 3.2
    scale=1
    img = Image.open(args.image).convert('RGB')
    img = img.resize((round(img.width/scale), round(img.height/scale)), Image.ANTIALIAS)
    cimg = cv2.imread(args.image)
    h, w = cimg.shape[:2]
    cimg = cv2.resize(cimg, (round(w/scale), round(h/scale)))
    transforms = Compose([
        Resize(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST),
        ToTensor(),
        Normalize(cfg.INPUT.PIXEL_MEAN, 
            cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255)
    ])
    img = transforms(img)
    gt = torch.from_numpy(np.array(eval(args.bbox))).float()
    gt = gt / scale
    print(F"source image shape: {cimg.shape[:2]}")
    print(F"gt -> {gt} :: {gt[2]-gt[0]}x{gt[3]-gt[1]} :: {torch.sqrt((gt[2]-gt[0])*(gt[3]-gt[1]))}")
    # gt = torch.from_numpy(np.array([53, 222, 367, 463])).float()
    gid = args.cat
    topk = 500

    # ccimg = cimg.copy()
    # cv2.rectangle(ccimg, (gt[1], gt[0]), (gt[3], gt[2]), (0,0,255), 2)
    # cv2.imshow("gt", ccimg)
    # cv2.waitKey(-1);exit()
    # network = network.cuda()
    # img = img.cuda()
    # data_loader = make_data_loader(cfg, is_train=False, is_distributed=False)
    network.eval()
    with torch.no_grad():
        locations, box_cls, box_regression, centerness, cls_towers, box_towers = network(img)

        locations = [_.cpu() for _ in locations]
        box_cls = [_.cpu() for _ in box_cls]
        box_regression = [_.cpu() for _ in box_regression]
        centerness = [_.cpu() for _ in centerness]
        cls_towers = [_.cpu() for _ in cls_towers]
        box_towers = [_.cpu() for _ in box_towers]

        strides = cfg.MODEL.FCOS.FPN_STRIDES
        size = img.size()[-2:]

        # print(len(box_cls))
        # print(box_cls[0].shape)
        # for i in range(len(box_cls)):
        #     print(box_cls[i][box_cls[i]>0])
        # exit()
        
        results = []
        height, width = box_cls[0].shape[-2:]
        for location, cls, regr, stride, cr, ct, bt in zip(locations, box_cls, box_regression, strides, centerness, cls_towers, box_towers):
            N,C,H,W = cls.shape

            # show_heatmap(ct, "cls_tower", height, width, stride)
            # show_heatmap(bt, "box_tower", height, width, stride)
            # cv2.destroyAllWindows()

            gb = (gt//stride).int()
            l, t, r, b = gb
            print(F"{stride} -> {r-l}x{b-t}")
            
            if (r-l)<1 or (b-t)<1:
                continue
            # cls, _ = torch.max(cls[0], dim=0)
            cls = cls[0][gid]

            regr = regr.view(N,4,H,W).permute(0,2,3,1)
            regr = regr.reshape(N,-1,4)
            regr = regr[0]

            mask = torch.zeros_like(cls)
            mask[l:r, t:b] = 1
            # mask[t:b, l:r] = 1
            mask = mask > 0

            # cls = cls[mask].sigmoid()
            cls = cls.flatten().sigmoid()

            minds = mask.view(-1).nonzero()
            print(F"{stride} -> step start: {minds[[0]]}")
            print(F"{stride} -> step end: {minds[[-1]]}")
            mid = (minds[[-1]]+minds[[0]])//2
            print(F"{stride} -> step mid: {mid}")
            minds = minds[:, 0]
            
            detections = torch.stack([
                location[:,0] - regr[:,0],
                location[:,1] - regr[:,1],
                location[:,0] + regr[:,2],
                location[:,1] + regr[:,3],
            ], dim=1)
            detections[:,0].clamp_(min=0, max=size[1]-1)
            detections[:,1].clamp_(min=0, max=size[0]-1)
            detections[:,2].clamp_(min=0, max=size[1]-1)
            detections[:,3].clamp_(min=0, max=size[0]-1)
            
            # detections = detections[minds]

            if stride in [-1]:#32, 64, 128]:
                for det in detections:
                    ccimg = cimg.copy()
                    cv2.rectangle(ccimg, (det[0],det[1]), (det[2], det[3]), (0,0,255), 2)
                    cv2.imshow("cimg", ccimg)
                    cv2.waitKey(-1)
            if stride == 128:
                cv2.destroyAllWindows()

            iou = calc_iou(detections, gt)

            # can = False
            # if can:
            #     if len(cls) > topk:
            #         cls, inds = cls.topk(topk, sorted=False)
            #         print(F"{stride} -> top cls score: {(inds<mid).sum()} -- {(inds>mid).sum()}")
            #         iou = iou[inds]
            #     else:
            #         print(F"{stride} -> top cls score: {(minds.view(-1)<mid).sum()} -- {(minds.view(-1)>mid).sum()}")
            # else:
            #     if len(iou) > topk:
            #         iou, inds = iou.topk(topk, sorted=False)
            #         print(F"{stride} -> top iou score: {(inds<mid).sum()} -- {(inds>mid).sum()}")
            #         cls = cls[inds]
            #     else:
            #         print(F"{stride} -> top iou score: {(minds.view(-1)<mid).sum()} -- {(minds.view(-1)>mid).sum()}")

            results.append((torch.sqrt(cls), iou))
            per = np.array([cls.numpy(), iou.numpy()])
            print(np.corrcoef(per))
            
            
        colors = randomColors(len(box_cls))
        visual_core_edge_feats(zip(results, colors), strides)
    

if __name__ == '__main__':
    visual()

