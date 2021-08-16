"""可视化分类和回归的特征热图 单张"""

import os
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import argparse
from random import choices
from fcos_core.config import cfg
from fcos_core.utils.checkpoint import DetectronCheckpointer
from visualer.network import GeneralizedRCNN
from fcos_core.data.build import make_data_loader
import numpy as np
import cv2
from tqdm import tqdm

torch.set_printoptions(precision=3, sci_mode=False, threshold=65536)


_COLORS = ["red", "blue", "yellow", "orange", "purple", "green"]
CATEGORIES = [ "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", 
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def randomColors(k):
    colors = _COLORS[:k]
    return colors


def visual_core_edge_feats(dots):
    """visual special object's core feature response and \\
    edge feature response in special level feature maps  \\
    with its grouth truth box
    """
    plt.ion()
    f = plt.figure(1, figsize=(20,16))
    f.canvas.set_window_title("fdosd")
    plt.xlabel("IoU", fontsize=18)
    plt.ylabel("score w/ IoU-ness", fontsize=16)
    plt.xlim(0,1)
    plt.ylim(0,1)
    # plt.title("fdosd")
    
    cls, iou, color = dots
        # print(cls)
        # da = 1 / len(cls)
        # a = 1
        # for idx,(c,i) in enumerate(zip(cls, iou)):
        #     plt.scatter(y=c, x=i, c=color, alpha=a)
        #     # plt.annotate(F"{idx}", (i, c))
        #     if a-da != 0:
        #         a -= da
    plt.scatter(y=cls, x=iou, c=color, s=15)
    
    plt.show()
    input(F"enter to continue")
    plt.close('all')

def show_heatmaps(features, h, w):
    features = features[0]
    # features = features.cpu().sigmoid().numpy()
    imgs = np.zeros((h*16+16, w*16+16, 3), dtype=np.uint8)
    
    r = []
    for idx,feat in enumerate(features):
        feat = ((feat - feat.min()) / (feat.max() - feat.min())) * 255
        feat = feat.cpu().numpy()
        # feat = feat * 255
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

    return imgs


def show_heatmap(feature, h, w):
    feat = feature[0]
    feat = ((feat - feat.min()) / (feat.max() - feat.min())) * 255
    feat = feat.cpu().numpy()
    feat = feat.astype(np.uint8)
    feat = cv2.resize(feat, (w, h))
    img = cv2.applyColorMap(feat, cv2.COLORMAP_JET)
    return img


def visual():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    # parser.add_argument("--image", type=str)
    parser.add_argument("--heat_dir", type=str)
    # parser.add_argument("--cat", type=int)
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

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    vis_dir = args.heat_dir
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir) 

    network = GeneralizedRCNN(cfg)
    network.to(cfg.MODEL.DEVICE)
    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, network, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    data_loader = make_data_loader(cfg, is_train=False, is_distributed=False)[0]
    network.eval()
    steps = cfg.MODEL.FCOS.FPN_STRIDES
    img_h, img_w = 0, 0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            images, targets, _ = batch
            assert len(targets)==1

            if not targets[0].image_id in [285, 52017, 460927, 132622, 1584, 145591, 
                 94614, 355905]:
                continue

            img_dir = f"{vis_dir}/{targets[0].image_id}"
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)

            targets = [target.to(cfg.MODEL.DEVICE) for target in targets]

            _, cls_heats, regr_heats  = network(images.to(cfg.MODEL.DEVICE), targets)
            if img_h == 0 or img_w == 0:
                w, h = targets[0].size
                img_h = h // 2
                img_w = w // 2

            for step, cls_heat, regr_heat in zip(steps, cls_heats, regr_heats):
                img_path = f"{img_dir}/{{}}-{step}.jpg"
                cls_heat = (F.adaptive_max_pool2d(cls_heat, 1).softmax(dim=1) * cls_heat).sum(dim=1)
                regr_heat = (F.adaptive_max_pool2d(regr_heat, 1).softmax(dim=1) * regr_heat).sum(dim=1)
                # cls_heat = cls_heat.sum(dim=1) / 256
                # regr_heat = regr_heat.sum(dim=1) / 256
                cls_heatmap = show_heatmap(cls_heat, img_h, img_w)
                regr_heatmap = show_heatmap(regr_heat, img_h, img_w)

                cv2.imwrite(img_path.format("cls"), cls_heatmap)
                cv2.imwrite(img_path.format("reg"), regr_heatmap)
                        

if __name__ == '__main__':
    visual()
