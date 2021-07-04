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


_COLORS = ["red", "blue", "yellow", "orange", "purple", "green"]


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
    # parser.add_argument("--image", type=str)
    # parser.add_argument("--bbox", type=str)
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

    network = GeneralizedRCNN(cfg)
    network.to(cfg.MODEL.DEVICE)
    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, network, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    data_loader = make_data_loader(cfg, is_train=False, is_distributed=False)[0]
    network.eval()
    with torch.no_grad():
        for batch in data_loader:
            images, targets, idx = batch
            image_id = images[0].image_id
            targets = [target.to(cfg.MODEL.DEVICE) for target in targets]

            cent = network(images.to(cfg.MODEL.DEVICE), targets)
            print(cent)
            exit()

if __name__ == '__main__':
    visual()

