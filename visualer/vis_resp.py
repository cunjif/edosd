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

def show_heatmap(feat, h, w):
    # features = features.cpu().sigmoid().numpy()
    
    feat = ((feat - feat.min()) / (feat.max() - feat.min())) * 255
    feat = feat.cpu().numpy()
    # feat = feat * 255
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
    # parser.add_argument("--bbox", type=str)
    parser.add_argument("--heat_dir", type=str)
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

    vis_cen = args.heat_dir
    if not os.path.exists(vis_cen):
        os.makedirs(vis_cen) 

    network = GeneralizedRCNN(cfg)
    network.to(cfg.MODEL.DEVICE)
    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, network, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    data_loader = make_data_loader(cfg, is_train=False, is_distributed=False)[0]
    network.eval()
    steps = cfg.MODEL.FCOS.FPN_STRIDES
    with torch.no_grad():
        for batch in tqdm(data_loader):
            images, targets, _ = batch

            assert len(targets) == 1

            if len(targets[0].bbox) == 0:
                continue

            targets = [target.to(cfg.MODEL.DEVICE) for target in targets]

            img_dir = f"{vis_cen}/{targets[0].image_id}"
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            
            resps, _, _  = network(images.to(cfg.MODEL.DEVICE), targets)
            if resps is None:
                continue

            confidences, ious, centerness = resps
            w, h = targets[0].size
            h = int(h/2)
            w = int(w/2)
            # print(h,w);exit()
            for step, confidence, iou in zip(steps, confidences, ious):
                filename = f"{img_dir}/{{}}-{step}.jpg"
                cls_resp = show_heatmap(confidence, h, w)
                reg_resp = show_heatmap(iou, h, w)
                cv2.imwrite(filename.format("cls"), cls_resp)
                cv2.imwrite(filename.format("reg"), reg_resp)
            # exit()

           
                        

if __name__ == '__main__':
    # vis_cen = "vis_response_heatmap_10k"
    visual()
