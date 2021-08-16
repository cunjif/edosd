"""可视化参数模块"""

import enum
import os
import torch
import matplotlib.pyplot as plt
import argparse
from random import choices
from fcos_core.config import cfg
from fcos_core.utils.checkpoint import DetectronCheckpointer
from visualer.network import GeneralizedRCNN
from fcos_core.data.build import make_data_loader
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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


def visual_scatter(dots, name):
    """visual special object's core feature response and \\
    edge feature response in special level feature maps  \\
    with its grouth truth box
    """
    plt.ion()
    f = plt.figure(1, figsize=(20,16))
    # f.canvas.set_window_title("fdosd")
    plt.xlabel("IoU", fontsize=18)
    plt.ylabel("Location Confidence", fontsize=16)
    plt.xlim(0,1)
    plt.ylim(0,1)
    
    iou, loc, color = dots
    plt.scatter(y=loc, x=iou, c=color, s=15)
    plt.savefig(name)


def visual():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
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

    data_loader = make_data_loader(cfg, is_train=False, is_distributed=False, shuffled=True)[0]
    network.eval()
    steps = cfg.MODEL.FCOS.FPN_STRIDES
    with torch.no_grad():
        max_step = 50
        ious = []
        locs = []
        idx = 0
        for batch in tqdm(data_loader):
            images, targets, _ = batch

            if len(targets[0].bbox) == 0:
                continue

            targets = [target.to(cfg.MODEL.DEVICE) for target in targets]

            (_, iou, loc), *_ = network(images.to(cfg.MODEL.DEVICE), targets)
            ious.append(iou.cpu())
            locs.append(loc.cpu())
            if (idx+1) % max_step == 0:
                filename = f"{vis_cen}/{idx}.jpg"  
                ious = torch.cat(ious, dim=0).numpy().tolist()
                locs = torch.cat(locs, dim=0).numpy().tolist()
                visual_scatter((ious, locs, "blue"), filename)
                ious = []
                locs = []
            idx += 1
               

if __name__ == '__main__':
    visual()

# cat = [ "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", 
# "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", 
# "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
# "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
# "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
# "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", 
# "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", 
# "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", 
# "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]