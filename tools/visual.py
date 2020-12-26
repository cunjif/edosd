import os
import torch
from tensorboardX import SummaryWriter
from fcos_core.utils.comm import get_rank
import cv2
import numpy as np
from torchvision.transforms import ToTensor, Compose, Resize, ToPILImage
from torchvision.utils import make_grid
from PIL import Image

transforms = Compose([
    ToPILImage(),
    Resize((128,128), interpolation=Image.NEAREST),
    ToTensor(),
])

LOGDIR = "logs"

sw: SummaryWriter = None

class Visioer:
    
    instance: SummaryWriter = None

    @classmethod
    def make_summarywriter(cls, directory):
        logdir = os.path.join(directory, LOGDIR)
        if os.path.exists(directory):
            if not os.path.exists(logdir):
                os.mkdir(logdir)
        else:
            os.makedirs(logdir)

        cls.instance = SummaryWriter(logdir=logdir)

    @classmethod
    def draw(cls, features, step=None):
        step = 0 if step is None else step
        for i, feature in enumerate(features):
            # if (i+1) % 2 == 0:
            #     continue
            
            appr = feature.appr[0].detach().cpu().sigmoid().numpy()
            agg = feature.agg[0].detach().cpu().sigmoid().numpy()
            appr_imgs = []
            agg_imgs = []

            for idx, (appr_img, agg_img) in enumerate(zip(appr, agg)):
                if idx==64:
                    break

                appr_img *= 255
                appr_img = appr_img.astype(np.uint8)
                appr_img = cv2.applyColorMap(appr_img, cv2.COLORMAP_JET)
                appr_img = transforms(appr_img)

                agg_img *= 255
                agg_img = agg_img.astype(np.uint8)
                agg_img = cv2.applyColorMap(agg_img, cv2.COLORMAP_JET)
                agg_img = transforms(agg_img)

                appr_imgs.append(appr_img)
                agg_imgs.append(agg_img)

            appr = make_grid(appr_imgs, nrow=8, padding=2)
            agg = make_grid(agg_imgs, nrow=8, padding=2)

            cls.instance.add_image(F"fpn_{i}/appr", appr, global_step=step)
            cls.instance.add_image(F"fpn_{i}/agg", agg, global_step=step)
            cls.instance.flush()