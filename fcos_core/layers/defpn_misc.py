import torch
from torch import nn
import torch.nn.functional as F

class PxAug(nn.Module):
    def __init__(self, channels, 
        pool_sizes=None, 
        use_max_pool=False,
        avg_sizes=None, 
        use_adapt_pool=False,
        need_fuse=False
    ):
        super(PxAug, self).__init__()

        self.use_adapt_pool = use_adapt_pool
        self.use_max_pool = use_max_pool
        self.need_fuse = need_fuse
        
        self.avg_pools = nn.ModuleList()
        if use_adapt_pool:
            for k in avg_sizes:
                self.avg_pools.append(nn.AdaptiveAvgPool2d(k))
            self.conv_adap = nn.Sequential(
                nn.Conv2d(channels*len(avg_sizes), channels, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True)
            )
        
        self.max_pools = nn.ModuleList()
        if use_max_pool:
            for k in pool_sizes:
                self.max_pools.append(nn.MaxPool2d(k, 1, padding=(k-1)//2))
            self.conv_max = nn.Sequential(
                nn.Conv2d(len(pool_sizes)*channels, channels, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True)
            )

        if need_fuse:
            self.conv_sum = nn.Sequential(
                nn.Conv2d(len(pool_sizes)*channels, channels, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True)
            )
        
        self._init_weight()

    def _init_weight(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 1e-6)
        self.apply(_init)

    def forward(self, x, need_fuse=False):
        x_size = x.size()[-2:]

        if use_max_pool:
            out = []
            for pool in self.max_pools:
                out.append(pool(x))    
            x_max = self.conv_max(torch.cat(out, dim=1))

        if self.use_adapt_pool:
            out = []
            for pool in self.avg_pools:
                out.append(F.interpolate(pool(x), size=x_size, mode="bilinear", align_corners=True))
            x_avg = self.conv_adap(torch.cat(out, dim=1))

        if self.need_fuse:
            x = self.conv_sum(x + x_max + x_avg)
            return x
        elif self.use_max_pool:
            x = self.conv_sum(x + x_max)
            return x
        elif self.use_adapt_pool:
            x = self.conv_sum(x + x_avg)
            return x
        else:
            return x


class FPNConcatFuseModule(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, dilation=1, groups=1):
        super(FPNConcatFuseModule, self).__init__()

        padding = 0 if ksize==1 else (ksize-1)*dilation//2
        self.surf = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.surf_scale = Scale()
        self.downfuse = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, dilation=dilation, groups=groups),
            nn.ReLU(inplace=True)
        )

        self._init_weight()

    def _init_weight(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 1e-6)
        self.apply(_init)

    def forward(self, agg, appr):
        x = torch.cat([self.surf_scale(self.surf(agg)), appr], dim=1)
        x = self.downfuse(x)
        return x



