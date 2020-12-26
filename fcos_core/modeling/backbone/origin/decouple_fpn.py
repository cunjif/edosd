import torch
from torch import nn
from easydict import EasyDict
import torch.nn.functional as F
from fcos_core.layers.batch_norm import FrozenBatchNorm2d
from fcos_core.layers.scale import Scale
from fcos_core.layers import DFConv2d, DeformConv


class DecoupleBranch(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DecoupleBranch, self).__init__()

        self.agg_down  = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.agg_trans = DFConv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=True)

        self.appr_down  = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.appr_trans = DFConv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=True)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features):
        agg = self.agg_down(features)
        agg = self.agg_trans(agg)
        agg = F.relu(agg)

        appr = self.appr_down(features)
        appr = self.appr_trans(appr)
        appr = F.relu(appr)

        return EasyDict({
            "agg": agg,
            "appr": appr
        })     


class FPNAddFuseModule(nn.Module):
    def __init__(self, channels, kernel_size=3, num_convs=None): #channels, num_convs=None):
        super().__init__()
        
        self.surf = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.surf_scale = Scale()
        
        self.conv_sum = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU()
        )

        self.use_tower = num_convs is not None and num_convs
        if num_convs:
            conv_tower = []
            for _ in range(num_convs):
                conv_tower.append(nn.Conv2d(channels, channels, 3, 1, 1))
                conv_tower.append(nn.GroupNorm(32, channels))
                conv_tower.append(nn.ReLU())
            self.convs = nn.Sequential(*conv_tower)

        self.weight = nn.Conv2d(
            channels,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

        self._init_weight()

    def _init_weight(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 1e-6)
            elif isinstance(m, nn.Sequential):
                for mod in m.modules():
                    if isinstance(mod, nn.Conv2d):
                        nn.init.kaiming_normal_(mod.weight, a=1)
                        if mod.bias is not None:
                            nn.init.constant_(mod.bias, 1e-6)
        self.apply(_init)

    def forward(self, agg, appr, oappr=None):
        assert oappr is None
        
        x = self.conv_sum(self.surf_scale(self.surf(agg))+appr)
        if self.use_tower:
            x = self.convs(x)

        op = oappr if oappr is not None else appr

        # print(F"decouple fpn 104 oappr: {oappr}")
        
        add_weight = torch.sigmoid(self.weight(x))
        x = add_weight * x + (1 - add_weight) * op

        return x


class DecoupleFPN(nn.Module):
    def __init__(self, in_channels_list,
        out_channels,
        num_outs,
        conv_block,
        bottom_block,
        top_block,
        agg_up_fuses=[True, True, True, True],
        appr_down_fuses=[False, False, False, False], # 0,1,2,3,4
        agg_appr_fuses=[True, True, True, True, False], # 0,1,2,3,4
        fuse_conv_num = [1, 1, 1, 1, 0],
        use_stride_4=False
    ):
        assert len(agg_up_fuses) == 4
        assert len(appr_down_fuses) == 4

        super(DecoupleFPN, self).__init__()

        self.in_channels = in_channels_list
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.use_stride_4 = use_stride_4

        self.agg_up_fuses = agg_up_fuses + [False]
        self.appr_down_fuses = [False] + appr_down_fuses
        self.agg_appr_fuses = agg_appr_fuses
        
        self.decouple_blocks = []
        self.semantic_blocks = []
        self.appearance_blocks = []
        self.fuse_blocks = {}

        if not use_stride_4:
            self.bottom_appr = bottom_block

        outs = 0
        ind_fuse = 0

        # 0,1,2,3
        for idx, in_channel in enumerate(in_channels_list):
            if in_channel == 0:
                continue
            
            outs += 1
            
            decouple_block = F"defpn_decouple{idx}"
            semantic_block = F"defpn_semantic{idx}"
            appearance_block = F"defpn_appearance{idx}"
            fuse_block = F"defpn_aggapprfuse{ind_fuse}"

            decouple_module = DecoupleBranch(in_channel, out_channels)
            semantic_module = conv_block(out_channels, out_channels, 3, 1)
            appearance_module = conv_block(out_channels, out_channels, 3, 1)
            
            self.add_module(decouple_block, decouple_module)
            self.add_module(semantic_block, semantic_module)
            self.add_module(appearance_block, appearance_module)
            
            self.decouple_blocks.append(decouple_block)
            self.semantic_blocks.append(semantic_block)
            self.appearance_blocks.append(appearance_block)

            if self.agg_appr_fuses[ind_fuse]:
                fuse_module = FPNAddFuseModule(out_channels, num_convs=fuse_conv_num[ind_fuse])
                self.add_module(fuse_block, fuse_module)
                self.fuse_blocks[ind_fuse] = fuse_block

            ind_fuse += 1
        
        self.top_decouples = top_block

        if num_outs > outs:
            for i in range(num_outs - outs):
                semantic_block = F"defpn_semantic_t{i}"
                appearance_block = F"defpn_appearance_t{i}"
                fuse_block = F"defpn_aggapprfuse_t{ind_fuse}"

                semantic_module = conv_block(out_channels, out_channels, 3, 1)
                appearance_module = conv_block(out_channels, out_channels, 3, 1)
                
                if agg_appr_fuses[ind_fuse]:
                    fuse_module = FPNAddFuseModule(out_channels, num_convs=fuse_conv_num[ind_fuse])
                    self.add_module(fuse_block, fuse_module)
                    self.fuse_blocks[ind_fuse] = fuse_block

                self.add_module(semantic_block, semantic_module)
                self.add_module(appearance_block, appearance_module)

                self.semantic_blocks.append(semantic_block)
                self.appearance_blocks.append(appearance_block)
                
                ind_fuse += 1

        self._init_weight()

    def _freeze(self):
        for name, m in self.named_parameters():
            if name.startswith("bottom_appr"):
                continue
            m.requires_grad = False

    def _init_weight(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 1e-6)
        self.apply(_init)
        
    def forward(self, features):
        assert len(features) == len(self.in_channels)
        
        if self.use_stride_4:
            decouples = [getattr(self, block)(feature)
                            for block, feature in zip(self.decouple_blocks, features)]

            if self.num_outs > len(features):
                decouples.extend(self.top_decouples(features[-1]))

            decouples = self._forward(decouples)
        else:

            decouples = [getattr(self, block)(feature)
                            for block, feature in zip(self.decouple_blocks, features[1:])]
            
            # bottom_features = self.bottom_appr(features[0], None)
            decouples[0] = self.bottom_appr(features[0], decouples[0])

            if self.num_outs > len(features)-1:
                decouples.extend(self.top_decouples(features[-1]))
            
            decouples = self._forward(decouples)
        
        results = [
            EasyDict({
                "agg": getattr(self, semantic)(decouple.agg),
                "appr": getattr(self, appearance)(decouple.appr)
            }) for semantic, appearance, decouple in zip(
                self.semantic_blocks,
                self.appearance_blocks,
                decouples)
        ]

        return results

    def _forward(self, decouples):
        sizes = [d.agg.shape[-2:] for d in decouples]
        if any(self.appr_down_fuses):
            assert False    # if obtain FPN-like regressive feature, annotate it
            apprs = [d.appr for d in decouples]

        # top-bottom
        for i, agg_up in enumerate(self.agg_up_fuses[::-1]):
            if agg_up:
                decouples[-(i+1)].agg = decouples[-(i+1)].agg + F.interpolate(decouples[-i].agg,
                    size=sizes[-(i+1)], mode='bilinear', align_corners=True)
        
        for i, appr_downed in enumerate(self.appr_down_fuses):
            if appr_downed:
                assert False
                # top -> bottom
                decouples[-(i+1)].appr = decouples[-(i+1)].appr \
                    + F.interpolate(
                        decouples[-i].appr, 
                        size=sizes[-(i+1)], 
                        mode='bilinear', 
                        align_corners=True)
        
        for i, (fused, appred) in enumerate(zip(self.agg_appr_fuses, self.appr_down_fuses)):
            if fused:
                fuse_block = self.fuse_blocks[i]
                if appred:
                    assert False  # if obtain FPN-like regressive feature, annotate it
                    decouples[i].appr = getattr(self, fuse_block)(decouples[i].agg.detach(), apprs[i], decouples[i].appr)
                else:
                    decouples[i].appr = getattr(self, fuse_block)(decouples[i].agg.detach(), decouples[i].appr)

        return decouples