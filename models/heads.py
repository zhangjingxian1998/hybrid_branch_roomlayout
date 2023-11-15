import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
from mmcv.ops import DeformConv2dPack as DCN

class CAM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SAM(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAM, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CA(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CA, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x): # 要w
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y

class CA_w(nn.Module):
    def __init__(self, reduction=16):
        super(CA_w, self).__init__()
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(256,256//reduction,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(256//reduction)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(256//reduction,256,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(256)

    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        x_w = self.pool_w(x).permute(0, 1, 3, 2) # (batch_size,256,160,1)
        x_w = self.conv1(x_w)
        x_w = self.bn1(x_w)
        x_w = self.relu(x_w)
        x_w = self.conv2(x_w)
        x_w = self.bn2(x_w).sigmoid()

        return x_w

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16,full=True):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.full = full
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        if self.full:
            return x * y.expand_as(x),y
        else:
            return y

class SENet(nn.Module):
    def __init__(self):
        super().__init__()
        self.plane_se = SELayer(256)
        self.line_se = SELayer(256)
        self.plane_head = Plane_Heads()
        self.line_head = Line_Heads()
    
    def forward(self,x):
        plane_se,_ = self.plane_se(x)
        line_se,_ = self.line_se(x)

        x_plane = self.plane_head(plane_se)
        x_line = self.line_head(line_se)
        x_plane.update(x_line)
        return x_plane

class HRMerge(pl.LightningModule):
    def __init__(self,
                 in_channels=(32, 64, 128, 256),
                 out_channels=256,
                 normalize=None):
        super(HRMerge, self).__init__()
        assert isinstance(in_channels, list) or isinstance(in_channels, tuple)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.with_bias = normalize is None
        self.reduction_conv = nn.Sequential(
            nn.Conv2d(in_channels=sum(in_channels),
                      out_channels=out_channels,
                      kernel_size=1),
        )

        self.fpn_conv = nn.Conv2d(in_channels=out_channels,
                                  out_channels=out_channels,
                                  kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # default init_weights for conv(msra) and norm in ConvModule

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        outs = []
        outs.append(inputs[0])
        for i in range(1, len(inputs)):
            outs.append(F.interpolate( # 特征图放缩为相同大小
                inputs[i], scale_factor=2 ** i, mode='bilinear',align_corners=True))
        out = torch.cat(outs, dim=1) # 不同尺度的特征图拼接，就是特征融合吧，让低层特征和高层特征互相杂糅

        out = self.reduction_conv(out)
        out = self.relu(out)
        out = self.fpn_conv(out)
        return out

class HRMerge_deconv(pl.LightningModule):
    def __init__(self,
                 in_channels=(32, 64, 128, 256),
                 out_channels=256,
                 normalize=None):
        super(HRMerge_deconv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.with_bias = normalize is None
        self.reduction_conv = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=1)
        self.bn = nn.BatchNorm2d(256)
        self.fpn_conv = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        reduction_conv1 = nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size = 8,stride=8)
        bn1 = nn.BatchNorm2d(128)
        reduction_conv2 = nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=4,stride=4)
        bn2 = nn.BatchNorm2d(64)
        reduction_conv3 = nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=2,stride=2)
        bn3 = nn.BatchNorm2d(32)

        self.reduction_conv_list = nn.ModuleList([reduction_conv3,reduction_conv2,reduction_conv1])
        self.bn_list = nn.ModuleList([bn3,bn2,bn1])
        # default init_weights for conv(msra) and norm in ConvModule

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        outs = []
        outs.append(inputs[0])
        for i in range(1, len(inputs)):
            feature = self.reduction_conv_list[i-1](inputs[i])
            feature = self.bn_list[i-1](feature)
            feature = self.relu(feature)
            outs.append(feature)
        out = torch.cat(outs, dim=1) # 不同尺度的特征图拼接，就是特征融合吧，让低层特征和高层特征互相杂糅

        # out = self.reduction_conv(out)
        # out = self.relu(out)
        out = self.fpn_conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out
    

class HRMerge_fpn_jiangwei(pl.LightningModule):
    def __init__(self,
                 in_channels=(32, 64, 128, 256),
                 out_channels=256,
                 normalize=None):
        super(HRMerge_fpn_jiangwei, self).__init__()
        assert isinstance(in_channels, list) or isinstance(in_channels, tuple)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.with_bias = normalize is None
        self.reduction_conv = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=1)
        self.bn = nn.BatchNorm2d(256)
        self.fpn_conv = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        reduction_conv1 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=1)
        bn1 = nn.BatchNorm2d(128)
        reduction_conv2 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=1)
        bn2 = nn.BatchNorm2d(64)
        reduction_conv3 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=1)
        bn3 = nn.BatchNorm2d(32)

        self.reduction_conv_list = nn.ModuleList([reduction_conv3,reduction_conv2,reduction_conv1])
        self.bn_list = nn.ModuleList([bn3,bn2,bn1])
        # default init_weights for conv(msra) and norm in ConvModule

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        outs = []
        outs.append(inputs[0])
        for i in range(1, len(inputs)):
            feature = self.reduction_conv_list[i-1](inputs[i])
            feature = self.bn_list[i-1](feature)
            feature = self.relu(feature)
            outs.append(F.interpolate(feature, scale_factor=2 ** i, mode='bilinear',align_corners=True))
        out = torch.cat(outs, dim=1) # 不同尺度的特征图拼接，就是特征融合吧，让低层特征和高层特征互相杂糅

        # out = self.reduction_conv(out)
        # out = self.relu(out)
        out = self.fpn_conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out
    
class HRMerge_fpn_jiangwei_wo_relu_bf_up(pl.LightningModule):
    def __init__(self,
                 in_channels=(32, 64, 128, 256),
                 out_channels=256,
                 normalize=None):
        super(HRMerge_fpn_jiangwei_wo_relu_bf_up, self).__init__()
        assert isinstance(in_channels, list) or isinstance(in_channels, tuple)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.with_bias = normalize is None
        self.reduction_conv = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=1)
        self.bn = nn.BatchNorm2d(256)
        self.fpn_conv = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        reduction_conv1 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=1)
        bn1 = nn.BatchNorm2d(128)
        reduction_conv2 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=1)
        bn2 = nn.BatchNorm2d(64)
        reduction_conv3 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=1)
        bn3 = nn.BatchNorm2d(32)

        self.reduction_conv_list = nn.ModuleList([reduction_conv3,reduction_conv2,reduction_conv1])
        self.bn_list = nn.ModuleList([bn3,bn2,bn1])
        # default init_weights for conv(msra) and norm in ConvModule

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        outs = []
        outs.append(inputs[0])
        for i in range(1, len(inputs)):
            feature = self.reduction_conv_list[i-1](inputs[i])
            feature = self.bn_list[i-1](feature)
            # feature = self.relu(feature)
            outs.append(F.interpolate(feature, scale_factor=2 ** i, mode='bilinear',align_corners=True))
        out = torch.cat(outs, dim=1) # 不同尺度的特征图拼接，就是特征融合吧，让低层特征和高层特征互相杂糅

        # out = self.reduction_conv(out)
        # out = self.relu(out)
        out = self.fpn_conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

class Plane_Heads(pl.LightningModule):
    def __init__(self, in_planes=256, out_planes=64):
        super(Plane_Heads, self).__init__()
        self.plane_center = nn.Sequential(
            # nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1),
            DCN(in_planes, in_planes, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=2),
            nn.ReLU(True),
            nn.Conv2d(in_planes, 3, kernel_size=1)
        )

        self.plane_xy = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_planes, 2, kernel_size=1)
        )

        self.plane_wh = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_planes, 2, kernel_size=1)
        )

        self.plane_params_pixelwise = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_planes, 4, kernel_size=1)
        )

        self.plane_params_instance = nn.Sequential(
            # nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1),
            DCN(in_planes, in_planes, kernel_size=(3, 3), stride=(1, 1), padding=1, deform_groups=2),
            nn.ReLU(True),
            nn.Conv2d(in_planes, 4, kernel_size=1)
        )

    def forward(self, x):
        plane_params_pixelwise = self.plane_params_pixelwise(x)
        plane_center = self.plane_center(x)
        plane_wh = self.plane_wh(x)
        plane_xy = self.plane_xy(x)
        plane_params_instance = self.plane_params_instance(x)

        out = {
            'plane_center': plane_center,
            'plane_offset': plane_xy,
            'plane_wh': plane_wh,
            'plane_params_pixelwise': plane_params_pixelwise,
            'plane_params_instance': plane_params_instance,
            # 'feature': x
        }
        return out

class Line_Heads(pl.LightningModule):
    def __init__(self, in_planes=256, out_planes=64):
        super(Line_Heads, self).__init__()
        self.line_region = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_planes, 1, kernel_size=1)
        )

        self.line_params = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_planes, 2, kernel_size=1)
        )
    
    def forward(self, x):
        line_region = self.line_region(x)
        line_params = self.line_params(x)

        out = {
            'line_region': line_region,
            'line_params': line_params,
            # 'feature': x
        }
        return out