from typing import Any
import pytorch_lightning as pl
import torch.nn as nn
from models.heads import Line_Heads, Plane_Heads, CAM, SAM, CA, CA_w
import torch

class reinforce_attention(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.plane_cam_0 = CAM(256)
        self.plane_cam_1 = CAM(256)
        self.line_cam_0 = CAM(256)
        self.line_cam_1 = CAM(256)
        self.conv_0 = nn.Conv2d(256*2,256,3,1,1,groups=256)
        self.plane_fc = nn.Linear(256*2, 256,bias=False)
        self.line_fc = nn.Linear(256*2, 256,bias=False)

        self.plane_sam_0 = SAM()
        self.plane_sam_1 = SAM()
        self.line_ca_0 = CA_w()
        self.line_ca_1 = CA_w()
        self.conv_1 = nn.Conv2d(2,1,7,padding=7//2,bias=False)
        self.conv_2 = nn.Conv2d(256*2,256,1,bias=False)

        self.line_heads = Line_Heads()
        self.plane_heads = Plane_Heads()
        pass

    def forward(self, x):
        b = x.shape[0]
        plane_xishu_0 = self.plane_cam_0(x) # (batch_size,256,1,1)
        plane_feature = plane_xishu_0.expand_as(x) * x # (batch_size,256,96,160)
        line_xishu_0 = self.line_cam_0(x) # (batch_size,256,1,1)
        line_feature = line_xishu_0.expand_as(x) * x # (batch_size,256,96,160)

        feature_plane_line = torch.cat([plane_feature,line_feature],dim=1) # (batch_size,256*2,96,160)
        feature_plane_line = self.conv_0(feature_plane_line) # (batch_size,256,96,160)
        plane_xishu_1 = self.plane_cam_1(feature_plane_line) # (batch_size,256,1,1)
        line_xishu_1 = self.line_cam_1(feature_plane_line) # (batch_size,256,1,1)
        plane_xishu_cat = torch.cat([plane_xishu_0,plane_xishu_1],dim=1).view(b,256*2) # (batch_size,256*2,1,1)
        line_xishu_cat = torch.cat([line_xishu_0,line_xishu_1],dim=1).view(b,256*2) # (batch_size,256*2,1,1)
        plane_xishu_cam = self.plane_fc(plane_xishu_cat).view(b,256,1,1) # (batch_size,256,1,1)
        line_xishu_cam = self.line_fc(line_xishu_cat).view(b,256,1,1) # (batch_size,256,1,1)
        plane_feature = plane_xishu_cam.expand_as(x) * x # (batch_size,256,96,160)
        line_feature = line_xishu_cam.expand_as(x) * x # (batch_size,256,96,160)
        identity_plane = plane_feature
        identity_line = line_feature

        plane_sam_xishu_0 = self.plane_sam_0(plane_feature) # (batch_size,1,96,160)
        plane_feature = plane_feature * plane_sam_xishu_0.expand_as(x) # # (batch_size,256,96,160)
        line_ca_xishu_0 = self.line_ca_0(line_feature) # (batch_size,256,160,1)
        line_feature = line_feature * line_ca_xishu_0.permute(0,1,3,2).expand_as(x)
        feature_plane_line = plane_feature + line_feature

        plane_sam_xishu_1 = self.plane_sam_1(feature_plane_line) # (batch_size,1,96,160)
        line_ca_xishu_1 = self.line_ca_1(feature_plane_line) # (batch_size,256,160,1)
        plane_xishu_cat = torch.cat([plane_sam_xishu_0,plane_sam_xishu_1],dim=1) # (batch_size,2,96,160)
        plane_xishu = self.conv_1(plane_xishu_cat)
        plane_xishu_show = plane_xishu.cpu().numpy()

        plane_xishu = plane_xishu.expand_as(x)
        line_xishu_cat = torch.cat([line_ca_xishu_0,line_ca_xishu_1],dim=1)
        line_xishu = self.conv_2(line_xishu_cat).permute(0,1,3,2)
        line_xishu_show = line_xishu.cpu().numpy()

        line_xishu = line_xishu.expand_as(x)
        plane_feature = plane_xishu * identity_plane
        line_feature = line_xishu * identity_line
        

        x_plane = self.plane_heads(plane_feature)
        x_line = self.line_heads(line_feature)
        x_plane.update(x_line)
        return x_plane