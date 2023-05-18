#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
from torchmeta.modules import (MetaModule, MetaSequential, MetaLinear, MetaConv1d, MetaBatchNorm1d)

import json
import sys
sys.path.append("..")
import utils # actual dir is ../utils

class PointNet(MetaModule):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        
        self.conv1 = MetaConv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = MetaConv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = MetaConv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = MetaConv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = MetaConv1d(128, self.latent_size, kernel_size=1, bias=False)
        self.bn1 = MetaBatchNorm1d(64)
        self.bn2 = MetaBatchNorm1d(64)
        self.bn3 = MetaBatchNorm1d(64)
        self.bn4 = MetaBatchNorm1d(128)
        self.bn5 = MetaBatchNorm1d(self.latent_size)

        # self.layers = MetaSequential(
        #     self.conv1, self.bn1, nn.ReLU(),
        #     self.conv2, self.bn2, nn.ReLU(),
        #     self.conv3, self.bn3, nn.ReLU(),
        #     self.conv4, self.bn4, nn.ReLU(),
        #     self.conv5, self.bn5,
        #     )


    # def forward(self, x, params=None):
    #     x = self.layers(x, params=self.get_subdict(params, 'pointnet'))
    #     x = x.max(dim=2, keepdim=False)[0]
    #     return x

    def forward(self, x, params=None):
        x = F.relu(self.bn1(self.conv1(x, params)))
        x = F.relu(self.bn2(self.conv2(x, params)))
        x = F.relu(self.bn3(self.conv3(x, params)))
        x = F.relu(self.bn4(self.conv4(x, params)))
        x = self.bn5(self.conv5(x, params))
        x = x.max(dim=2, keepdim=False)[0]
        return x
