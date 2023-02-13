#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F

import sys
import os 
from pathlib import Path
import numpy as np 
import math

from einops import rearrange, reduce
import pytorch_lightning as pl
from models.archs.sdf_decoder import * 
from models.archs.encoders.conv_pointnet import ConvPointnet
from utils import mesh, evaluate


class SdfModel(pl.LightningModule):

    def __init__(self, specs):
        super().__init__()

        self.specs = specs
    
        model_specs = self.specs["SdfModelSpecs"]
        self.num_layers = model_specs["num_layers"]
        self.hidden_dim = model_specs["hidden_dim"]
        self.latent_dim = model_specs["latent_dim"]
        self.dropout = model_specs.get("dropout", 0.0)
        self.latent_in = model_specs.get("latent_input", True)
        self.pos_enc = model_specs.get("pos_enc", False)
        self.skip_connection = model_specs.get("skip_connection", [4])
        self.tanh_act = model_specs.get("tanh_act", False)
        self.pn_hidden = model_specs.get("pn_hidden_dim", self.latent_dim)

        self.pointnet = ConvPointnet(c_dim=self.latent_dim, hidden_dim=self.pn_hidden, plane_resolution=64)
        
        self.model = ModulatedMLP(latent_size=self.latent_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers, 
                                dropout_prob=self.dropout, latent_in=self.latent_in, pos_enc=self.pos_enc,
                                skip_connection=self.skip_connection, tanh_act=self.tanh_act)
        
        self.model.train()
        #print(self.model)

        #print("encoder params: ", sum(p.numel() for p in self.pointnet.parameters() if p.requires_grad))
        #print("mlp params: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))


    def configure_optimizers(self):
    
        optimizer = torch.optim.Adam(self.parameters(), self.outer_lr)

        return optimizer 


 
    def training_step(self, x, idx):

        xyz = x['xyz'].cuda() # (B, 16000, 3)
        gt = x['gt_sdf'].cuda() # (B, 16000)
        pc = x['point_cloud'].cuda() # (B, 1024, 3)

        modulations = self.pointnet(pc, xyz) 

        pred_sdf, new_mod = self.model(xyz, modulations)

        sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction = 'none')
        sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()

        return sdf_loss 
            
    

    def forward(self, modulations, xyz):
        modulations = self.pointnet(modulations, xyz)
        return self.model(xyz, modulations)[0].squeeze()

    def forward_with_plane_features(self, plane_features, xyz):
        point_features = self.pointnet.forward_with_plane_features(plane_features, xyz)
        pred_sdf = self.model(xyz, point_features)[0].squeeze()
        return pred_sdf
