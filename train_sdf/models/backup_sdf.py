#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl 

import sys
import os 
from pathlib import Path
import numpy as np 
import math

from einops import rearrange, reduce

from models.archs.sdf_decoder import * 
from models.archs.encoders.conv_pointnet import ConvPointnet
from utils import mesh, evaluate


class SdfModel(pl.LightningModule):

    def __init__(self, specs):
        super().__init__()
        
        self.specs = specs
        model_specs = self.specs["SdfModelSpecs"]
        self.hidden_dim = model_specs["hidden_dim"]
        self.latent_dim = model_specs["latent_dim"]
        self.skip_connection = model_specs.get("skip_connection", True)
        self.tanh_act = model_specs.get("tanh_act", False)
        self.pn_hidden = model_specs.get("pn_hidden_dim", self.latent_dim)

        self.pointnet = ConvPointnet(c_dim=self.latent_dim, hidden_dim=self.pn_hidden, plane_resolution=64)
        
        self.model = SdfDecoder(latent_size=self.latent_dim, hidden_dim=self.hidden_dim, skip_connection=self.skip_connection, tanh_act=self.tanh_act)
        
        self.model.train()
        #print(self.model)


    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), self.specs["sdf_lr"])

        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5, verbose=False)
        return optimizer

 
    def training_step(self, x, idx):

        xyz = x['xyz'] # (B, 16000, 3)
        gt = x['gt_sdf'] # (B, 16000)
        pc = x['point_cloud'] # (B, 1024, 3)

        shape_features = self.pointnet(pc, xyz)

        pred_sdf = self.model(xyz, shape_features)

        sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction = 'none')
        sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()
    
        return sdf_loss 
            
    

    def forward(self, pc, xyz):
        shape_features = self.pointnet(pc, xyz)

        return self.model(xyz, shape_features).squeeze()

    def forward_with_plane_features(self, plane_features, xyz):
        '''
        plane_features: B, D*3, res, res (e.g. B, 768, 128, 128)
        xyz: B, N, 3
        '''
        point_features = self.pointnet.forward_with_plane_features(plane_features, xyz) # point_features: B, N, D
        pred_sdf = self.model( torch.cat((xyz, point_features),dim=-1) )  
        return pred_sdf # [B, num_points] 



    # take_mod: None, or string path to modulation 
    def reconstruct(self, model, test_data, eval_dir, take_mod):
        recon_samplesize_param = 64
        recon_batch = 100000000
        model.eval() 

        query = test_data['xyz'].float()
        gt = test_data['gt_sdf'].float()
        gt_pc = test_data['point_cloud'].float()
        mesh_name = test_data["mesh_name"]
        #indices = test_data["indices"]

        if not take_mod:
            modulations = gt_pc[:, torch.randperm(gt_pc.shape[1])[:1024] ]
            #print("mod shape: ", modulations.shape)

        else:
            modulations = np.loadtxt(take_mod[0])
            mod2 = np.loadtxt(take_mod[1]) if len(take_mod)>1 else modulations
            mod2 = torch.from_numpy(mod2).float()
            modulations = torch.from_numpy(modulations).float()
            #print("modulations shape: ", modulations.shape)
            #print("2mod shape: ", mod2.shape)
            print("diff: ", (modulations-mod2).mean(), torch.all(modulations==mod2))
            #modulations = (modulations*0.75 + mod2*0.25)
            modulations = modulations.unsqueeze(0)
            #exit()


        with torch.no_grad():
            Path(eval_dir).mkdir(parents=True, exist_ok=True)
            mesh_filename = os.path.join(eval_dir, "reconstruct") #ply extension added in mesh.py
            evaluate_filename = os.path.join("/".join(eval_dir.split("/")[:-2]), "evaluate.csv")

            #if not take_mod:
                #modulations_folder = os.path.join("/".join(eval_dir.split("/")[:-2]), "stored_mods")
                #Path(modulations_folder).mkdir(parents=True, exist_ok=True)
                #print("modulations folder, shape: ", modulations.shape, modulations_folder, mesh_name)

                #for idx, mod in enumerate(modulations):
                    #np.savetxt("{}/{}.txt".format(modulations_folder, idx), mod.cpu().numpy())
                    #np.savetxt("{}/{}/{}.txt".format(modulations_folder, mesh_name[0].split("/")[0], mesh_name[0].split("/")[1]), mod.cpu().numpy()) # need to first create directory for this
                #    np.savetxt("{}/{}.txt".format(modulations_folder, mesh_name[0].split("/")[1]), mod.cpu().numpy()) 


            mesh.create_mesh(model, modulations, mesh_filename, recon_samplesize_param, recon_batch)
            try:
                evaluate.main(gt_pc, mesh_filename, evaluate_filename, mesh_name)
            except Exception as e:
                print(e)






    



