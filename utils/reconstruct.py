#!/usr/bin/env python3

import torch
import torch.utils.data 
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning import loggers as pl_loggers

import os
import json
import time
from tqdm.auto import tqdm
from einops import rearrange, reduce
import numpy as np
import trimesh

# add paths in model/__init__.py for new models
from models import * 
from utils import mesh, evaluate, reconstruct
from diff_utils.helpers import * 

def vis_recon(test_dataloader, sdf_model, vae_model, recon_dir, take_mod=False, calc_cd=False):
    resolution = 64
    recon_batch = 2**20
    # run visualization of the reconstructed plane features to confirm that recon loss is low enough 
    with torch.no_grad():
        if args.evaluate:
            point_clouds, pc_paths = test_dataloader.get_all_files()

            point_clouds = torch.stack(point_clouds) # stack [(1024,3), (1024,3)...] to (B, 1024, 3)

            recon_meshes = torch.empty(*point_clouds.shape)

            # ***change to MODULATION generated paths later!!!!
            for idx, path in enumerate(pc_paths):
                #print("path: ", path)
                cls_name = path.split("/")[-3]
                mesh_name = path.split("/")[-2]
                mesh_filename = os.path.join(recon_dir, "{}/{}/reconstruct".format(cls_name, mesh_name))
                recon_mesh = trimesh.load(os.path.join(os.getcwd(), mesh_filename)+".ply")
                recon_pc, _ = trimesh.sample.sample_surface(recon_mesh, point_clouds.shape[1])
                recon_meshes[idx] = torch.from_numpy(recon_pc)

            print("ref, recon shapes: ", recon_meshes.shape, point_clouds.shape) # should both be = B, N, 3
            results = evaluation_metrics.compute_all_metrics(recon_meshes.float(), point_clouds.float(), accelerated_cd=False)
            for k,v in results.items():
                print(k, ": ", v)

        elif args.take_mod and not args.sample:

            lst = []
            if args.mod_folder:
                files = os.listdir(args.mod_folder)
                for f in files:
                    if os.path.isfile(os.path.join(args.mod_folder, f)) and f[-4:]=='.txt':
                        lst.append(os.path.join(args.mod_folder, f))
            else:
                lst = args.take_mod

            for idx, m in enumerate(lst):
                latent = torch.from_numpy(np.loadtxt(m)).float().cuda()
                recon = vae_model.decode(latent) 
                name = args.output_name if args.output_name else "mod_recon"
                name += "{}".format(idx)
                os.makedirs(os.path.join(recon_dir, "modulation_recon"), exist_ok=True)
                mesh_filename = os.path.join(recon_dir, "modulation_recon", name)
                mesh.create_mesh(sdf_model, recon, mesh_filename, resolution, recon_batch, from_plane_features=True)
        elif args.sample:
            recon = vae_model.sample(num_samples=1)
            name = args.output_name if args.output_name else "mod_recon"
            os.makedirs(os.path.join(recon_dir, "modulation_recon"), exist_ok=True)
            mesh_filename = os.path.join(recon_dir, "modulation_recon", name)
            mesh.create_mesh(sdf_model, recon, mesh_filename, resolution, recon_batch, from_plane_features=True)
        else:
            for idx, data in enumerate(test_dataloader): # test_loader does not shuffle 
                # if idx % 10 != 0:
                #     continue
                data, filename = data # filename = path to the csv file of sdf data
                filename = filename[0] # filename is a tuple for some reason


                random_flip = specs.get("random_flip", False)
                # if random_flip:
                #     flip_axes = torch.tensor([[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1]], device=data.device)
                #     prob = torch.randint(low=0, high=4, size=(1,))
                #     flip_axis = flip_axes[prob] # shape=[1,3]
                #     data *= flip_axis.unsqueeze(0).repeat(data.shape[0], data.shape[1], 1)

                # if random_flip:
                #     flip_axes = torch.tensor([[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1]], device=data.device)
                #     for axis in flip_axes:
                #         flipped_data = data * axis.unsqueeze(0).repeat(data.shape[0], data.shape[1], 1)


                cls_name = filename.split("/")[-3]
                mesh_name = filename.split("/")[-2]
                outdir = os.path.join(recon_dir, "{}/{}".format(cls_name, mesh_name))
                os.makedirs(outdir, exist_ok=True)
                mesh_filename = os.path.join(outdir, "reconstruct")
               
                plane_features = sdf_model.pointnet.get_plane_features(data.cuda())  # 3 items with ([1, 256, 64, 64])
                plane_features = torch.cat(plane_features, dim=1) # ([1, 768, 64, 64])
                recon = vae_model.generate(plane_features) # ([1, 768, 64, 64])

                # create_mesh samples the grid points, then calls sdf_model.forward_with_plane_features, which calls pointnet.forward_with_plane_features
                #print("mesh filename: ", mesh_filename)
                mesh.create_mesh(sdf_model, recon, mesh_filename, resolution, recon_batch, from_plane_features=True)

                
                if calc_cd:
                    evaluate_filename = os.path.join(recon_dir, "cd.csv")
                    mesh_log_name = cls_name+"/"+mesh_name
                    try:
                        evaluate.main(data, mesh_filename, evaluate_filename, mesh_log_name)
                    except Exception as e:
                        print(e)

                # try:
                #     if not filter_threshold(mesh_filename, data, 0.0018):
                #         continue
                #     outdir = os.path.join(latent_dir, "{}/{}".format(cls_name, mesh_name))
                #     os.makedirs(outdir, exist_ok=True)
                #     features = sdf_model.pointnet.get_plane_features(data.cuda())
                #     #print("features shape: ", features[0].shape) # ([1, 256, 64, 64])
                #     features = torch.cat(features, dim=1)
                #     latent = vae_model.get_latent(features)

                #     #print("latent shape: ", latent.shape)
                #     np.savetxt(os.path.join(outdir, "latent.txt"), latent.cpu().numpy())
                # except Exception as e:
                #     print(e)


           

def filter_threshold(mesh, gt_pc, threshold): # mesh is path to mesh without .ply ext
    cd = evaluate.main(gt_pc, mesh, None, None, return_value=True, prioritize_cov=True)
    return cd <= threshold



def extract_latents(test_dataloader, sdf_model, vae_model, save_dir):
    # only extract the latent vectors 
    latent_dir = os.path.join(save_dir, "modulations")
    os.makedirs(latent_dir, exist_ok=True)
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader): # test_loader does not shuffle 

            data, filename = data
            filename = filename[0] # filename is a tuple for some reason
            cls_name = filename.split("/")[-3]
            mesh_name = filename.split("/")[-2]

            # if filtering based on CD threshold
            saved_mesh = os.path.join(recon_dir, "{}/{}/reconstruct".format(cls_name, mesh_name))
            gt_pc = data
            try:
                if not filter_threshold(saved_mesh, gt_pc, 0.0022):
                    continue

                outdir = os.path.join(latent_dir, "{}/{}".format(cls_name, mesh_name))
                os.makedirs(outdir, exist_ok=True)

                random_flip = specs.get("random_flip", False)
                if random_flip:
                    flip_axes = torch.tensor([[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1]], device=data.device)
                    for idx, axis in enumerate(flip_axes):
                        flipped_data = data * axis.unsqueeze(0).repeat(data.shape[0], data.shape[1], 1)
                
                        features = sdf_model.pointnet.get_plane_features(flipped_data.cuda())
                        #print("features shape: ", features[0].shape) # ([1, 256, 64, 64])
                        features = torch.cat(features, dim=1)
                        latent = vae_model.get_latent(features)

                        #print("latent shape: ", latent.shape)
                        np.savetxt(os.path.join(outdir, "latent_{}.txt".format(idx)), latent.cpu().numpy())
                
                else:
                    features = sdf_model.pointnet.get_plane_features(data.cuda())
                    #print("features shape: ", features[0].shape) # ([1, 256, 64, 64])
                    features = torch.cat(features, dim=1)
                    latent = vae_model.get_latent(features)

                    #print("latent shape: ", latent.shape)
                    np.savetxt(os.path.join(outdir, "latent.txt"), latent.cpu().numpy())

            except Exception as e:
                print(e)

