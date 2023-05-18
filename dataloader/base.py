#!/usr/bin/env python3

import numpy as np
import time 
import logging
import os
import random
import torch
import torch.utils.data

import pandas as pd 
import csv

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split_file, # json filepath which contains train/test classes and meshes 
        subsample,
        gt_filename,
        #pc_size=1024,
    ):

        self.data_source = data_source 
        self.subsample = subsample
        self.split_file = split_file
        self.gt_filename = gt_filename
        #self.pc_size = pc_size

        # example
        # data_source: "data"
        # ws.sdf_samples_subdir: "SdfSamples"
        # self.gt_files[0]: "acronym/couch/meshname/sdf_data.csv"
            # with gt_filename="sdf_data.csv"

    def __len__(self):
        return NotImplementedError

    def __getitem__(self, idx):     
        return NotImplementedError

    def sample_pointcloud(self, csvfile, pc_size):
        f=pd.read_csv(csvfile, sep=',',header=None).values

        f = f[f[:,-1]==0][:,:3]

        if f.shape[0] < pc_size:
            pc_idx = np.random.choice(f.shape[0], pc_size)
        else:
            pc_idx = np.random.choice(f.shape[0], pc_size, replace=False)

        return torch.from_numpy(f[pc_idx]).float()

    def labeled_sampling(self, f, subsample, pc_size=1024, load_from_path=True):
        if load_from_path:
            f=pd.read_csv(f, sep=',',header=None).values
            f = torch.from_numpy(f)

        half = int(subsample / 2) 
        neg_tensor = f[f[:,-1]<0]
        pos_tensor = f[f[:,-1]>0]

        if pos_tensor.shape[0] < half:
            pos_idx = torch.randint(0, pos_tensor.shape[0], (half,))
        else:
            pos_idx = torch.randperm(pos_tensor.shape[0])[:half]

        if neg_tensor.shape[0] < half:
            if neg_tensor.shape[0]==0:
                neg_idx = torch.randperm(pos_tensor.shape[0])[:half] # no neg indices, then just fill with positive samples
            else:
                neg_idx = torch.randint(0, neg_tensor.shape[0], (half,))
        else:
            neg_idx = torch.randperm(neg_tensor.shape[0])[:half]

        pos_sample = pos_tensor[pos_idx]

        if neg_tensor.shape[0]==0:
            neg_sample = pos_tensor[neg_idx]
        else:
            neg_sample = neg_tensor[neg_idx]

        pc = f[f[:,-1]==0][:,:3]
        pc_idx = torch.randperm(pc.shape[0])[:pc_size]
        pc = pc[pc_idx]

        samples = torch.cat([pos_sample, neg_sample], 0)

        return pc.float().squeeze(), samples[:,:3].float().squeeze(), samples[:, 3].float().squeeze() # pc, xyz, sdv


    def get_instance_filenames(self, data_source, split, gt_filename="sdf_data.csv", filter_modulation_path=None):
            
            do_filter = filter_modulation_path is not None 
            csvfiles = []
            for dataset in split: # e.g. "acronym" "shapenet"
                for class_name in split[dataset]:
                    for instance_name in split[dataset][class_name]:
                        instance_filename = os.path.join(data_source, dataset, class_name, instance_name, gt_filename)

                        if do_filter:
                            mod_file = os.path.join(filter_modulation_path, class_name, instance_name, "latent.txt")

                            # do not load if the modulation does not exist; i.e. was not trained by diffusion model
                            if not os.path.isfile(mod_file):
                                continue
                        
                        if not os.path.isfile(instance_filename):
                            logging.warning("Requested non-existent file '{}'".format(instance_filename))
                            continue

                        csvfiles.append(instance_filename)
            return csvfiles
