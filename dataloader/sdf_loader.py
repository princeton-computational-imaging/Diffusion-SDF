#!/usr/bin/env python3

import time 
import logging
import os
import random
import torch
import torch.utils.data
from . import base 

import pandas as pd 
import numpy as np
import csv, json

from tqdm import tqdm

class SdfLoader(base.Dataset):

    def __init__(
        self,
        data_source, # path to points sampled around surface
        split_file, # json filepath which contains train/test classes and meshes 
        grid_source=None, # path to grid points; grid refers to sampling throughout the unit cube instead of only around the surface; necessary for preventing artifacts in empty space
        samples_per_mesh=16000,
        pc_size=1024,
        modulation_path=None # used for third stage of training; needs to be set in config file when some modulation training had been filtered
    ):
 
        self.samples_per_mesh = samples_per_mesh
        self.pc_size = pc_size
        self.gt_files = self.get_instance_filenames(data_source, split_file, filter_modulation_path=modulation_path)

        subsample = len(self.gt_files) 
        self.gt_files = self.gt_files[0:subsample]

        self.grid_source = grid_source
        #print("grid source: ", grid_source)
    
        if grid_source:
            self.grid_files = self.get_instance_filenames(grid_source, split_file, gt_filename="grid_gt.csv", filter_modulation_path=modulation_path)
            self.grid_files = self.grid_files[0:subsample]
            lst = []
            with tqdm(self.grid_files) as pbar:
                for i, f in enumerate(pbar):
                    pbar.set_description("Grid files loaded: {}/{}".format(i, len(self.grid_files)))
                    lst.append(torch.from_numpy(pd.read_csv(f, sep=',',header=None).values))
            self.grid_files = lst
            
            assert len(self.grid_files) == len(self.gt_files)


        # load all csv files first 
        print("loading all {} files into memory...".format(len(self.gt_files)))
        lst = []
        with tqdm(self.gt_files) as pbar:
            for i, f in enumerate(pbar):
                pbar.set_description("Files loaded: {}/{}".format(i, len(self.gt_files)))
                lst.append(torch.from_numpy(pd.read_csv(f, sep=',',header=None).values))
        self.gt_files = lst


    def __getitem__(self, idx): 

        near_surface_count = int(self.samples_per_mesh*0.7) if self.grid_source else self.samples_per_mesh

        pc, sdf_xyz, sdf_gt =  self.labeled_sampling(self.gt_files[idx], near_surface_count, self.pc_size, load_from_path=False)
        

        if self.grid_source is not None:
            grid_count = self.samples_per_mesh - near_surface_count
            _, grid_xyz, grid_gt = self.labeled_sampling(self.grid_files[idx], grid_count, pc_size=0, load_from_path=False)
            # each getitem is one batch so no batch dimension, only N, 3 for xyz or N for gt 
            # for 16000 points per batch, near surface is 11200, grid is 4800
            #print("shapes: ", pc.shape,  sdf_xyz.shape, sdf_gt.shape, grid_xyz.shape, grid_gt.shape)
            sdf_xyz = torch.cat((sdf_xyz, grid_xyz))
            sdf_gt = torch.cat((sdf_gt, grid_gt))
            #print("shapes after adding grid: ", pc.shape, sdf_xyz.shape, sdf_gt.shape, grid_xyz.shape, grid_gt.shape)

        data_dict = {
                    "xyz":sdf_xyz.float().squeeze(),
                    "gt_sdf":sdf_gt.float().squeeze(), 
                    "point_cloud":pc.float().squeeze(),
                    }

        return data_dict

    def __len__(self):
        return len(self.gt_files)



    
