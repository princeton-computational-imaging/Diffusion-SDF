#!/usr/bin/env python3

import numpy as np
import time 
import logging
import os
import random
import torch
import torch.utils.data
from . import base 

import pandas as pd 
import csv

class TestAcronymDataset(base.Dataset):

    def __init__(
        self,
        data_source,
        split_file, # json filepath which contains train/test classes and meshes 
        subsample,
        pc_size=1024,
        gt_filename= "sdf_data.csv",
    ):

        super().__init__(data_source, split_file, subsample, gt_filename)
        self.gt_files = self.get_instance_filenames(data_source, split_file, gt_filename)
        self.pc_size = pc_size

    def __getitem__(self, idx): 
        np.random.seed()    
        pc, sdf_xyz, sdf_gt =  self.labeled_sampling(self.gt_files[idx], self.subsample, self.pc_size)

        mesh_name = self.gt_files[idx].split("/")[-3:-1] # class and mesh
        mesh_name = os.path.join(mesh_name[0],mesh_name[1])
        data_dict = {"point_cloud":pc,
                    "xyz":sdf_xyz,
                    "gt_sdf":sdf_gt,
                    "indices":idx,
                    "mesh_name":mesh_name,
                    }

        return data_dict

    def __len__(self):
        return len(self.gt_files)



    
