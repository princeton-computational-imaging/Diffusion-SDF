#!/usr/bin/env python3

import time 
import logging
import os
import random
import torch
import torch.utils.data
from . import base 
from tqdm import tqdm

import pandas as pd 
import csv

class PCloader(base.Dataset):

    def __init__(
        self,
        data_source,
        split_file, # json filepath which contains train/test classes and meshes 
        pc_size=1024,
        return_filename=False
    ):

        self.pc_size = pc_size
        self.gt_files = self.get_instance_filenames(data_source, split_file)
        self.return_filename = return_filename

        self.pc_paths = self.get_instance_filenames(data_source, split_file)
        self.pc_paths = self.pc_paths[:5] 
        print("loading {} point clouds into memory...".format(len(self.pc_paths)))
        lst = []
        with tqdm(self.pc_paths) as pbar:
            for i, f in enumerate(pbar):
                pbar.set_description("Files loaded: {}/{}".format(i, len(self.pc_paths)))
                lst.append(self.sample_pc(f, pc_size))
        self.point_clouds = lst

        #print("each pc shape: ", self.point_clouds[0].shape)

    def get_all_files(self):
        return self.point_clouds, self.pc_paths 
    
    def __getitem__(self, idx): 
        if self.return_filename:
            return self.point_clouds[idx], self.pc_paths[idx]
        else:
            return self.point_clouds[idx]


    def __len__(self):
        return len(self.point_clouds)


    def sample_pc(self, f, samp=1024): 
        '''
        f: path to csv file
        '''
        # data = torch.from_numpy(np.loadtxt(f, delimiter=',')).float()
        data = torch.from_numpy(pd.read_csv(f, sep=',',header=None).values).float()
        pc = data[data[:,-1]==0][:,:3]
        pc_idx = torch.randperm(pc.shape[0])[:samp] 
        pc = pc[pc_idx]
        #print("pc shape, dtype: ", pc.shape, pc.dtype) # [1024,3], torch.float32
        #pc = normalize_pc(pc)
        #print("pc shape: ", pc.shape, pc.max(), pc.min())
        return pc



    
