#!/usr/bin/env python3

import time 
import logging
import os
import random
import torch
import torch.utils.data
from diff_utils.helpers import * 

import pandas as pd 
import numpy as np
import csv, json

from tqdm import tqdm

class ModulationLoader(torch.utils.data.Dataset):
    def __init__(self, data_path, pc_path=None, split_file=None, pc_size=None):
        super().__init__()

        self.conditional = pc_path is not None 

        if self.conditional:
            self.modulations, pc_paths = self.load_modulations(data_path, pc_path, split_file)
        else:
            self.modulations = self.unconditional_load_modulations(data_path, split_file)
        #self.modulations = self.modulations[0:8]
        #pc_paths = pc_paths[0:8]

        print("data shape, dataset len: ", self.modulations[0].shape, len(self.modulations))
        #assert args.batch_size <= len(self.modulations)
        
        if self.conditional:
            print("loading ground truth point clouds...")            
            lst = []
            with tqdm(pc_paths) as pbar:
                for i, f in enumerate(pc_paths):
                    pbar.set_description("Point clouds loaded: {}/{}".format(i, len(pc_paths)))
                    lst.append(sample_pc(f, pc_size))
            self.point_clouds = lst

            assert len(self.point_clouds) == len(self.modulations)
        
        
    def __len__(self):
        return len(self.modulations)

    def __getitem__(self, index):

        pc = self.point_clouds[index] if self.conditional else False
        return {
            "point_cloud" : pc,
            "latent" : self.modulations[index]         
        }
        

    def load_modulations(self, data_source, pc_source, split, f_name="latent.txt", add_flip_augment=False, return_filepaths=True):
        #split = json.load(open(split))
        files = []
        filepaths = [] # return filepaths for loading pcs
        for dataset in split: # dataset = "acronym" 
            for class_name in split[dataset]:
                for instance_name in split[dataset][class_name]:

                    if add_flip_augment:
                        for idx in range(4):
                            instance_filename = os.path.join(data_source, class_name, instance_name, "latent_{}.txt".format(idx))
                            if not os.path.isfile(instance_filename):
                                print("Requested non-existent file '{}'".format(instance_filename))
                                continue
                            files.append( torch.from_numpy(np.loadtxt(instance_filename)).float() )
                        filepaths.append( os.path.join(pc_source, dataset, class_name, instance_name, "sdf_data.csv") )

                    else:
                        instance_filename = os.path.join(data_source, class_name, instance_name, f_name)
                        if not os.path.isfile(instance_filename):
                            #print("Requested non-existent file '{}'".format(instance_filename))
                            continue
                        files.append( torch.from_numpy(np.loadtxt(instance_filename)).float() )
                        filepaths.append( os.path.join(pc_source, dataset, class_name, instance_name, "sdf_data.csv") )
        if return_filepaths:
            return files, filepaths
        return files

    def unconditional_load_modulations(self, data_source, split, f_name="latent.txt", add_flip_augment=False):
        files = []
        for dataset in split: # dataset = "acronym" 
            for class_name in split[dataset]:
                for instance_name in split[dataset][class_name]:

                    if add_flip_augment:
                        for idx in range(4):
                            instance_filename = os.path.join(data_source, class_name, instance_name, "latent_{}.txt".format(idx))
                            if not os.path.isfile(instance_filename):
                                print("Requested non-existent file '{}'".format(instance_filename))
                                continue
                            files.append( torch.from_numpy(np.loadtxt(instance_filename)).float() )

                    else:
                        instance_filename = os.path.join(data_source, class_name, instance_name, f_name)
                        if not os.path.isfile(instance_filename):
                            continue
                        files.append( torch.from_numpy(np.loadtxt(instance_filename)).float() )
        return files