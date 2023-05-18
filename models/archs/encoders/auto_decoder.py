#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import json
import math

class AutoDecoder():
    # specs is a json filepath that contains the specifications for the experiment 
    # also requires total num_scenes (all meshes from all classes), which is given by len of dataset
    def __init__(self, num_scenes, latent_size):
        self.num_scenes = num_scenes
        self.latent_size = latent_size

    def build_model(self):
        lat_vecs = nn.Embedding(self.num_scenes, self.latent_size, max_norm=1.0)
        nn.init.normal_(lat_vecs.weight.data, 0.0, 1.0/math.sqrt(self.latent_size))
        return lat_vecs
