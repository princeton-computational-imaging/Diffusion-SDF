#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


class SdfDecoder(nn.Module):
    def __init__(self, latent_size=256, hidden_dim=512,
                 skip_connection=True, tanh_act=False,
                 geo_init=True, input_size=None
                 ):
        super().__init__()
        self.latent_size = latent_size
        self.input_size = latent_size+3 if input_size is None else input_size
        self.skip_connection = skip_connection
        self.tanh_act = tanh_act

        skip_dim = hidden_dim+self.input_size if skip_connection else hidden_dim 

        self.block1 = nn.Sequential(
            nn.Linear(self.input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Linear(skip_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )


        self.block3 = nn.Linear(hidden_dim, 1)

        if geo_init:
            for m in self.block3.modules():
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, mean=2 * np.sqrt(np.pi) / np.sqrt(hidden_dim), std=0.000001)
                    init.constant_(m.bias, -0.5)

            for m in self.block2.modules():
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, mean=0.0, std=np.sqrt(2) / np.sqrt(hidden_dim))
                    init.constant_(m.bias, 0.0)

            for m in self.block1.modules():
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, mean=0.0, std=np.sqrt(2) / np.sqrt(hidden_dim))
                    init.constant_(m.bias, 0.0)


    def forward(self, x):
        '''
        x: concatenated xyz and shape features, shape: B, N, D+3 
        '''        
        block1_out = self.block1(x)

        # skip connection, concat 
        if self.skip_connection:
            block2_in = torch.cat([x, block1_out], dim=-1) 
        else:
            block2_in = block1_out

        block2_out = self.block2(block2_in)

        out = self.block3(block2_out)

        if self.tanh_act:
            out = nn.Tanh()(out)

        return out
