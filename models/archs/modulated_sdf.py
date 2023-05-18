#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import json
import sys
import torch.nn.init as init
import numpy as np


# no dropout or skip connections for now 

class Layer(nn.Module):
    def __init__(self, dim_in=512, dim_out=512, dim=512, dropout_prob=0.0, geo_init='first', activation='relu'):
        super().__init__()

        self.linear = nn.Linear(dim_in, dim_out)
        if activation=='relu':
            self.activation = nn.ReLU() 
        elif activation=='tanh':
            self.activation = nn.Tanh() 
        else:
            self.activation = nn.Identity()

        #self.dropout = nn.Dropout(p=dropout_prob)

        if geo_init == 'first':
            init.normal_(self.linear.weight, mean=0.0, std=np.sqrt(2) / np.sqrt(dim))
            init.constant_(self.linear.bias, 0.0)
        elif geo_init == 'last':
            init.normal_(self.linear.weight, mean=2 * np.sqrt(np.pi) / np.sqrt(dim), std=0.000001)
            init.constant_(self.linear.bias, -0.5)

    
    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        #out = self.dropout(out)

        return out 



class ModulatedMLP(nn.Module):
    def __init__(self, latent_size=512, hidden_dim=512, num_layers=9, latent_in=True,
                 skip_connection=[4], dropout_prob=0.0, pos_enc=False, pe_num_freq=5, tanh_act=False
                 ):
        super().__init__()

        self.skip_connection = skip_connection # list of the indices of layer to add skip connection
        self.hidden_dim = hidden_dim
        self.pe_num_freq = pe_num_freq
        self.pos_enc = pos_enc
        self.latent_in = latent_in

        #print("tanh act: ", tanh_act)
        #print("latent in, skip layer: ", latent_in, skip_connection)

        first_dim_in = 3

        # must remove last tanh layer if using positional encoding 
        # posititional encoding
        if pos_enc:
            pe_func = []
            for i in range(self.pe_num_freq):
                pe_func.append(lambda data, freq=i: torch.sin(data * (2**i)))
                pe_func.append(lambda data, freq=i: torch.cos(data * (2**i)))
            self.pe_func = pe_func
            first_dim_in = 3*pe_num_freq*2 


        # latent code concatenated to coordinates as input to model
        if latent_in:
            num_modulations = hidden_dim
            #num_modulations = hidden_dim * (num_layers - 1)
            first_dim_in += latent_size # num_modulations
            mod_act = nn.ReLU()

        else: # use shifting instead of concatenation
            # We modulate features at every *hidden* layer of the base network and
            # therefore have dim_hidden * (num_layers - 1) modulations, since the last layer is not modulated
            num_modulations = hidden_dim * (num_layers - 1)
            mod_act = nn.Identity() 
        
        #self.mod_net = nn.Sequential(nn.Linear(latent_size, num_modulations), mod_act)

        layers = []
        #print("index, dim in: ", end='')
        for i in range(num_layers-1):
            if i==0:
                dim_in = first_dim_in
            elif i in skip_connection:
                dim_in = hidden_dim+3+latent_size #num_modulations+3+hidden_dim
            else:
                dim_in = hidden_dim

            #print(i, dim_in, end = '; ')

            layers.append(
                Layer(
                    dim_in=dim_in,
                    dim_out=hidden_dim,
                    activation='relu',
                    geo_init='first',
                    dim=hidden_dim,
                    dropout_prob=dropout_prob
                )
            )

        self.net = nn.Sequential(*layers)
        last_act = 'tanh' if tanh_act else 'identity'
        self.last_layer = Layer(dim_in=hidden_dim,dim_out=1,activation=last_act,geo_init='last',dim=hidden_dim)


    def pe_transform(self, data):
        pe_data = torch.cat([f(data) for f in self.pe_func], dim=-1)
        return pe_data
    def forward(self, xyz, latent):
        '''
        xyz: B, 16000, 3 (query coordinates for predicting)
        latent: B, 512 (latent vector from 3 gradient steps)
        '''
        #print("latent: ",latent.shape)
        modulations = latent#self.mod_net(latent)
        #print("mod size: ",modulations.shape, xyz.shape)
        #print("latent size: ", latent.shape, modulations.shape) # B,512 and B,512

        if self.pos_enc:
            xyz = self.pe_transform(xyz)

        x = xyz.clone()

        if self.latent_in:
        #    modulations = modulations.unsqueeze(-2).repeat(1,xyz.shape[1],1) # [B, 16000, 512] or [B, 16000, 256]
            #print("repeated mod shape: ",modulations.shape)
            x = torch.cat((x, modulations),dim=-1)

        #print("input size: ", x.shape, xyz.shape) # [8, 16000, 515], [8, 16000, 3]

        idx = 0

        for i, layer in enumerate(self.net):

            if i in self.skip_connection:
                x = torch.cat(( x, torch.cat((xyz, modulations),dim=-1)), dim=-1)
            
            x = layer.linear(x)
            if not self.latent_in:
                shift = modulations[:, idx : idx + self.hidden_dim].unsqueeze(1)
                x = x + shift
                idx += self.hidden_dim
            x = layer.activation(x)
            #x = layer.dropout(x)

        out = self.last_layer(x)

        return out, modulations


    
    









