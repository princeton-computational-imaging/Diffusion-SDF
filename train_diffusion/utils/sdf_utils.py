import math
import torch
import json 
import torch.nn.functional as F
from torch import nn, einsum 
import os

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from sdf_model.model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# load base network
specs_path = "sdf_model/config/siren/specs.json"
specs = json.load(open(specs_path))
model = MetaSDF(specs).to(device)
checkpoint = torch.load("sdf_model/config/siren/last.ckpt", map_location=device)
model.load_state_dict(checkpoint['state_dict'])

for p in model.parameters():
    p.requires_grad=False

def pred_sdf_loss(x0, idx):

    pc, xyz, gt = sdf_sampling(gt_files[idx], 16000, batch=x0.shape[0])
    xyz = xyz.to(device)
    gt = gt.to(device)
    sdf_loss = functional_sdf_model(x0, xyz, gt)

    return sdf_loss


def functional_sdf_model(modulation, xyz, gt):
    '''
    modulation: modulation vector with shape 512
    xyz: query points, input to the model, dim= 16000x3
    gt: ground truth; calculate l1 loss with prediction, dim= 16000x1

    return: sdf_loss 
    '''
    #print("shapes: ", modulation.shape, xyz.shape)
    pred_sdf = model(modulation, xyz)
    sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze())

    return sdf_loss


def sdf_sampling(f, subsample, pc_size=1024, batch=1):
    # f=pd.read_csv(f, sep=',',header=None).values
    # f = torch.from_numpy(f)

    pcs = torch.empty(batch, pc_size, 3)
    xyz = torch.empty(batch, subsample, 3)
    gt = torch.empty(batch, subsample)

    for i in range(batch):
        half = int(subsample / 2) 
        neg_tensor = f[f[:,-1]<0]
        pos_tensor = f[f[:,-1]>0]

        if pos_tensor.shape[0] < half:
            pos_idx = torch.randint(pos_tensor.shape[0], (half,))
        else:
            pos_idx = torch.randperm(pos_tensor.shape[0])[:half]

        if neg_tensor.shape[0] < half:
            neg_idx = torch.randint(neg_tensor.shape[0], (half,))
        else:
            neg_idx = torch.randperm(neg_tensor.shape[0])[:half]

        pos_sample = pos_tensor[pos_idx]
        neg_sample = neg_tensor[neg_idx]

        pc = f[f[:,-1]==0][:,:3]
        pc_idx = torch.randperm(pc.shape[0])[:pc_size]
        pc = pc[pc_idx]

        samples = torch.cat([pos_sample, neg_sample], 0)


        pcs[i] = pc.float()
        xyz[i] = samples[:,:3].float()
        gt[i] = samples[:, 3].float()


    return pcs, xyz, gt

def apply_to_sdf(f, x):
    for idx, l in enumerate(x):
        x[idx] = f(l)
    return x