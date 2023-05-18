#!/usr/bin/env python3

import argparse
import logging
import json
import numpy as np
import pandas as pd 
import os, sys
import trimesh
from scipy.spatial import cKDTree as KDTree

from utils import uhd, tmd

import csv

def main(gt_pc, recon_mesh, out_file, mesh_name, return_value=False, return_sampled_pc=False, prioritize_cov=False, pc_size=None):

    gt_pc = gt_pc.cpu().detach().numpy().squeeze()

    recon_mesh = trimesh.load(os.path.join(os.getcwd(), recon_mesh)+".ply")

    recon_pc, _ = trimesh.sample.sample_surface(recon_mesh, gt_pc.shape[0])

    full_recon_pc = trimesh.sample.sample_surface(recon_mesh, pc_size)[0] if pc_size is not None else recon_pc

    recon_kd_tree = KDTree(recon_pc)
    one_distances, one_vertex_ids = recon_kd_tree.query(gt_pc)
    gt_to_recon_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_kd_tree = KDTree(gt_pc)
    two_distances, two_vertex_ids = gt_kd_tree.query(recon_pc)
    recon_to_gt_chamfer = np.mean(np.square(two_distances))
    
    if prioritize_cov: # higher CD for gaps/holes
        loss_chamfer = gt_to_recon_chamfer * 2.0 + recon_to_gt_chamfer * 0.5
    else:
        loss_chamfer = gt_to_recon_chamfer + recon_to_gt_chamfer

    if return_value:
        return loss_chamfer

    out_file = os.path.join(os.getcwd(), out_file)

    with open(out_file,"a",) as f:
        writer = csv.writer(f)
        writer.writerow([mesh_name,loss_chamfer])

    if return_sampled_pc:
        return full_recon_pc, loss_chamfer


def calc_cd(gt_pc, recon_pc):

    gt_pc = gt_pc.cpu().detach().numpy().squeeze()

    recon_kd_tree = KDTree(recon_pc)
    one_distances, one_vertex_ids = recon_kd_tree.query(gt_pc)
    gt_to_recon_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_kd_tree = KDTree(gt_pc)
    two_distances, two_vertex_ids = gt_kd_tree.query(recon_pc)
    recon_to_gt_chamfer = np.mean(np.square(two_distances))
    
    return gt_to_recon_chamfer + recon_to_gt_chamfer


def single_eval(gt_csv, recon_mesh):
    # f=pd.read_csv(gt_csv, sep=',',header=None).values
    # f = f[f[:,-1]==0][:,:3]

    recon_mesh = trimesh.load( recon_mesh )
    recon_pc, _ = trimesh.sample.sample_surface(recon_mesh, 30000)
    print("recon pc min max: ", recon_pc.max(), recon_pc.min())
    # load from SIREN .xyz file 
    f = np.genfromtxt(gt_csv)
    pc = f[:,:3]
    coord_max = np.amax(pc, axis=0, keepdims=True)
    coord_min = np.amin(pc, axis=0, keepdims=True)
    coords = (pc - coord_min) / (coord_max - coord_min)
    coords -= 0.5
    coords *= 2.
    # pc -= np.mean(pc, axis=0, keepdims=True)
    # bbox_length = np.sqrt( np.sum((np.max(pc, axis=0) - np.min(pc, axis=0))**2) )
    # pc /= bbox_length
    f = coords
    print("f min max: ", f.max(), f.min())

    pc_idx = np.random.choice(f.shape[0], 30000, replace=False)
    gt_pc = f[pc_idx] 

    recon_mesh = trimesh.load( recon_mesh )
    recon_pc, _ = trimesh.sample.sample_surface(recon_mesh, 30000)

    recon_kd_tree = KDTree(recon_pc)
    one_distances, one_vertex_ids = recon_kd_tree.query(gt_pc)
    gt_to_recon_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_kd_tree = KDTree(gt_pc)
    two_distances, two_vertex_ids = gt_kd_tree.query(recon_pc)
    recon_to_gt_chamfer = np.mean(np.square(two_distances))
    
    loss_chamfer = gt_to_recon_chamfer + recon_to_gt_chamfer

    print("CD loss: ", loss_chamfer)



if __name__ == "__main__":
    single_eval(sys.argv[1], sys.argv[2])
