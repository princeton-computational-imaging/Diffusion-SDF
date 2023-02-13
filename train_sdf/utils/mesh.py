#!/usr/bin/env python3

import logging
import math
import numpy as np
import plyfile
import skimage.measure
import time
import torch


# N: resolution of grid; 256 is typically sufficient 
# max batch: as large as GPU memory will allow
# shape_feature is either point cloud, mesh_idx (neuralpull), or generated latent code (deepsdf)
def create_mesh(
    model, shape_feature, filename, N=256, max_batch=1000000, level_set=0.0, occupancy=False, point_cloud=None, from_plane_features=False, from_pc_features=False
):
    
    start_time = time.time()
    ply_filename = filename

    model.eval()

    # the voxel_origin is the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)
    cube = create_cube(N)
    cube_points = cube.shape[0]

    head = 0
    while head < cube_points:
        
        query = cube[head : min(head + max_batch, cube_points), 0:3].unsqueeze(0)
        
        # inference defined in forward function per pytorch lightning convention
        #print("shapes: ", shape_feature.shape, query.shape)
        if from_plane_features:
            pred_sdf = model.forward_with_plane_features(shape_feature.cuda(), query.cuda()).detach().cpu()
        else:
            pred_sdf = model(shape_feature.cuda(), query.cuda()).detach().cpu()

        cube[head : min(head + max_batch, cube_points), 3] = pred_sdf.squeeze()
            
        head += max_batch
    
    # for occupancy instead of SDF, subtract 0.5 so the surface boundary becomes 0
    sdf_values = cube[:, 3] - 0.5 if occupancy else cube[:, 3] 
    sdf_values = sdf_values.reshape(N, N, N) 

    #print("inference time: {}".format(time.time() - start_time))

    convert_sdf_samples_to_ply(
        sdf_values.data,
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        level_set
    )


# create cube from (-1,-1,-1) to (1,1,1) and uniformly sample points for marching cube
def create_cube(N):

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # the voxel_origin is the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)
    
    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long().float() / N) % N
    samples[:, 0] = ((overall_index.long().float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    samples.requires_grad = False

    return samples



def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    level_set=0.0
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    # use marching_cubes_lewiner or marching_cubes depending on pytorch version 
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=level_set, spacing=[voxel_size] * 3
        )
    except Exception as e:
        print("skipping {}; error: {}".format(ply_filename_out, e))
        return

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)


