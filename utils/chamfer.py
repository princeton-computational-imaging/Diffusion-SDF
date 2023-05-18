import numpy as np
from scipy.spatial import cKDTree as KDTree


def compute_trimesh_chamfer(gt_points, gen_points, offset=0, scale=1):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.
    gt_points: numpy array. trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)
    gen_mesh: numpy array. trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)
    """

    # gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]

    gen_points = gen_points / scale - offset

    # one direction
    gen_points_kd_tree = KDTree(gen_points)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer


def scale_to_unit_sphere(points):
    """
    scale point clouds into a unit sphere
    :param points: (n, 3) numpy array
    :return:
    """
    midpoints = (np.max(points, axis=0) + np.min(points, axis=0)) / 2
    points = points - midpoints
    scale = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    points = points / scale
    return points
