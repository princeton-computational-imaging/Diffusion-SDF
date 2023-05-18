import numpy as np
import trimesh
import sys
import h5py
import open3d as o3d

def create_point_marker(center, color):

    # point cloud point info (e.g. radius)
    point = trimesh.primitives.Sphere(
        radius=0.002, 
        center=center
    )
    point.visual.vertex_colors = color
    return point

def get_color(labels):
    return np.stack([np.ones(labels.shape[0]) - labels, labels, np.zeros(labels.shape[0])], axis=1)


def vis_pc(obj_pc):
    point_color = get_color(np.array([1.0]))
    point_markers = [create_point_marker(center=obj_pc[t], color=point_color[0]) for t in range(len(obj_pc))]

    trimesh.Scene(point_markers).show()

def main(mesh, pc, query):

    lst = []

    if pc:
        point_color = get_color(np.array([0.5]))
        point_markers = [create_point_marker(center=point_cloud[t], color=point_color[0]) for t in range(len(point_cloud))]
        lst.append(trimesh.Scene(point_markers))

    if query: 
        q_color = get_color(np.array([0.0]))
        q_markers = [create_point_marker(center=queries[t], color=q_color[0]) for t in range(len(queries))]
        lst.append(trimesh.Scene(q_markers))

    if mesh:
        lst.append(object_mesh)

    scene = trimesh.scene.scene.append_scenes( lst ).show() 
    #trimesh.exchange.export.export_scene(scene, "vis.ply") 


object_mesh = trimesh.load(sys.argv[1])
object_mesh.apply_scale(0.1)
pc = object_mesh.vertices

point_cloud = np.loadtxt(sys.argv[2], dtype=float, delimiter=',') # num of points x 3
p_idx = np.random.choice(point_cloud.shape[0], 10000)
point_cloud = point_cloud[p_idx][:,0:3]

# for visualizing pc only
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(point_cloud)
# #o3d.io.write_point_cloud("./pc.ply", pcd)
# o3d.visualization.draw_geometries([pcd])


#     mesh, pc, query
main(False, True, False)

