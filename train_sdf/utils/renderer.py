import numpy as np
import trimesh
import trimesh.transformations as tra
import pyrender
import matplotlib.pyplot as plt 
from scipy.spatial.transform import Rotation as R

COLORS = [
    np.array([255, 10, 10, 255]), 
    np.array([10, 255, 10, 255]), 
    np.array([10, 234, 255, 255])
]
class OnlineObjectRenderer:
    def __init__(self, fov=np.pi / 6, caching=True):
        """
        Args:
          fov: float, 
        """
        self._fov = fov
        self._scene = None
        self._init_scene()
        self._caching = caching
        self._nodes = []

    def _init_scene(self, height=480, width=480):
        self._scene = pyrender.Scene()
        camera = pyrender.PerspectiveCamera(yfov=self._fov, znear=0.001) # do not change aspect ratio
        camera_pose = tra.euler_matrix(np.pi, 0, 0)
        self._scene.add(camera, pose=camera_pose, name='camera')
        direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
        self._scene.add(direc_l, pose=camera_pose)
        self.renderer = pyrender.OffscreenRenderer(height, width)
        
    def add_mesh(self, path, name, rotation=None, translation=None):
        
        mesh = trimesh.load(path)
        color = np.tile(COLORS[len(self._nodes) % len(COLORS)], (mesh.vertices.shape[0], 1))
        # mesh_mean = np.mean(mesh.vertices, 0)
        # mesh.vertices -= np.expand_dims(mesh_mean, 0)
        mesh.visual.vertex_colors = color
        mesh = pyrender.Mesh.from_trimesh(mesh.copy(), smooth=False)
        if rotation is not None and (isinstance(rotation, list) or isinstance(rotation, tuple)):
            rotation = np.array(rotation)
        if rotation is not None and rotation.shape == (3, 3):
            rotation = R.from_matrix(rotation).as_quat()
        elif rotation is not None and rotation.shape == (3,):
            rotation = R.from_euler("xyz", rotation).as_quat()
        if translation is None:
            translation = np.array((0, 0, 3))
        else:
            translation = translation.copy()
            translation[2] += 3.
        node = pyrender.Node(mesh=mesh, rotation=rotation, translation=translation, name=name)
        self._scene.add_node(node)
        self._nodes.append(node)
        return name
    
    def add_pointcloud(self, path, name, colors=None, rotation=None, translation=None):
        pc = np.loadtxt(path, delimiter=",")
        sm = trimesh.creation.uv_sphere(radius=0.008)
        if colors is None:
            colors = COLORS[len(self._nodes) % len(COLORS)]
        sm.visual.vertex_colors = colors
        tfs = np.tile(np.eye(4), (len(pc), 1, 1))
        tfs[:,:3,3] = pc
        if rotation is not None and (isinstance(rotation, list) or isinstance(rotation, tuple)):
            rotation = np.array(rotation)
        if rotation is not None and rotation.shape == (3, 3):
            rotation = R.from_matrix(rotation).as_quat()
        elif rotation is not None and rotation.shape == (3,):
            rotation = R.from_euler("xyz", rotation).as_quat()
        if translation is None:
            translation = np.array((0, 0, 3))
        else:
            translation = translation.copy()
            translation[2] += 3.
        # pts = rotation @ pc + translation
        mesh = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        node = pyrender.Node(mesh=mesh, rotation=rotation, translation=translation, name=name)
        self._scene.add_node(node)
        self._nodes.append(node)
        return name
        
    
    def clear(self):
        for node in self._nodes:
            self._scene.remove_node(node)
        self._nodes = []
    
    def render(self):
        return self.renderer.render(self._scene)

if __name__ == "__main__":
    renderer = OnlineObjectRenderer()
    renderer.add_mesh(
        "/path/to/mesh", 
        "o1", # name of output
        np.array([np.pi/2, -np.pi/2, 0]),
        np.array([0, .5, 0])
    )
    img, dp = renderer.render()
    plt.axis("off")
    plt.imshow(img)
    plt.show()    