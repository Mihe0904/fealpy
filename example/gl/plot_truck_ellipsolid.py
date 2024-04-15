
import numpy as np

from fealpy.mesh import TriangleMesh
from fealpy.plotter.gl import OpenGLPlotter, OCAMModel

cam = np.array([
    [ 8.35/2.0, -3.47/2.0, 1.515-3.0], # 右前
    [-8.25/2.0, -3.47/2.0, 1.505-3.0], # 右后
    [-17.5/2.0,       0.0, 1.295-3.0], # 后
    [-8.35/2.0,  3.47/2.0, 1.495-3.0], # 左后
    [ 8.35/2.0,  3.47/2.0, 1.495-3.0], # 左前
    [ 17.5/2.0,       0.0, 1.345-3.0]  # 前
    ], dtype=np.float64)

mesh= TriangleMesh.from_ellipsoid_surface(20, 20, 
        radius=(17.5, 3.47, 3), 
        theta=(np.pi/2, np.pi/2+np.pi/3),
        phi=(-np.pi/4, np.pi/4))

node = mesh.entity('node')
cell = mesh.entity('cell')

cmodel = OCAMModel()
node0 = node - cam[-1] # 相对于相机的坐标 
node0 /= np.linalg.norm(node0, axis=-1, keepdims=True)
uv = cmodel.sphere_to_cam(node0)
uv[:, 0] = (uv[:, 0] - np.min(uv[:, 0]))/(np.max(uv[:, 0])-np.min(uv[:, 0]))
uv[:, 1] = (uv[:, 1] - np.min(uv[:, 1]))/(np.max(uv[:, 1])-np.min(uv[:, 1]))

node = np.hstack((node, uv), dtype=np.float32)
cell = np.array(cell, dtype=np.uint32)
vertices = node[cell].reshape(-1, node.shape[1])

plotter = OpenGLPlotter()
plotter.add_mesh(vertices, cell=None, texture_path='/home/why/frame1_0.jpg')
#plotter.add_mesh(node, cell=cell, texture_path='/home/why/frame1_0.jpg')
plotter.run()
