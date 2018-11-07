import sys

import numpy as np

import matplotlib.pyplot as plt

from fealpy.mesh.TriangleMesh import TriangleMesh, TriangleMeshDataStructure

node = np.array([
    (0, 0), 
    (1, 0), 
    (1, 1),
    (0, 1)], dtype=np.float)
cell = np.array([
    (1, 2, 0),
    (3, 0, 2)], dtype=np.int)

tmesh = TriangleMesh(node, cell)
tmesh.uniform_refine(1)
node = tmesh.node
cell = tmesh.entity('cell')
print(tmesh.ds.node_to_cell())
N = node.shape[0]
tmeshstrcture = TriangleMeshDataStructure(N, cell)
fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes)
tmesh.find_node(axes, showindex=True)
tmesh.find_edge(axes, showindex=True)
tmesh.find_cell(axes, showindex=True)
plt.show()
