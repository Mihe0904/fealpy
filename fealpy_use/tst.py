from fealpy.mesh import TetrahedronMesh
from fealpy.backend import backend_manager as bm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

node=bm.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, bm.sqrt(3)/2, 0.0],
                [0.5, bm.sqrt(3)/6, bm.sqrt(2/3)]], dtype=bm.float64)
cell=bm.array([[0, 1, 2, 3]])
mesh1 = TetrahedronMesh.from_box(nx=1, ny=1,nz=1)  # 未加密的网格
mesh = TetrahedronMesh.from_box(nx=2, ny=2,nz=2)    # 加密后的网格
# mesh = TetrahedronMesh(node,cell)
# mesh1 = TetrahedronMesh(node,cell)
# print(mesh.number_of_nodes())
# P = mesh.uniform_refine(n=1,returnim=True)
# print(mesh.number_of_nodes())
# print(P[0].sum())

# 创建画布并设置子图
fig = plt.figure(figsize=(12, 6))  # 创建一个画布

# 绘制未加密的网格（mesh1）
ax1 = fig.add_subplot(121, projection='3d')  # 创建第一个三维子图
mesh1.add_plot(ax1)                  # 添加网格到第一个子图
mesh1.find_node(ax1, showindex=True) # 显示节点索引
mesh1.find_edge(ax1, showindex=True)
# mesh1.find_cell(ax1, showindex=True) # 显示边索引
ax1.set_title("Initial Mesh")        # 设置标题

# 绘制加密后的网格（mesh）
ax2 = fig.add_subplot(122, projection='3d')  # 创建第二个三维子图
mesh.add_plot(ax2)                   # 添加网格到第二个子图
mesh.find_node(ax2, showindex=True)  # 显示节点索引
mesh.find_edge(ax2, showindex=True)
# mesh.find_cell(ax1, showindex=True)  # 显示边索引
ax2.set_title("Refined Mesh")        # 设置标题

# 调整布局并显示图形
plt.tight_layout()
plt.show()