import numpy as np
from fealpy.geometry.implicit_surface import SphereSurface
from fealpy.mesh.triangle_mesh import TriangleMesh

# 定义多个典型的 LagrangeTriangleMesh 对象
surface = SphereSurface() #以原点为球心，1 为半径的球
mesh = TriangleMesh.from_unit_sphere_surface()
node = mesh.interpolation_points(3)
cell = mesh.cell_to_ipoint(3)

init_data = [
        {
            "p": 3,
            "node": np.array([[ 0.        ,  0.85065081,  0.52573111],
               [ 0.        ,  0.85065081, -0.52573111],
               [ 0.85065081,  0.52573111,  0.        ],
               [ 0.85065081, -0.52573111,  0.        ],
               [ 0.        , -0.85065081, -0.52573111],
               [ 0.        , -0.85065081,  0.52573111],
               [ 0.52573111,  0.        ,  0.85065081],
               [-0.52573111,  0.        ,  0.85065081],
               [ 0.52573111,  0.        , -0.85065081],
               [-0.52573111,  0.        , -0.85065081],
               [-0.85065081,  0.52573111,  0.        ],
               [-0.85065081, -0.52573111,  0.        ],
               [ 0.        ,  0.85065081, -0.1752437 ],
               [ 0.        ,  0.85065081,  0.1752437 ],
               [ 0.56710054,  0.63403768,  0.1752437 ],
               [ 0.28355027,  0.74234424,  0.35048741],
               [ 0.1752437 ,  0.56710054,  0.63403768],
               [ 0.35048741,  0.28355027,  0.74234424],
               [-0.1752437 ,  0.56710054,  0.63403768],
               [-0.35048741,  0.28355027,  0.74234424],
               [-0.56710054,  0.63403768,  0.1752437 ],
               [-0.28355027,  0.74234424,  0.35048741],
               [ 0.28355027,  0.74234424, -0.35048741],
               [ 0.56710054,  0.63403768, -0.1752437 ],
               [ 0.35048741,  0.28355027, -0.74234424],
               [ 0.1752437 ,  0.56710054, -0.63403768],
               [-0.1752437 ,  0.56710054, -0.63403768],
               [-0.35048741,  0.28355027, -0.74234424],
               [-0.28355027,  0.74234424, -0.35048741],
               [-0.56710054,  0.63403768, -0.1752437 ],
               [ 0.85065081, -0.1752437 ,  0.        ],
               [ 0.85065081,  0.1752437 ,  0.        ],
               [ 0.63403768,  0.1752437 ,  0.56710054],
               [ 0.74234424,  0.35048741,  0.28355027],
               [ 0.63403768,  0.1752437 , -0.56710054],
               [ 0.74234424,  0.35048741, -0.28355027],
               [ 0.56710054, -0.63403768, -0.1752437 ],
               [ 0.28355027, -0.74234424, -0.35048741],
               [ 0.28355027, -0.74234424,  0.35048741],
               [ 0.56710054, -0.63403768,  0.1752437 ],
               [ 0.63403768, -0.1752437 ,  0.56710054],
               [ 0.74234424, -0.35048741,  0.28355027],
               [ 0.74234424, -0.35048741, -0.28355027],
               [ 0.63403768, -0.1752437 , -0.56710054],
               [ 0.        , -0.85065081,  0.1752437 ],
               [ 0.        , -0.85065081, -0.1752437 ],
               [ 0.1752437 , -0.56710054, -0.63403768],
               [ 0.35048741, -0.28355027, -0.74234424],
               [-0.35048741, -0.28355027, -0.74234424],
               [-0.1752437 , -0.56710054, -0.63403768],
               [-0.56710054, -0.63403768, -0.1752437 ],
               [-0.28355027, -0.74234424, -0.35048741],
               [ 0.35048741, -0.28355027,  0.74234424],
               [ 0.1752437 , -0.56710054,  0.63403768],
               [-0.35048741, -0.28355027,  0.74234424],
               [-0.1752437 , -0.56710054,  0.63403768],
               [-0.28355027, -0.74234424,  0.35048741],
               [-0.56710054, -0.63403768,  0.1752437 ],
               [ 0.1752437 ,  0.        ,  0.85065081],
               [-0.1752437 ,  0.        ,  0.85065081],
               [-0.63403768,  0.1752437 ,  0.56710054],
               [-0.74234424,  0.35048741,  0.28355027],
               [-0.74234424, -0.35048741,  0.28355027],
               [-0.63403768, -0.1752437 ,  0.56710054],
               [ 0.1752437 ,  0.        , -0.85065081],
               [-0.1752437 ,  0.        , -0.85065081],
               [-0.63403768,  0.1752437 , -0.56710054],
               [-0.74234424,  0.35048741, -0.28355027],
               [-0.74234424, -0.35048741, -0.28355027],
               [-0.63403768, -0.1752437 , -0.56710054],
               [-0.85065081,  0.1752437 ,  0.        ],
               [-0.85065081, -0.1752437 ,  0.        ],
               [ 0.45879397,  0.45879397,  0.45879397],
               [ 0.74234424,  0.        ,  0.28355027],
               [ 0.45879397, -0.45879397,  0.45879397],
               [ 0.        , -0.28355027,  0.74234424],
               [ 0.        ,  0.28355027,  0.74234424],
               [ 0.74234424,  0.        , -0.28355027],
               [ 0.45879397,  0.45879397, -0.45879397],
               [ 0.28355027,  0.74234424,  0.        ],
               [-0.28355027,  0.74234424,  0.        ],
               [-0.45879397,  0.45879397, -0.45879397],
               [ 0.        ,  0.28355027, -0.74234424],
               [ 0.45879397, -0.45879397, -0.45879397],
               [ 0.28355027, -0.74234424,  0.        ],
               [-0.28355027, -0.74234424,  0.        ],
               [-0.74234424,  0.        ,  0.28355027],
               [-0.45879397,  0.45879397,  0.45879397],
               [-0.45879397, -0.45879397, -0.45879397],
               [ 0.        , -0.28355027, -0.74234424],
               [-0.45879397, -0.45879397,  0.45879397],
               [-0.74234424,  0.        , -0.28355027]], dtype=np.float64),
            "cell": np.array([[ 6, 32, 17, 33, 72, 16,  2, 14, 15,  0],
               [ 3, 30, 41, 31, 73, 40,  2, 33, 32,  6],
               [ 5, 38, 53, 39, 74, 52,  3, 41, 40,  6],
               [ 5, 53, 55, 52, 75, 54,  6, 58, 59,  7],
               [ 6, 17, 58, 16, 76, 59,  0, 18, 19,  7],
               [ 3, 42, 30, 43, 77, 31,  8, 34, 35,  2],
               [ 2, 35, 23, 34, 78, 22,  8, 24, 25,  1],
               [ 2, 23, 14, 22, 79, 15,  1, 12, 13,  0],
               [ 0, 13, 21, 12, 80, 20,  1, 28, 29, 10],
               [ 1, 26, 28, 27, 81, 29,  9, 66, 67, 10],
               [ 8, 64, 24, 65, 82, 25,  9, 27, 26,  1],
               [ 4, 46, 37, 47, 83, 36,  8, 43, 42,  3],
               [ 4, 37, 45, 36, 84, 44,  3, 39, 38,  5],
               [ 4, 45, 51, 44, 85, 50,  5, 56, 57, 11],
               [ 7, 60, 63, 61, 86, 62, 10, 70, 71, 11],
               [ 0, 21, 18, 20, 87, 19, 10, 61, 60,  7],
               [ 4, 51, 49, 50, 88, 48, 11, 68, 69,  9],
               [ 8, 47, 64, 46, 89, 65,  4, 49, 48,  9],
               [ 5, 55, 56, 54, 90, 57,  7, 63, 62, 11],
               [10, 67, 70, 66, 91, 71,  9, 69, 68, 11]],dtype=np.int32),
            "surface": surface,
            "NN": 92,
            "NE": 30,
            "NF": 30,
            "NC": 20
            }
]

from_triangle_mesh_data = [
        {
            "p": 3,
            "surface": surface,
            "cell": np.array([[ 6, 32, 17, 33, 72, 16,  2, 14, 15,  0],
               [ 3, 30, 41, 31, 73, 40,  2, 33, 32,  6],
               [ 5, 38, 53, 39, 74, 52,  3, 41, 40,  6],
               [ 5, 53, 55, 52, 75, 54,  6, 58, 59,  7],
               [ 6, 17, 58, 16, 76, 59,  0, 18, 19,  7],
               [ 3, 42, 30, 43, 77, 31,  8, 34, 35,  2],
               [ 2, 35, 23, 34, 78, 22,  8, 24, 25,  1],
               [ 2, 23, 14, 22, 79, 15,  1, 12, 13,  0],
               [ 0, 13, 21, 12, 80, 20,  1, 28, 29, 10],
               [ 1, 26, 28, 27, 81, 29,  9, 66, 67, 10],
               [ 8, 64, 24, 65, 82, 25,  9, 27, 26,  1],
               [ 4, 46, 37, 47, 83, 36,  8, 43, 42,  3],
               [ 4, 37, 45, 36, 84, 44,  3, 39, 38,  5],
               [ 4, 45, 51, 44, 85, 50,  5, 56, 57, 11],
               [ 7, 60, 63, 61, 86, 62, 10, 70, 71, 11],
               [ 0, 21, 18, 20, 87, 19, 10, 61, 60,  7],
               [ 4, 51, 49, 50, 88, 48, 11, 68, 69,  9],
               [ 8, 47, 64, 46, 89, 65,  4, 49, 48,  9],
               [ 5, 55, 56, 54, 90, 57,  7, 63, 62, 11],
               [10, 67, 70, 66, 91, 71,  9, 69, 68, 11]],dtype=np.int32),
            "NN": 92,
            "NE": 30,
            "NF": 30,
            "NC": 20
                }
]

cell_area_data = [
        {
            "sphere_cm": 4*np.pi*(1**2)
            }
]

edge_length_data = [
        {
            "el": np.array([1.1097288, 1.1097288, 1.1097288, 1.1097288, 1.1097288, 1.1097288,
               1.1097288, 1.1097288, 1.1097288, 1.1097288, 1.1097288, 1.1097288,
               1.1097288, 1.1097288, 1.1097288, 1.1097288, 1.1097288, 1.1097288,
               1.1097288, 1.1097288, 1.1097288, 1.1097288, 1.1097288, 1.1097288,
               1.1097288, 1.1097288, 1.1097288, 1.1097288, 1.1097288, 1.1097288], dtype=np.float64)
       }
]

# 网格阶数 p=2
uI_error_data = [
        {"uI_error_ratio": np.array([8.0, 8.0, 8.0, 8.0], dtype=np.float64)}
        ]
