from fealpy.backend import backend_manager as bm
from fealpy.fem import PoissonLFEMSolver
from fealpy.mesh import TriangleMesh
from fealpy.pde.poisson_2d import CosCosData 
from fealpy.utils import timer
from fealpy import logger
from fealpy.old.solver import amg_coarsen,amg_interpolation
from fealpy.sparse import CSRTensor
logger.setLevel('INFO')

tmr = timer()
next(tmr)

pde = CosCosData()
domain = pde.domain()
n = 10
mesh = TriangleMesh.from_box(box=domain, nx=n, ny=n)
# mesh.uniform_refine(n=10)
s0 = PoissonLFEMSolver(pde, mesh, p=1, timer=tmr, logger=logger)
A = s0.A

amg_coarsen.aggregation_coarsen(A)
# print(type(A))
# isC,G = amg_coarsen.ruge_stuben_coarsen(A)
# # print(P)
# isC,G = amg_coarsen.ruge_stuben_coarsen(A)
# P,R = amg_interpolation.standard_interpolation(G,isC)
# P,R = amg_interpolation.two_points_interpolation(G,isC)
# P,R = amg_interpolation.interpolation_n(G,isC)
# print(P)

