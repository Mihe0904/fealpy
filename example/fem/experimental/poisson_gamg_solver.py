#!/usr/bin/python3
# import ipdb
from fealpy.decorator import barycentric
from fealpy.utils import timer
from fealpy import logger
logger.setLevel('INFO')
from fealpy.backend import backend_manager as bm
from fealpy.pde.poisson_2d import CosCosData 
from fealpy.pde.poisson_3d import CosCosCosData
from fealpy.mesh import TriangleMesh,TetrahedronMesh
from fealpy.fem import PoissonLFEMSolver
import time
# bm.set_default_device(cuda)

tmr = timer()
next(tmr)

p = 1
n = 2
m = 6
pde = CosCosCosData()
# pde = CosCosData()


domain = pde.domain()
# mesh = TriangleMesh.from_box(box=domain,nx=n,ny=n)
mesh = TetrahedronMesh.from_box(box=domain,nx=n,ny=n,nz=n)
IM = mesh.uniform_refine(n=m,returnim=True)

s0 = PoissonLFEMSolver(pde, mesh, p, timer=tmr, logger=logger)

s0.gamg_solve(IM)
s0.cg_solve()
tmr.send(None)

