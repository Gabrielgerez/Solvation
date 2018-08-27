from scipy import *
from scipy import special as sp


import vampyr3d as vp
import numpy as np
import matplotlib.pyplot as plt

min_scale = -4
max_depth = 25
order = 5
prec = 1e-4

corner = np.array([-1, -1, -1])
boxes = np.array([2, 2, 2])

world = vp.BoundingBox(min_scale, corner, boxes)

basis = vp.InterpolatingBasis(order)

MRA = vp.MultiResolutionAnalysis(world, basis, max_depth)


def C(x, y, z):
    eps = 80
    s = 0.1
    ri = np.array([0., 0., 0.])         # position of the nucleus
    Ri = 1.20                           # *10**-10 m van der waal radius of H

    r = np.array([x, y, z])
    si = np.linalg.norm(r-ri) - Ri
    C_i = 1 - 0.5*(1 + sp.erf(si/s))

    return eps*(1 - C_i) + C_i


def rho(x, y, z):
    alpha = 100
    A = np.sqrt((alpha**3)/(np.pi**3))
    return A*np.exp(-alpha*(x**2 + y**2 + z**2))


P = vp.PoissonOperator(MRA, prec)

C_tree = vp.FunctionTree(MRA)
rho_tree = vp.FunctionTree(MRA)
U_tree = vp.FunctionTree(MRA)

# initializing functiontrees

vp.project(prec, C_tree, C)
vp.project(prec, rho_tree, rho)
vp.apply(prec, U_tree, P, rho_tree)

# projecting and applying operators on the FunctionTrees

x_plt = np.linspace(-2, 2, 100)
C_plt = np.array([C_tree.evalf(x, 0, 0) for x in x_plt])
U_plt = np.array([U_tree.evalf(x, 0, 0) for x in x_plt])
# y_plt = np.array([C(x, 0, 0) for x in x_plt])

# all the []_plt variables above are used to check the output by taking 2D
# slices off the 3D functions

plt.plot(x_plt, C_plt, 'r')
plt.plot(x_plt, U_plt, 'b')
plt.show()
