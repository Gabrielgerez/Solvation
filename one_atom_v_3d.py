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
    r1 = np.array([0., 0., 0.])         # position of the nucleus
    R1 = 1.20                           # *10**-10 m van der waal radius of H

    r = np.array([x, y, z])
    s1 = np.linalg.norm(r-r1) - R1
    C_1 = 1 - 0.5*(1 + sp.erf(s1/s))

    return eps*(1 - C_1) + C_1


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

# all the []_plt variables above are used to check the output of the functions
# by taking 2D slices off the 3D shapes.

plt.plot(x_plt, C_plt, 'r')
plt.plot(x_plt, U_plt, 'b')
plt.show()
