from scipy import *
from scipy import special as sp
import math as m

import vampyr3d as vp
import numpy as np
import matplotlib.pyplot as plt

min_scale = -4
max_depth = 25
order = 5
prec = 1e-3

corner = np.array([-1, -1, -1])
boxes = np.array([2, 2, 2])

world = vp.BoundingBox(min_scale, corner, boxes)

basis = vp.InterpolatingBasis(order)

MRA = vp.MultiResolutionAnalysis(world, basis, max_depth)


def C(x, y, z):
    eps = 80
    s = 0.0001
    ri = np.array([0., 0., 0.])         # position of the nucleus
    Ri = 1.20                           # *10**-10 m van der waal radius of H

    r = np.array([x, y, z])
    si = np.linalg.norm(r-ri) - Ri

    return 1 - 0.5*(1 + sp.erf(si/s)) 



C_tree = vp.FunctionTree(MRA)
vp.project(prec, C_tree, C)


x_plt = np.linspace(-2, 2, 60)
y_plt = np.array([C_tree.evalf(i, 0, 0) for i in x_plt])

plt.plot(x_plt, y_plt)
plt.show()
