from scipy import special as sp
from sympy import *

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

help(vp.apply)
