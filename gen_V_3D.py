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


class C_i:
    def __init__(self):
        self.r1 = np.array([0., 0., 0.])    # position of the nucleus
        self.R1 = 1.20                      # *10**-10 m van der waal r. of H

        self.eps = 80
        self.s = 0.1

        self.atoms = [[self.r1, self.R1]]
        # will most surelly change it so just eps and sigma are on __init__

    def add_atom(self, coord, VdW_radius):
        self.atoms.append([coord, VdW_radius])

    def __call__(self, x, y, z):
        r = np.array([x, y, z])

        C = 1
        for atom in self.atoms:
            s_n = np.linalg.norm(r-atom[0]) - atom[1]
            Ci = 1 - 0.5*(1 + sp.erf(s_n/self.s))
            C = C*(1 - Ci)

        C = 1 - C
        return self.eps*(1 - C) + C


C_inst = C_i()

C_tree = vp.FunctionTree(MRA)

vp.project(prec, C_tree, C_inst)
print(C_tree)


x_plt = np.linspace(-2, 2, 100)
C_plt = np.array([C_tree.evalf(x, 0, 0) for x in x_plt])

plt.plot(x_plt, C_plt)
plt.show()
