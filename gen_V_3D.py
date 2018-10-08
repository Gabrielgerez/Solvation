from scipy.special import erf


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


class Cavity:
    def __init__(self):
        self.s = 0.2
        self.molecule = []

        # self.R1 = 1.00
        # self.r1 = [0, 0, 0]
        # self.s = 0.1
        # self.molecule = [(self.r1, self.R1)]

        # [(atom 1 information), (atom 2), ..., (atom I)]
        # [(atom position, VdW_radius), ...]
        # initialize with an atom of
        # unit radius and placed at
        # origo

    def add_atom(self, coord, VdW_radius):
        self.molecule.append((coord, VdW_radius))

    def __call__(self, x, y, z):
        r = np.array([x, y, z])
        C = 1
        for i in range(len(self.molecule)):
            s_n = np.linalg.norm(r - self.molecule[i][0]) - self.molecule[i][1]
            Ci = 1 - 0.5*(1 + erf(s_n/self.s))
            C = C*(1 - Ci)

        C = 1 - C
        return C


C_inst = Cavity()
print(C_inst(0, 0, 0))
