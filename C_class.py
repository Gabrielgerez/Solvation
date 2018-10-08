from scipy.special import erf
import numpy as np


class C_class:
    def __init__(self):
        self.s = 0.2
        self.molecule = []
        # [(atom 1 information), (atom 2), ..., (atom I)]
        # [(atom position, VdW_radius), ...]
        # initialize with an atom of
        # unit radius and placed at
        # origo

    def set_unit_atom(self):
        r1 = [0, 0, 0]
        R1 = 1.00
        self.add_atom(r1, R1)

    def add_atom(self, coord, VdW_radius):
        self.molecule.append((coord, VdW_radius))

    def clear(self):
        self.molecule = []

    def __call__(self, x, y, z):
        r = np.array([x, y, z])
        C = 1
        for i in range(len(self.molecule)):
            s_n = np.linalg.norm(r - self.molecule[i][0]) - self.molecule[i][1]
            Ci = 1 - 0.5*(1 + erf(s_n/self.s))
            C = C*(1 - Ci)

        C = 1 - C
        return C
