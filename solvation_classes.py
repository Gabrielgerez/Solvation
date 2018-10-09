from scipy.special import erf
import numpy as np


class C_class:
    def __init__(self):
        self.s = 0.2
        self.molecule = []
        self.r1 = [0.0, 0.0, 0.0]
        self.R1 = 1.0   # used for setting unit atom
        # molecule = [(atom 1 information), (atom 2), ..., (atom I)]
        # atom information = (atom position, VdW_radius)

    def set_unit_atom(self):
        unit_atom = (self.r1, self.R1)
        self.add_atom(unit_atom)

    def add_atom(self, atom):
        '''atom is a tuple with coordinates in index 0 and radius in index 1'''
        self.molecule.append(atom)

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
