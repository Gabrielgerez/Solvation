from scipy.special import erf
import numpy as np
import matplotlib.pyplot as plt


class Cavity_func(object):
    def __init__(self, coord, rad, d):
        '''
        coord = [[pos_sphere1],[pos_sphere2], ..., [pos_spheren]]
        rad   = [rad1, rad2, ..., radn]
        '''
        self.d = d
        self.Pos = coord
        self.Rad = rad

    def clear(self):
        del self.Pos
        del self.d
        del self.Rad

    def __call__(self, r_point):
        r = np.array(r_point)
        C = 1
        for i in range(len(self.Pos)):
            s_n = np.linalg.norm(r - self.Pos[i]) - self.Rad[i]
            Ci = 1 - 0.5*(1 + erf(s_n/self.d))
            C = C*(1 - Ci)

        C = 1 - C
        return C
