from scipy import special as sp


import vampyr3d as vp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

min_scale = -4
max_depth = 25
order = 5
prec = 1e-4

e_inf = 2

corner = np.array([-1, -1, -1])
boxes = np.array([2, 2, 2])

world = vp.BoundingBox(min_scale, corner, boxes)

basis = vp.InterpolatingBasis(order)

MRA = vp.MultiResolutionAnalysis(world, basis, max_depth)


def Ci(x, y, z):
    s = 0.1
    r1 = np.array([0., 0., 0.])     # position of the nucleus
    R1 = 1.20                       # *10**-10 m van der waal radius of H

    r = np.array([x, y, z])
    s1 = np.linalg.norm(r-r1) - R1

    return 1 - 0.5*(1 + sp.erf(s1/s))


def eps_Ispheres(x, y, z):
    e_inf = 2      # e_0 = 1 so i ommit it
    C = Ci(x, y, z)
    return e_inf*(1 - C) + C


def eps_exp(x, y, z):
    e_inf = 2      # e_0 = 1 so i ommit it
    C = Ci(x, y, z)

    return np.exp((np.log10(e_inf))*(1 - C))


def eps_exp_inv(x, y, z):
    e_inf = 2
    C = Ci(x, y, z)

    return np.exp((np.log10(1/e_inf))*(1 - C))


def f(x, y, z):
    alpha = 100
    A = np.sqrt((alpha**3)/(np.pi**3))
    return A*np.exp(-alpha*(x**2 + y**2 + z**2))


def V_0(x, y, z):
    return 1


def D_functree(D, in_tree, MRA):
    out_treex = vp.FunctionTree(MRA)
    out_treey = vp.FunctionTree(MRA)
    out_treez = vp.FunctionTree(MRA)

    vp.apply(out_treex, D, in_tree, 0)   # diff. in with respect to x
    vp.apply(out_treey, D, in_tree, 1)   # diff. in with respect to y
    vp.apply(out_treez, D, in_tree, 2)   # diff. in with respect to z

    return out_treex, out_treey, out_treez


P = vp.PoissonOperator(MRA, prec)
D = vp.ABGVOperator(MRA, 0.0, 0.0)


# initializing functiontrees
C_tree = vp.FunctionTree(MRA)      # used to make Deps_tree then add1
f_tree = vp.FunctionTree(MRA)      # used to make rho_tree then add2
rho_tree = vp.FunctionTree(MRA)         # describes a point charge
# Deps_tree = vp.FunctionTree(MRA)        # this tree will have D eps_exp
eps_inv_tree = vp.FunctionTree(MRA)     # gives the reciprocal of eps
add1_tree = vp.FunctionTree(MRA)
add2_tree = vp.FunctionTree(MRA)
DV_tree = vp.FunctionTree(MRA)
V_tree = vp.FunctionTree(MRA)


# projecting and applying operators on the FunctionTrees
# vp.project(prec, eps_tree, eps_exp)
vp.project(prec, f_tree, f)
vp.project(prec, C_tree, Ci)
vp.project(prec, eps_inv_tree, eps_exp_inv)
vp.project(prec, DV_tree, DV_0)

vp.apply(prec, rho_tree, P, f_tree)

vp.multiply(prec, add1_tree, 1, eps_inv_tree, rho_tree)

C_tree.rescale(np.log10(1/e_inf))
DC_treex, DC_treey, DC_treez = D_functree(D, C_tree, MRA)
Deps_tree_vec = np.array([DC_treex, DC_treey, DC_treez])

def dot_product(prec, factor_array1, factor_array2, scalar):
    # implementer en måte å dot multiplisere disse FunctionTrees

vp.multiply(prec, add2_tree, (1/(4*np.pi)), Deps_tree, DV_tree)

f_tree.clear()

vp.add(prec, f_tree, (-4*np.pi), add1_tree, (-4*np.pi), add2_tree)

vp.apply(prec, V_tree, P, f_tree)

for i in range(10):
    DV_tree.clear()
    vp.apply(DV_tree, D, V_tree, 0)
    add2_tree.clear()
    vp.multiply(prec, add2_tree, (1/(4*np.pi)), Deps_tree, DV_tree)
    f_tree.clear()
    vp.add(prec, f_tree, (-4*np.pi), add1_tree, (-4*np.pi), add2_tree)
    V_tree.clear()
    vp.apply(prec, V_tree, P, f_tree)

# the following are just tests to check if the output is correct

# all the _plt variables below are used to check the output of the functions
# by taking 2D slices off the 4D surfaces.

# x_plt = np.linspace(-2, 2, 60)
# y_plt = np.linspace(-2, 2, 60)

# eps_plt = np.array([eps_tree.evalf(x, 0, 0) for x in x_plt])
# rho_plt = np.array([rho_tree.evalf(x, 0, 0) for x in x_plt])
# plt.plot(x_plt, eps_plt, 'r')
# plt.plot(x_plt, rho_plt, 'b')

# taking 3D slices of 4D surfaces to check the output below
# xx, yy = np.meshgrid(x_plt, y_plt)
# C_3d = np.zeros(xx.shape)
# rho_3d = np.zeros(xx.shape)

# for x in range(len(x_plt)):
#     for y in range(len(y_plt)):
#         C_3d[x][y] = C_tree.evalf(x_plt[x], y_plt[y], 0)
#         rho_3d[x][y] = rho_tree.evalf(x_plt[x], y_plt[y], 0)


# C_3d = np.array(C_3d)
# rho_3d = np.array(rho_3d)

# fig = plt.figure()
# ax = plt.axes(projection='3d')

# ax.plot_wireframe(xx, yy, C_3d, color='blue')
# ax.plot_wireframe(xx, yy, rho_3d, color='red')

# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('Z')

# plt.show()
