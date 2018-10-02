import solvation_models_funcs as sfuncs
from scipy import special as spec


import vampyr3d as vp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

min_scale = -4
max_depth = 25
order = 7
prec = 1e-6

e_inf = 2.27
e_0 = 1

corner = np.array([-1, -1, -1])
boxes = np.array([2, 2, 2])

world = vp.BoundingBox(min_scale, corner, boxes)

basis = vp.InterpolatingBasis(order)

MRA = vp.MultiResolutionAnalysis(world, basis, max_depth)

P = vp.PoissonOperator(MRA, prec)
D = vp.ABGVOperator(MRA, 0.0, 0.0)


# initializing functiontrees
rho_eff_Sph_tree = vp.FunctionTree(MRA)
rho_eff_exp_tree = vp.FunctionTree(MRA)

Cavity_exp_tree = vp.FunctionTree(MRA)
# Cavity_Sph_tre = vp.FunctionTree(MRA)
eps_inv_Sph_tree = vp.FunctionTree(MRA)

V_Sph_tree = vp.FunctionTree(MRA)
V_exp_tree = vp.FunctionTree(MRA)

# sfuncs.setup_initial_guess(V_Sph_tree, rho_eff_Sph_tree,
#                            sfuncs.rho_eff_Sph, P, prec)
sfuncs.setup_initial_guess(V_exp_tree, rho_eff_exp_tree,
                           sfuncs.rho_eff_exp, P, prec)

vp.project(prec, Cavity_exp_tree, sfuncs.Cavity)
Cavity_exp_tree.rescale(np.log10(e_0/e_inf))


x_plt = np.linspace(-2, 2, 60)
j = 1
error = 1
old_V_tree = vp.FunctionTree(MRA)

while(error > prec):

    gamma_tree = sfuncs.V_solver_exp(rho_eff_exp_tree, V_exp_tree,
                                     Cavity_exp_tree, D, P, MRA, prec,
                                     old_V_tree)

    temp_tree = vp.FunctionTree(MRA)
    vp.add(prec/10, temp_tree, 1.0, V_exp_tree, -1.0, old_V_tree)
    error = np.sqrt(temp_tree.getSquareNorm())

    a = gamma_tree.integrate()
    print('iterations %i error %f energy %f' % (j, error, a))
    print('exact energy %f' % ((1 - e_inf)/e_inf))
    if(j % 5 == 0):
        V_exp_plt = np.array([V_exp_tree.evalf(x, 0, 0) for x in x_plt])
        rho_array = np.array(rho_eff_exp_tree)
        plt.figure()
        plt.plot(x_plt, V_exp_plt, 'b')
        plt.plot(x_plt, 1/x_plt, '1')
        plt.title('iterations %i' % (j))
        plt.show()
    j += 1

# the following are just tests to check if the output is correct

# all the _plt variables below are used to check the output of the functions
# by taking 2D slices off the 4D surfaces.

# y_plt = np.linspace(-2, 2, 60)

# eps_plt = np.array([eps_tree.evalf(x, 0, 0) for x in x_plt])
# rho_plt = np.array([rho_tree.evalf(x, 0, 0) for x in x_plt])
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
