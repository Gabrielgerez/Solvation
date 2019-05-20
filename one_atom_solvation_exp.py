import vampyr3d as vp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from solvation_models_funcs import V_SCF_exp, exp_eps_diff, poisson_solver, \
                                   exact_delta_G, initialize_cavity,\
                                   destroy_Cavity, change_e_inf

min_scale = -5
max_depth = 25
order = 7
prec = 1e-3
corner = [-1, -1, -1]
boxes = [2, 2, 2]
world = vp.BoundingBox(min_scale, corner, boxes)
basis = vp.InterpolatingBasis(order)
MRA = vp.MultiResolutionAnalysis(world, basis, max_depth)

P = vp.PoissonOperator(MRA, prec)
D = vp.ABGVOperator(MRA, 0.0, 0.0)

rho_tree = vp.FunctionTree(MRA)
V_tree = vp.FunctionTree(MRA)


charge = 1
Radius = [3.6]
d = 0.2
e_inf = 2.00

Heh_p = np.array([[+0.0000, 0.0000, -0.1250],
                  [-1.4375, 0.0000,  1.0250],
                  [+1.4375, 0.0000,  1.0250]])  # [H, He]
Heh_p_Z = np.array([8.0,
                    1.0,
                    1.0])


while (Radius[0] <= 6.0):
    initialize_cavity([[0.0, 0.0, 0.0]], Radius, d)
    change_e_inf(e_inf)

    gamma_tree = V_SCF_exp(MRA, prec, P, D, charge, rho_tree,
                           V_tree, Heh_p, Heh_p_Z)
    # finding E_r
    V_r_tree = vp.FunctionTree(MRA)
    eps_diff_tree = vp.FunctionTree(MRA)
    rho_diff_tree = vp.FunctionTree(MRA)
    poiss_tree = vp.FunctionTree(MRA)
    integral_tree = vp.FunctionTree(MRA)
    print('set V_r')
    vp.project(prec, eps_diff_tree, exp_eps_diff)
    vp.multiply(prec, rho_diff_tree, 1, eps_diff_tree, rho_tree)
    vp.add(prec, poiss_tree, 1, gamma_tree, -1, rho_diff_tree)
    poisson_solver(V_r_tree, poiss_tree, P, prec)
    # vp.multiply(prec, integral_tree, 1, rho_tree, V_r_tree)
    print('plotting potentials')
    x_plt = np.linspace(-7, 7, 1000)
    z_plt = np.linspace(-7, 7, 1000)
    X, Z = np.meshgrid(x_plt, z_plt)
    gamma_plt = np.zeros_like(X)
    V_r_plt = np.zeros_like(X)
    for i in range(len(x_plt)):
        for j in range(len(z_plt)):
            # gamma_plt[i][j] = gamma_tree.evalf([X[i][j], 0.0, Z[i][j]])]
            V_r_plt[i][j] = V_r_tree.evalf([X[i][j], 0.0, Z[i][j]])
    # plt.plot(x_plt, gamma_plt, 'r')
    # plt.plot(x_plt, V_r_plt, 'b')
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Z, V_r_plt)
    plt.show()
    # E_r = 0.5*integral_tree.integrate()
    # print('Reaction field energy V_r:\t\t', E_r)

    rho_tree.clear()
    V_tree.clear()
    V_r_tree.clear()
    eps_diff_tree.clear()
    rho_diff_tree.clear()
    poiss_tree.clear()

    del V_r_tree
    del eps_diff_tree
    del rho_diff_tree
    del poiss_tree

    destroy_Cavity()
    Radius[0] += 0.2
