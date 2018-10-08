import solvation_models_funcs as sfuncs
import vampyr3d as vp
import numpy as np
import matplotlib.pyplot as plt

min_scale = -4
max_depth = 25
order = 7
prec = 1e-6

e_inf = 78.30
e_0 = 1

corner = [-1, -1, -1]
boxes = [2, 2, 2]

world = vp.BoundingBox(min_scale, corner, boxes)

basis = vp.InterpolatingBasis(order)

MRA = vp.MultiResolutionAnalysis(world, basis, max_depth)

P = vp.PoissonOperator(MRA, prec)
D = vp.ABGVOperator(MRA, 0.0, 0.0)

# making rho as a GaussFunc
rho_tree = vp.FunctionTree(MRA)
pos = [0, 0, 0]
power = [0, 0, 0]
beta = 100
alpha = (beta/np.pi)*np.sqrt(beta / np.pi)
rho_gauss = vp.GaussFunc(beta, alpha, pos, power)
vp.build_grid(rho_tree, rho_gauss)
vp.project_gauss(prec, rho_tree, rho_gauss)
# rho_tree has a GaussFunc for rho

# exponential dielectric function
eps_inv_tree = vp.FunctionTree(MRA)
rho_eff_tree = vp.FunctionTree(MRA)
Cavity_tree = vp.FunctionTree(MRA)

V_tree = vp.FunctionTree(MRA)

# making rho_eff_tree containing rho_eff
vp.project(prec, eps_inv_tree, sfuncs.diel_f_exp_inv)
vp.multiply(prec, rho_eff_tree, 1, eps_inv_tree, rho_tree)
# this will not change

vp.project(prec, Cavity_tree, sfuncs.Cavity)
Cavity_tree.rescale(np.log(e_0/e_inf))

# start solving the poisson equation with an initial guess
sfuncs.poisson_solver(V_tree, rho_eff_tree, P, prec)

x_plt = np.linspace(-2, 2, 60)
j = 1
error = 1
old_V_tree = vp.FunctionTree(MRA)

while(error > prec):
    # solving the poisson equation once
    gamma_tree = sfuncs.V_solver_exp(rho_eff_tree, V_tree,
                                     Cavity_tree, D, P, MRA, prec,
                                     old_V_tree)

    # finding error once
    temp_tree = vp.FunctionTree(MRA)
    vp.add(prec/10, temp_tree, 1.0, V_tree, -1.0, old_V_tree)
    error = np.sqrt(temp_tree.getSquareNorm())

    a = gamma_tree.integrate()

    print('iterations %i error %f energy %f' % (j, error, a))

    print('exact energy %f' % ((1 - e_inf)/e_inf))
    if(j % 5 == 0 or j == 1):
        V_exp_plt = np.array([V_tree.evalf(x, 0, 0) for x in x_plt])
        eps_inv_plt = np.array([sfuncs.diel_f_exp_inv(x, 0, 0) for x in x_plt])
        plt.figure()
        plt.plot(x_plt, V_exp_plt, 'b')
        plt.plot(x_plt, 1/x_plt, 'g')
        plt.plot(x_plt, eps_inv_plt, 'y')
        plt.title('iterations %i' % (j))
        plt.show()

    elif(error <= prec):
        print('converged')
        V_exp_plt = np.array([V_tree.evalf(x, 0, 0) for x in x_plt])
        eps_inv_plt = np.array([sfuncs.diel_f_exp_inv(x, 0, 0) for x in x_plt])
        plt.figure()
        plt.plot(x_plt, V_exp_plt, 'b')
        plt.plot(x_plt, 1/x_plt, 'g')
        plt.plot(x_plt, eps_inv_plt, 'y')
        plt.title('iterations %i' % (j))
        plt.show()
    j += 1
