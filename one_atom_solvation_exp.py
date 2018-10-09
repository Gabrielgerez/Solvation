import solvation_models_funcs as sfuncs
import vampyr3d as vp
import numpy as np
import matplotlib.pyplot as plt

min_scale = -4
max_depth = 25
order = 7
prec = 1e-6

charge = 1.0
Radius = 3.0

corner = [-1, -1, -1]
boxes = [2, 2, 2]

world = vp.BoundingBox(min_scale, corner, boxes)

basis = vp.InterpolatingBasis(order)

MRA = vp.MultiResolutionAnalysis(world, basis, max_depth)

P = vp.PoissonOperator(MRA, prec)
D = vp.ABGVOperator(MRA, 0.0, 0.0)

rho_tree = vp.FunctionTree(MRA)
V_tree = vp.FunctionTree(MRA)

gamma_tree = sfuncs.V_SCF_exp(MRA, prec, P, D, charge, Radius, rho_tree,
                              V_tree)
# finding E_r
print('Finding Reaction field energy')

V_r_tree = vp.FunctionTree(MRA)
eps_diff_tree = vp.FunctionTree(MRA)
rho_diff_tree = vp.FunctionTree(MRA)
poiss_tree = vp.FunctionTree(MRA)

print('initialized FunctionTrees')

vp.project(prec, eps_diff_tree, sfuncs.exp_eps_diff)
vp.multiply(prec, rho_diff_tree, 1, eps_diff_tree, rho_tree)
vp.add(prec, poiss_tree, 1, gamma_tree, -1, rho_diff_tree)

sfuncs.poisson_solver(V_r_tree, poiss_tree, P, prec)

integral_tree = vp.FunctionTree(MRA)
vp.multiply(prec, integral_tree, 1, rho_tree, V_r_tree)

E_r = 0.5*integral_tree.integrate()
print('Reaction field energy with V_r:\t\t', E_r)


delta_G = sfuncs.exact_delta_G(charge, Radius)
print('delta G:\t\t\t\t', delta_G)

outfile = open('data.txt', 'a')
outfile.write('|q:\t%f\t|Radius:\t%f\t|E_r:\t%f\t|delta_G:\t%f\n' % (charge,
              Radius, E_r, delta_G))
outfile.close()
