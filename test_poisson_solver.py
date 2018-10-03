import solvation_models_funcs as sfuncs
import vampyr3d as vp
import numpy as np
from math import isclose

min_scale = -4
max_depth = 25
order = 7
prec = 1e-4

corner = np.array([-1, -1, -1])
boxes = np.array([2, 2, 2])

world = vp.BoundingBox(min_scale, corner, boxes)

basis = vp.InterpolatingBasis(order)

MRA = vp.MultiResolutionAnalysis(world, basis, max_depth)


def V(x, y, z):
    alpha = 100
    A = np.sqrt((alpha**3)/(np.pi**3))
    return np.exp(-alpha*((x**2)+(y**2)+(z**2)))


def rho(x, y, z):
    alpha = 100

    eps = sfuncs.diel_f_Sph(x, y, z)

    DV = -2*alpha*(2*alpha*((x**2) + (y**2) + (z**2)) - 3)*V(x, y, z)

    return (eps*DV)/(4*np.pi)


rho_tree = vp.FunctionTree(MRA)
V_tree = vp.FunctionTree(MRA)

vp.project(prec, rho_tree, rho)


def test_poisson_solver():
    P = vp.PoissonOperator(MRA, prec)
    sfuncs.poisson_solver(V_tree, rho_tree, P, prec)
    assert isclose(V_tree.evalf(0, 0, 0), V(0, 0, 0), abs_tol=prec*10)

def test_find_err():
    Tree_a = vp.FunctionTree(MRA)
    Tree_b = vp.FunctionTree(MRA)
    vp.project(prec, Tree_a, V)
    vp.project(prec, Tree_b, V)
    error = sfuncs.find_err(Tree_a, Tree_b, prec)
    assert isclose(error, 0.0, abs_tol=prec*10)
