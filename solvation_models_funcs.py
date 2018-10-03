from scipy.special import erf
import vampyr3d as vp
import numpy as np


min_scale = -4
max_depth = 25
order = 5
prec = 1e-4

corner = np.array([-1, -1, -1])
boxes = np.array([2, 2, 2])

world = vp.BoundingBox(min_scale, corner, boxes)

basis = vp.InterpolatingBasis(order)

MRA = vp.MultiResolutionAnalysis(world, basis, max_depth)


def Cavity(x, y, z):
    s = 0.4
    r1 = np.array([0., 0., 0.])     # position of the nucleus
    R1 = 1.00                     # *10**-10 m van der waal radius of an atom

    r = np.array([x, y, z])
    s1 = np.linalg.norm(r-r1) - R1

    return 1 - 0.5*(1 + erf(s1/s))


def diel_f_Lin(x, y, z):
    e_inf = 2.27
    e_0 = 1
    C = Cavity(x, y, z)

    return e_inf*(1 - C) + e_0*C


def diel_f_Lin_inv(x, y, z):
    e_inf = 2.27
    e_0 = 1
    C = Cavity(x, y, z)

    return 1/(e_inf*(1 - C) + e_0*C)


def diel_f_exp(x, y, z):
    e_inf = 2.27
    e_0 = 1
    C = Cavity(x, y, z)

    return e_0*np.exp(np.log10(e_inf/e_0)*(1 - C))


def diel_f_exp_inv(x, y, z):
    e_inf = 2.27
    e_0 = 1
    C = Cavity(x, y, z)

    return (e_0**(-1))*np.exp(np.log10(e_0/e_inf)*(1 - C))


def rho_eff_Lin(x, y, z):
    alpha = 2000
    A = np.sqrt((alpha**3)/(np.pi**3))
    rho = A*np.exp(-alpha*(x**2 + y**2 + z**2))
    return rho*(1)*diel_f_Lin_inv(x, y, z)


def rho_eff_exp(x, y, z):
    alpha = 2000
    A = np.sqrt((alpha**3)/(np.pi**3))
    rho = A*np.exp(-alpha*(x**2 + y**2 + z**2))
    return rho*(1)*diel_f_exp_inv(x, y, z)


def D_functree(D, in_tree, MRA):
    '''
    differentiates a 3d functions

    Args:
        D: derivative operator
        in_tree: 3d functiontree to be differentiated
        MRA: from vampyr3d module MultiResolutionAnalysis
    output:
        list of 3 FunctionTrees which contain a partial derivative of the
        in_tree. This is in the form [d/dx, d/dy, d/dz]
    '''
    out_treex = vp.FunctionTree(MRA)
    out_treey = vp.FunctionTree(MRA)
    out_treez = vp.FunctionTree(MRA)

    vp.apply(out_treex, D, in_tree, 0)   # diff. in with respect to x
    vp.apply(out_treey, D, in_tree, 1)   # diff. in with respect to y
    vp.apply(out_treez, D, in_tree, 2)   # diff. in with respect to z

    return [out_treex, out_treey, out_treez]


# add en out_tree til argumnetene
def dot_product(prec, factor_array1, factor_array2, out_tree, MRA):
    '''
    performs the dot product operation on two tree arrays of size 3.
    the operation is performed as if each of the arrays is a 3D vector and each
    of the trees are a value in that vector.
    '''
    add_tree1 = vp.FunctionTree(MRA)
    mult_tree1 = vp.FunctionTree(MRA)
    mult_tree2 = vp.FunctionTree(MRA)

    vp.multiply(prec, mult_tree1, 1, factor_array1[0], factor_array2[0])
    vp.multiply(prec, mult_tree2, 1, factor_array1[1], factor_array2[1])

    vp.add(prec/10, add_tree1, 1.0, mult_tree1, 1.0, mult_tree2)
    mult_tree1.clear()

    vp.multiply(prec, mult_tree1, 1, factor_array1[2], factor_array2[2])
    vp.add(prec/10, out_tree, 1.0, mult_tree1, 1.0, add_tree1)


def poisson_solver(V_tree, rho_tree, PoissonOperator, prec):
    P = PoissonOperator
    vp.apply(prec, V_tree, P, rho_tree)


# don't use this
def setup_initial_guess(V_tree, rho_eff_tree, func, PoissonOperator, prec):
    P = PoissonOperator
    vp.project(prec, rho_eff_tree, func)
    rho_eff_tree.normalize()
    poisson_solver(V_tree, rho_eff_tree, P, prec)


def find_err(Tree_a, Tree_b, prec):
    temp_tree = vp.FunctionTree(MRA)
    vp.add(prec/10, temp_tree, 1.0, Tree_a, -1.0, Tree_b)
    error = np.sqrt(temp_tree.getSquareNorm())
    return error


def clone_tree(in_tree, out_tree, prec):
    def ones(x, y, z):
        return 1
    temp_tree = vp.FunctionTree(MRA)
    vp.project(prec, temp_tree, ones)
    vp.multiply(prec, out_tree, 1, temp_tree, in_tree)


def gamma_exp(DC_vector, DV_vector, out_tree, prec, MRA):
    dot_product(prec, DV_vector, DC_vector, out_tree, MRA)
    out_tree.rescale(1/4*np.pi)


def gamma_Lin(eps_inv_tree, DC_vector, DV_vector, out_tree, prec, MRA):
    temp_tree = vp.FunctionTree(MRA)
    dot_product(prec, DV_vector, DC_vector, temp_tree, MRA)
    vp.multiply(prec, out_tree, 1, temp_tree, eps_inv_tree)
    out_tree.rescale(1/4*np.pi)  # probably wrong recheck


def V_solver_Lin(prec, MRA, DerivativeOperator, PoissonOperator, eps_inv_tree,
                 C_tree, V_tree, rho_eff_tree, old_V_tree):
    gamma_tree = vp.FunctionTree(MRA)
    poiss_tree = vp.FunctionTree(MRA)
    D = DerivativeOperator
    P = PoissonOperator

    DV_vector = D_functree(D, V_tree, MRA)
    clone_tree(V_tree, old_V_tree, prec)

    V_tree.clear()
    DC_vector = D_functree(D, C_tree, MRA)

    gamma_Lin(eps_inv_tree, DC_vector, DV_vector, gamma_tree, prec, MRA)

    vp.add(prec, poiss_tree, 1, rho_eff_tree, 1, gamma_tree)

    poisson_solver(V_tree, poiss_tree, P, prec)

    return gamma_tree


def V_solver_exp(rho_eff_tree, V_tree, C_tree, DerivativeOperator,
                 PoissonOperator, MRA, prec, old_V_tree):
    gamma_tree = vp.FunctionTree(MRA)
    poiss_tree = vp.FunctionTree(MRA)
    D = DerivativeOperator
    P = PoissonOperator

    DV_vector = D_functree(D, V_tree, MRA)
    clone_tree(V_tree, old_V_tree, prec)

    V_tree.clear()
    DC_vector = D_functree(D, C_tree, MRA)

    gamma_exp(DC_vector, DV_vector, gamma_tree, prec, MRA)

    vp.add(prec, poiss_tree, 1, rho_eff_tree, 1, gamma_tree)

    poisson_solver(V_tree, poiss_tree, P, prec)

    return gamma_tree


# not needed
def new_rho_eff(rho_eff_tree, V_tree, gamma_tree, DerivativeOperator, MRA):

    D = DerivativeOperator

    DV_vector = vp.gradient(D, V_tree)
    DDV_tree = vp.FunctionTree(MRA)
    vp.divergence(DDV_tree, D, DV_vector)

    rho_eff_tree.clear()
    vp.add(prec, rho_eff_tree, 1, DDV_tree, 1, gamma_tree)
    rho_eff_tree.normalize()
