from scipy import special
import vampyr3d as vp
import numpy as np
from solvation_classes import C_class

min_scale = -4
max_depth = 25
order = 5
prec = 1e-4

corner = [-1, -1, -1]
boxes = [2, 2, 2]

world = vp.BoundingBox(min_scale, corner, boxes)

basis = vp.InterpolatingBasis(order)

MRA = vp.MultiResolutionAnalysis(world, basis, max_depth)

e_0 = 1
e_inf = 2

Cavity = C_class()

Cavity.set_unit_atom()


def input_new_molecule(molecule):
    '''molecule is an array of tuples. each tuple represents an atom with
    atom position in index 0 and atom radius in index 1'''
    global Cavity
    Cavity.clear()
    for atom in molecule:
        Cavity.add_atom(atom)


def append_atom(atom):
    '''atom is the same as atom in C_class'''
    global Cavity
    Cavity.add_atom(atom)


def Change_unit_radius(new_radius):
    global Cavity
    Cavity.R1 = new_radius
    Cavity.set_unit_atom()


def change_unit_pos(new_pos):
    global Cavity
    Cavity.r1 = new_pos
    Cavity.set_unit_atom()


def diel_f_Lin(x, y, z):
    global e_inf
    global e_0
    global Cavity
    C = Cavity(x, y, z)

    return e_inf*(1 - C) + e_0*C


def diel_f_Lin_inv(x, y, z):
    global e_inf
    global e_0
    global Cavity
    C = Cavity(x, y, z)

    return 1/(e_inf*(1 - C) + e_0*C)


def diel_f_exp(x, y, z):
    global e_inf
    global e_0
    global Cavity
    C = Cavity(x, y, z)

    return e_0*np.exp(np.log(e_inf/e_0)*(1 - C))


def diel_f_exp_inv(x, y, z):
    global e_inf
    global e_0
    global Cavity
    C = Cavity(x, y, z)

    return (e_0**(-1))*np.exp(np.log(e_0/e_inf)*(1 - C))


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
    differentiates a 3d FunctionTree

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
    mult_tree3 = vp.FunctionTree(MRA)

    vp.multiply(prec, mult_tree1, 1, factor_array1[0], factor_array2[0])
    vp.multiply(prec, mult_tree2, 1, factor_array1[1], factor_array2[1])
    vp.multiply(prec, mult_tree3, 1, factor_array1[2], factor_array2[2])

    vp.add(prec/10, add_tree1, 1.0, mult_tree1, 1.0, mult_tree2)
    vp.add(prec/10, out_tree, 1.0, mult_tree3, 1.0, add_tree1)


def poisson_solver(out_tree, in_tree, PoissonOperator, prec):
    P = PoissonOperator
    vp.apply(prec, out_tree, P, in_tree)


# this doesn't work when called
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
    out_tree.rescale((1/(4*np.pi)))


def gamma_Lin(eps_inv_tree, DC_vector, DV_vector, out_tree, prec, MRA):
    temp_tree = vp.FunctionTree(MRA)
    dot_product(prec, DV_vector, DC_vector, temp_tree, MRA)
    vp.multiply(prec, out_tree, 1, temp_tree, eps_inv_tree)
    out_tree.rescale((1/(4*np.pi)))  # probably wrong recheck


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


def exp_eps_diff(x, y, z):
    return (diel_f_exp(x, y, z) - 1)*diel_f_exp_inv(x, y, z)


def Lin_eps_diff(x, y, z):
    return (diel_f_Lin(x, y, z) - 1)*diel_f_Lin_inv(x, y, z)


def make_gauss_tree(beta, alpha, pos, power, gauss_tree, prec):
    rho_gauss = vp.GaussFunc(beta, alpha, pos, power)
    vp.build_grid(gauss_tree, rho_gauss)
    vp.project_gauss(prec, gauss_tree, rho_gauss)


def V_SCF_exp(MRA, prec, P, D, charge, Radius, rho_tree, V_tree):
    global e_inf
    global e_0
    global Cavity

    Change_unit_radius(Radius)

    print('making rho')

    # making rho as a GaussFunc
    pos = [0, 0, 0]
    power = [0, 0, 0]
    beta = 100
    alpha = (beta/np.pi)*np.sqrt(beta / np.pi)*charge
    make_gauss_tree(beta, alpha, pos, power, rho_tree, prec)
    # rho_tree has a GaussFunc for rho

    print('initializing FunctionTrees')

    # initializing FunctionTrees
    eps_inv_tree = vp.FunctionTree(MRA)
    rho_eff_tree = vp.FunctionTree(MRA)
    Cavity_tree = vp.FunctionTree(MRA)

    print('projecting all the FunctionTrees')

    # making rho_eff_tree containing rho_eff
    vp.project(prec, eps_inv_tree, diel_f_exp_inv)
    vp.multiply(prec, rho_eff_tree, 1, eps_inv_tree, rho_tree)
    # this will not change

    print('projected rho_eff')

    vp.project(prec, Cavity_tree, Cavity)
    Cavity_tree.rescale(np.log(e_0/e_inf))

    print('prepared Cavity_tree')

    # start solving the poisson equation with an initial guess
    poisson_solver(V_tree, rho_eff_tree, P, prec)

    print('made the first guess')

    j = 1
    error = 1
    old_V_tree = vp.FunctionTree(MRA)

    print('starting loop')

    while(error > prec):
        # solving the poisson equation once
        gamma_tree = V_solver_exp(rho_eff_tree, V_tree,
                                  Cavity_tree, D, P, MRA, prec,
                                  old_V_tree)

        # finding error once
        temp_tree = vp.FunctionTree(MRA)
        vp.add(prec/10, temp_tree, 1.0, V_tree, -1.0, old_V_tree)
        error = np.sqrt(temp_tree.getSquareNorm())

        a = gamma_tree.integrate()

        print('iter:\t\t\t%i\nerror:\t\t\t%f\nR charge:\t\t%f' % (j, error, a))

        print('exact Reaction charge:\t%f\n\n' % ((1 - e_inf)/e_inf))
        if(error <= prec):
            print('converged total electrostatic potential\n')
        j += 1

    return gamma_tree


def exact_delta_G(charge, Radius):
    global e_inf
    return ((1 - e_inf)*(charge**2))/(2*e_inf*Radius)
