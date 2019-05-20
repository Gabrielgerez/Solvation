from scipy import special
import numpy as np
import vampyr3d as vp
from solvation_classes import Cavity_func
import matplotlib.pyplot as plt

e_0 = 1
e_inf = 2


def initialize_cavity(pos, rad, d):
    global Cavity
    Cavity = Cavity_func(pos, rad, d)


def destroy_Cavity():
    global Cavity
    Cavity.clear()
    del Cavity


def change_e_inf(new_val):
    global e_inf
    e_inf = new_val


def diel_f_Lin(r):
    global e_inf
    global e_0
    global Cavity
    C = Cavity(r)

    return e_inf*(1 - C) + e_0*C


def diel_f_Lin_inv(r):
    global e_inf
    global e_0
    global Cavity
    C = Cavity(r)

    return 1/(e_inf*(1 - C) + e_0*C)


def diel_f_exp(r):
    global e_inf
    global e_0
    global Cavity
    C = Cavity(r)

    return e_0*np.exp(np.log(e_inf/e_0)*(1 - C))


def diel_f_exp_inv(r):
    global e_inf
    global e_0
    global Cavity
    C = Cavity(r)

    return (e_0**(-1))*np.exp(np.log(e_0/e_inf)*(1 - C))


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
    mult_tree1.clear()
    mult_tree2.clear()
    mult_tree3.clear()
    add_tree1.clear()
    del mult_tree1
    del mult_tree2
    del mult_tree3
    del add_tree1


def poisson_solver(out_tree, in_tree, PoissonOperator, prec):
    P = PoissonOperator
    vp.apply(prec, out_tree, P, in_tree)


def clone_tree(in_tree, out_tree, prec, MRA):
    def ones(r):
        return 1
    temp_tree = vp.FunctionTree(MRA)
    vp.project(prec, temp_tree, ones)
    vp.multiply(prec, out_tree, 1, temp_tree, in_tree)
    temp_tree.clear()
    del temp_tree


def gamma_exp(DC_vector, DV_vector, out_tree, prec, MRA):
    dot_product(prec, DV_vector, DC_vector, out_tree, MRA)
    del DV_vector
    del DC_vector
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
    clone_tree(V_tree, old_V_tree, prec, MRA)

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
    old_V_tree.clear()
    clone_tree(V_tree, old_V_tree, prec, MRA)

    V_tree.clear()
    DC_vector = D_functree(D, C_tree, MRA)

    gamma_exp(DC_vector, DV_vector, gamma_tree, prec, MRA)

    vp.add(prec, poiss_tree, 1, rho_eff_tree, 1, gamma_tree)

    poisson_solver(V_tree, poiss_tree, P, prec)

    return gamma_tree


def exp_eps_diff(r):
    return (diel_f_exp(r) - 1)*diel_f_exp_inv(r)


def Lin_eps_diff(r):
    return (diel_f_Lin(r) - 1)*diel_f_Lin_inv(r)


def set_rho(pos, Z, charge, width, rho_tree, prec):
    # making rho as a GaussFunc
    power = [0, 0, 0]
    norm = (width/np.pi)*np.sqrt(width / np.pi)
    s = 0
    z = 0
    rho_tot = vp.GaussExp()
    biggest = [0, -1]
    for i in range(len(pos)):
        s += Z[i]*pos[i]
        z += Z[i]
        new_rho = vp.GaussFunc(width, norm * float(Z[i]), list(pos[i]), power)
        rho_tot.append(new_rho)
        if(Z[i] > biggest[0]):
            biggest = [Z[i], i]

    pos_rhoel = s/z
    width_rho_el = np.log(10)/(np.linalg.norm(pos_rhoel - pos[biggest[1]])**2)
    norm_rho_el = (width_rho_el/np.pi)*np.sqrt(width_rho_el/np.pi)*(charge - z)
    rho_el = vp.GaussFunc(width_rho_el, norm_rho_el, list(pos_rhoel), power)
    rho_tot.append(rho_el)

    vp.build_grid(rho_tree, rho_tot)
    vp.project(prec, rho_tree, rho_tot)


def grad_G(gamma_tree, cavity_tree, rho_tree, grad_G_func, MRA, prec):
    temp_tree = FunctionTree(MRA)

    # vp.multiply(prec, temp_tree, 1, , eps_inv_tree)


def V_SCF_exp(MRA, prec, P, D, charge, rho_tree, V_tree, pos, Z):
    global e_inf
    global e_0
    global Cavity

    # initializing FunctionTrees
    eps_inv_tree = vp.FunctionTree(MRA)
    rho_eff_tree = vp.FunctionTree(MRA)
    Cavity_tree = vp.FunctionTree(MRA)
    old_V_tree = vp.FunctionTree(MRA)
    print('set rho')
    set_rho(pos, Z, charge, 1000.0, rho_tree, prec)
    # making rho_eff_tree containing rho_eff
    print('set cavity functions')
    vp.project(prec/100, eps_inv_tree, diel_f_exp_inv)
    vp.project(prec/100, Cavity_tree, Cavity)

    vp.multiply(prec, rho_eff_tree, 1, eps_inv_tree, rho_tree)
    print("plotting the cavity")
    x_plt = np.linspace(-7.0, 7.0, 1000)
    Cavity_plt = np.array([Cavity_tree.evalf([x, 0., 0.]) for x in x_plt])
    plt.plot(x_plt, Cavity_plt)
    plt.show()

    Cavity_tree.rescale(np.log(e_0/e_inf))

    j = 1
    error = 1
    poisson_solver(V_tree, rho_eff_tree, P, prec)

    while(error >= prec):
        # solving the poisson equation once
        gamma_tree = V_solver_exp(rho_eff_tree, V_tree,
                                  Cavity_tree, D, P, MRA, prec,
                                  old_V_tree)

        # finding error once
        temp_tree = vp.FunctionTree(MRA)
        vp.add(prec/10, temp_tree, 1.0, V_tree, -1.0, old_V_tree)
        error = np.sqrt(temp_tree.getSquareNorm())
        temp_tree.clear()
        del temp_tree

        # Reaction_charge = gamma_tree.integrate()
        # print('iter:\t\t\t%i\nerror:\t\t\t%f\nR charge:\t\t%f' % (j, error,
        #      Reaction_charge))
        print('iter:\t', j, '\t error:\t', error)
        # print('exact Reaction charge:\t%f\n'%((charge)*((1 - e_inf)/e_inf)))
        j += 1

    print('converged total electrostatic potential\n')

    eps_inv_tree.clear()
    rho_eff_tree.clear()
    Cavity_tree.clear()
    old_V_tree.clear()
    del eps_inv_tree
    del rho_eff_tree
    del Cavity_tree
    del old_V_tree

    return gamma_tree


def exact_delta_G(charge, Radius):
    global e_inf
    return ((1 - e_inf)*(charge**2))/(2*e_inf*Radius)
