import cvxpy as cvx
import numpy as np
from cvxpy.error import SolverError

def HingePlace_p1(Jdes, Isafety, Jtol, Itotal, Af=None, Ac=None):
    I = cvx.Variable(Af.shape[1])
    constraints = [Af@I== Jdes, cvx.sum(I) == 0, cvx.atoms.norm_inf(I) <= Isafety, cvx.atoms.norm1(I) <= Itotal]

    J = Ac@I
    term1 = cvx.sum(cvx.atoms.maximum(0,J-Jtol*np.ones((Ac.shape[0]))))
    term2 = cvx.sum(cvx.atoms.maximum(0,-J-Jtol*np.ones((Ac.shape[0]))))

    obj = cvx.Minimize(term1+term2)
    prob = cvx.Problem(obj, constraints)
    try:
        prob.solve(verbose=True, max_iters=300, solver=cvx.ECOS)#solver=cvx.MOSEK #max_iters  # Returns the optimal value.
    except SolverError:
        ## decreasing reltol and abstol. and increasing max_iters
        prob.solve(verbose=True, max_iters=500, solver=cvx.ECOS, abstol=1e-05, reltol=1e-05)
    print("status:", prob.status)
    print("optimal value", prob.value)
    if prob.status == 'infeasible':
        return None
    else:
        return I.value

def HingePlace_p2(Jdes, Isafety, Jtol, Itotal, Af=None, Ac=None):
    I = cvx.Variable(Af.shape[1])
    constraints = [Af@I== Jdes, cvx.sum(I) == 0, cvx.atoms.norm_inf(I) <= Isafety, cvx.atoms.norm1(I) <= Itotal]
    J = Ac@I
    term1 = cvx.atoms.sum(cvx.atoms.square(cvx.atoms.maximum(0, J-Jtol*np.ones((Ac.shape[0])))))
    term2 = cvx.atoms.sum(cvx.atoms.square(cvx.atoms.maximum(0,-J-Jtol*np.ones((Ac.shape[0])))))

    obj = cvx.Minimize(term1+term2)
    prob = cvx.Problem(obj, constraints)
    try:
        prob.solve(verbose=True, solver=cvx.ECOS, max_iters=300)#solver=cvx.MOSEK #max_iters  # Returns the optimal value.
    except SolverError:
        prob.solve(verbose=True, solver=cvx.ECOS, max_iters=1000, abstol=1e-05, reltol=1e-05)
    print("status:", prob.status)
    print("optimal value", prob.value)
    if prob.status == 'infeasible':
        return None
    else:
        return I.value

def HingePlace_p3(Jdes, Isafety, Jtol, Itotal, Af=None, Ac=None):
    I = cvx.Variable(Af.shape[1])
    constraints = [Af@I== Jdes, cvx.sum(I) == 0, cvx.atoms.norm_inf(I) <= Isafety, cvx.atoms.norm1(I) <= Itotal]
    J = Ac@I
    term1 = cvx.atoms.sum(cvx.atoms.power(cvx.atoms.maximum(0,J-Jtol * np.ones((Ac.shape[0]))), 3))
    term2 = cvx.atoms.sum(cvx.atoms.power(cvx.atoms.maximum(0,-J-Jtol*np.ones((Ac.shape[0]))), 3))

    obj = cvx.Minimize(term1+term2)
    prob = cvx.Problem(obj, constraints)
    try:
        prob.solve(verbose=True, solver=cvx.ECOS, max_iters=300)#solver=cvx.MOSEK #max_iters  # Returns the optimal value.
    except SolverError:
        prob.solve(verbose=True, solver=cvx.ECOS, max_iters=1000, reltol=1e-05, abstol=1e-05)
    print("status:", prob.status)
    print("optimal value", prob.value)
    if prob.status == 'infeasible':
        return None
    else:
        return I.value

def SparsePlace(Jdes, Isafety, Itotal, Af=None, Ac=None):
    I = cvx.Variable(Af.shape[1])
    constraints = [Af@I== Jdes, cvx.sum(I) == 0, cvx.atoms.norm_inf(I) <= Isafety, cvx.atoms.norm1(I) <= Itotal]

    J = Ac@I
    term1 = cvx.atoms.square(cvx.atoms.norm(J))

    obj = cvx.Minimize(term1)
    prob = cvx.Problem(obj, constraints)
    try:
        prob.solve(verbose=True, max_iters=800, solver=cvx.ECOS)#solver=cvx.MOSEK #max_iters  # Returns the optimal value.
    except SolverError:
        prob.solve(verbose=True, solver=cvx.ECOS, max_iters=1000, reltol=1e-05, abstol=1e-05)
    print("status:", prob.status)
    print("optimal value", prob.value)
    if prob.status == 'infeasible':
        return None
    else:
        return I.value

def HingePlaceMultiSite_p1(Jdes, Isafety, Jtol, Itotal, Af=None, Ac=None):

    I = cvx.Variable(Af.shape[1])
    constraints = [cvx.sum(I) == 0, cvx.atoms.norm_inf(I) <= Isafety, cvx.atoms.norm1(I) <= Itotal]
    for i in range(len(Jdes)):
        constraints = constraints + [(Af[i].reshape(1,-1))@I== Jdes[i]]
    J = Ac@I
    term1 = cvx.sum(cvx.atoms.maximum(0, J - Jtol * np.ones((Ac.shape[0]))))
    term2 = cvx.sum(cvx.atoms.maximum(0, -J - Jtol * np.ones((Ac.shape[0]))))

    obj = cvx.Minimize(term1+term2)
    prob = cvx.Problem(obj, constraints)
    try:
        prob.solve(verbose=True, max_iters=300,solver=cvx.ECOS)  # solver=cvx.MOSEK #max_iters  # Returns the optimal value.
    except SolverError:
        prob.solve(verbose=True, solver=cvx.ECOS, max_iters=1000, reltol=1e-04, abstol=1e-04)
    print("status:", prob.status)
    print("optimal value", prob.value)
    if prob.status == 'infeasible':
        return None
    else:
        return I.value

def HingePlaceMultiSite_p2(Jdes, Isafety, Jtol, Itotal, Af=None, Ac=None):

    I = cvx.Variable(Af.shape[1])
    constraints = [cvx.sum(I) == 0, cvx.atoms.norm_inf(I) <= Isafety, cvx.atoms.norm1(I) <= Itotal]
    for i in range(len(Jdes)):
        constraints = constraints + [(Af[i].reshape(1,-1))@I== Jdes[i]]
    J = Ac@I

    term1 = cvx.sum(cvx.atoms.power(cvx.atoms.maximum(0, J - Jtol * np.ones((Ac.shape[0]))),2))
    term2 = cvx.sum(cvx.atoms.power(cvx.atoms.maximum(0, -J - Jtol * np.ones((Ac.shape[0]))),2))

    obj = cvx.Minimize(term1+term2)
    prob = cvx.Problem(obj, constraints)
    try:
        prob.solve(verbose=True, max_iters=300,solver=cvx.ECOS)  # solver=cvx.MOSEK #max_iters  # Returns the optimal value.
    except SolverError:
        prob.solve(verbose=True, solver=cvx.ECOS, max_iters=1000, reltol=1e-05, abstol=1e-05)
    print("status:", prob.status)
    print("optimal value", prob.value)
    if prob.status == 'infeasible':
        return None
    else:
        return I.value

def HingePlaceMultiSite_p3(Jdes, Isafety, Jtol, Itotal, Af=None, Ac=None):

    I = cvx.Variable(Af.shape[1])
    constraints = [cvx.sum(I) == 0, cvx.atoms.norm_inf(I) <= Isafety, cvx.atoms.norm1(I) <= Itotal]
    for i in range(len(Jdes)):
        constraints = constraints + [(Af[i].reshape(1,-1))@I== Jdes[i]]
    J = Ac@I
    term1 = cvx.sum(cvx.atoms.power(cvx.atoms.maximum(0, J - Jtol * np.ones((Ac.shape[0]))),3))
    term2 = cvx.sum(cvx.atoms.power(cvx.atoms.maximum(0, -J - Jtol * np.ones((Ac.shape[0]))),3))

    obj = cvx.Minimize(term1+term2)
    prob = cvx.Problem(obj, constraints)
    try:
        prob.solve(verbose=True, max_iters=800,solver=cvx.ECOS)  # solver=cvx.MOSEK #max_iters  # Returns the optimal value.
    except SolverError:
        prob.solve(verbose=True, solver=cvx.ECOS, max_iters=1000, reltol=1e-05, abstol=1e-05)
    print("status:", prob.status)
    print("optimal value", prob.value)
    if prob.status == 'infeasible':
        return None
    else:
        return I.value

def SparsePlaceMultiSite(Jdes, Isafety, Itotal, Af=None, Ac=None):

    I = cvx.Variable(Af.shape[1])
    constraints = [cvx.sum(I) == 0, cvx.atoms.norm_inf(I) <= Isafety, cvx.atoms.norm1(I) <= Itotal]

    for i in range(len(Jdes)):
        constraints = constraints + [(Af[i].reshape(1,-1))@I== Jdes[i]]

    J = Ac @ I
    term1 = cvx.atoms.square(cvx.atoms.norm(J))

    obj = cvx.Minimize(term1)
    prob = cvx.Problem(obj, constraints)
    try:
        prob.solve(verbose=True, max_iter=800,solver=cvx.CLARABEL)  # solver=cvx.MOSEK #max_iters  # Returns the optimal value.
    except SolverError:
        prob.solve(verbose=True, solver=cvx.ECOS, max_iters=1000, reltol=1e-05, abstol=1e-05)
    print("status:", prob.status)
    print("optimal value", prob.value)
    if prob.status == 'infeasible':
        return None
    else:
        return I.value