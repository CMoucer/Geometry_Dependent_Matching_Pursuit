import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

from generate_data.logreg_data import gradient_logreg
from archives.main_test_linear_opt_problem import realproblemB, problemA
from algorithms.proximal_gradient_descent import proximal_gradient_descent_lasso
from algorithms.chambolle_pock import chambolle_pock_G
from algorithms.alternating_minimization import alternating_minimization, ar_bcd


## UTILS FUNCTIONS for the regularization matching pursuit
def g_i(z, x, L, index_x):
    g = -z / L
    for l in index_x:
        g += np.abs(x[l])
    return g

def g_i_lip(z, x, Ls, index_x):
    g = -z
    for l in index_x:
        g += np.abs(x[l]) * np.sqrt(Ls[l])
    return g

def g_i_lmo(z, x, L, index_x):
    g = -z / L
    for l in index_x:
        g += x[l]
    return g


def regularized_matching_pursuit_zero(X, y, lam=1, num_iters=10, L=0):
    """
    This method returns the regularized matching pursuit starting from zero.
    :param X: input data (np.ndarray)
    :param y: labels to predict (np.ndarray)
    :param lam: regularization parameter
    :param num_iters: iteration number
    :param L: Lipschitz constant (to compute if L==0)
    :return: sequence produced by the method
    """
    n, d = X.shape
    if L == 0:
        L = max([np.linalg.norm(X[:, i]) for i in range(d)]) ** 2 / n

    # Compute the set of nonzero values for U for the initialization (with projected set)
    ws = np.zeros((num_iters, d))
    ws[0] = np.zeros(d)
    U = [] # U is empty at the beginning and corresponds to the set of active atoms
    ws_proj = [] # empty corresponding values

    U_follow = U.copy()
    ws_proj_follow = ws_proj.copy()

    # Compute the oracle and z_min
    for i in range(1, num_iters):

        # compute the minimal value for z with an LMO
        grad = 1 / n * X.T @ (X @ ws[i - 1] - y)
        grad_idx = -np.max(np.abs(grad))
        id_grad = np.argmax(np.abs(grad))
        P_idx = np.zeros(d)
        P_idx[id_grad] = -np.sign(grad[id_grad])
        z_min = max(0, - grad_idx - lam)
        # counter to evaluate if the LMO is an active atom or not + compute the set of projection onto the active atoms
        grad_P = np.zeros(len(U))
        idx = -1
        for u in range(len(U)):
            grad_P[u] = np.array(U[u]) @ grad
            if (P_idx == U[u]).all():
                idx = u

        if len(U) == 0 and z_min > 0:
            if idx == -1:
                U_follow.append(P_idx)
                ws_proj_follow.append(1 / L * z_min)
            assert len(ws_proj_follow) == len(U_follow)
            follow = 'new point'

        else:
            # Compute the values for non-zeros elements of ws : the zi
            z = np.zeros(len(U))
            for u in range(len(U)):
                z[u] = lam + grad_P[u]
            sorted_z = np.sort(z)
            argsort_z = np.argsort(z)
            if sorted_z[-1] <= z_min and z_min > 0:
                if idx == - 1:
                    U_follow.append(P_idx)
                    ws_proj_follow.append(1 / L * z_min)
                else:
                    ws_proj_follow[idx] += max(1 / L * z_min + ws_proj[idx], 0) - ws_proj[idx]
                assert len(ws_proj_follow) == len(U_follow)
                follow = 'I added one new point after sorting'

            elif z_min < 0 and sorted_z[-1] < z_min:
                print('You found the exact solution')
                for k in range(i, num_iters):
                    ws[k] = ws[i - 1]
                return ws

            else:
                # count the number of z_i that are bigger than z_min
                u = 0  # indices and at least one z[k] is larger than z_min
                index_maxi = []
                for k in range(len(U)):
                    if z_min > sorted_z[k]:
                        u += 1
                    else:
                        index_maxi.append(argsort_z[k])
                g_maxi = g_i_lmo(sorted_z[u], ws_proj, L, index_maxi)
                g_mini = g_i_lmo(z_min, ws_proj, L, index_maxi)

                # case 1 : z_opt is lower than z_min (g(z_min) <= 0)
                if g_mini <= 0:
                    # Set all larger points to 0 : z_i >= z_min
                    sum = 0
                    for k in range(u, len(U)):
                        index_k = argsort_z[k]
                        sum += ws_proj[index_k]
                        ws_proj_follow[index_k] = 0.
                    # add the point corresponding to the LMO
                    if idx == -1: # new point
                        ws_proj_follow.append(- max(1 / L * z_min - sum, 0))
                        U_follow.append(P_idx)
                    else: # already in the active atoms
                        ws_proj_follow[idx] += - (max(1 / L * z_min - sum + ws_proj[idx], 0) - ws_proj[idx])
                    follow = 'I removed points and added one possibly new'

                # case 2 : z_opt is in [z_min, z_u] g(z_min) > 0 and g_maxi < 0
                elif g_mini >= 0 and g_maxi <= 0:
                    # Set all larger points to 0 : z_i >= z_min
                    for k in range(u, len(U)):
                        index_k = argsort_z[k]
                        ws_proj_follow[index_k] = 0.
                        follow = 'I removed points'
                # case 3 : g_mini > 0 and g_maxi > 0
                else:
                    u_change = u
                    while g_mini >= 0 and g_maxi >= 0 and u_change + 2 <= len(U):
                        # modify the set index_maxi
                        index_u = argsort_z[u_change]
                        index_maxi.remove(index_u)
                        # compute the derivative on the left
                        g_mini = g_i_lmo(sorted_z[u_change], ws_proj, L, index_maxi)
                        # compute the derivative on the right
                        g_maxi = g_i_lmo(sorted_z[u_change + 1], ws_proj, L, index_maxi)
                        u_change += 1

                    if g_mini <= 0:
                        # the optimum is exactly in z_i, we delete points and update one
                        sum = 0
                        for k in range(u_change, len(U)):
                            index_k = argsort_z[k]
                            ws_proj_follow[index_k] = 0.
                            sum += ws_proj[index_k]
                        index_u = argsort_z[u_change - 1]
                        ws_proj_follow[index_u] += - (
                                    max((lam + grad_P[index_u]) / L - sum + ws_proj[index_u], 0) - ws_proj[index_u])
                        if ws_proj_follow[index_u] == 0:
                            ws_proj_follow[index_u] = 0.
                        follow = 'I removed points, and modify at most one'
                    elif g_mini >= 0 and g_maxi <= 0:
                        # the optimum is in the middle : we only delete some intermediary points
                        for k in range(u_change, len(U)):
                            index_k = argsort_z[k]
                            ws_proj_follow[index_k] = 0.
                        follow = 'I removed the points in between'

                    else:
                        # No points are set to zero, and the last one is updated
                        assert u_change == len(U) - 1, 'error somewhere'
                        index_u = argsort_z[u_change]
                        ws_proj_follow[index_u] += - (
                                max((lam + grad_P[index_u]) / L + ws_proj[index_u], 0) - ws_proj[index_u])
                        if ws_proj_follow[index_u] == 0:
                            ws_proj_follow[index_u] = 0.
                        follow = 'I updated the last point'
        ws[i] = np.array(U_follow).T @ np.array(ws_proj_follow)
        for k in reversed(range(len(U))):
            if ws_proj_follow[k] == 0.:
                del ws_proj_follow[k]
                del U_follow[k]
        U = U_follow.copy()
        ws_proj = ws_proj_follow.copy()
    return ws

def regularized_matching_pursuit(X, y, w_0, lam=1, num_iters=10, L=0):
    """
    This method returns the regularized matching pursuit.
    :param X: input data (np.ndarray)
    :param y: labels to predict (np.ndarray)
    :param w_0: starting point (np.ndarray)
    :param lam: regularization parameter
    :param num_iters: iteration number
    :param L: Lipschitz constant (to compute of L==0)
    :return:
    """
    n, d = X.shape
    assert d == np.shape(w_0)[0]
    if L == 0:
        L = max([np.linalg.norm(X[:, i]) for i in range(d)]) ** 2 / n

    # Compute the set of nonzero values for U for the initialization (with projected set)
    ws = np.zeros((num_iters, d))
    ws[0] = w_0
    U = [] # U is empty at the beginning and corresponds to the set of active atoms
    ws_proj = [] # empty corresponding values

    U_follow = U.copy()
    ws_proj_follow = ws_proj.copy()

    # Compute the oracle and z_min
    for i in range(1, num_iters):

        # compute the minimal value for z with an LMO
        grad = 1 / n * X.T @ (X @ ws[i - 1] - y)
        grad_idx = -np.max(np.abs(grad))
        id_grad = np.argmax(np.abs(grad))
        P_idx = np.zeros(d)
        P_idx[id_grad] = -np.sign(grad[id_grad])
        z_min = max(0, - grad_idx - lam)
        # counter to evaluate if the LMO is an active atom or not + compute the set of projection onto the active atoms
        grad_P = np.zeros(len(U))
        idx = -1
        for u in range(len(U)):
            grad_P[u] = np.array(U[u]) @ grad
            if (P_idx == U[u]).all():
                idx = u

        if len(U) == 0 and z_min > 0:
            if idx == -1:
                U_follow.append(P_idx)
                ws_proj_follow.append(1 / L * z_min)
            assert len(ws_proj_follow) == len(U_follow)
            follow = 'new point'

        else:
            # Compute the values for non-zeros elements of ws : the zi
            z = np.zeros(len(U))
            for u in range(len(U)):
                z[u] = lam + grad_P[u]
            sorted_z = np.sort(z)
            argsort_z = np.argsort(z)
            if sorted_z[-1] <= z_min and z_min > 0:
                if idx == - 1:
                    U_follow.append(P_idx)
                    ws_proj_follow.append(1 / L * z_min)
                else:
                    ws_proj_follow[idx] += max(1 / L * z_min + ws_proj[idx], 0) - ws_proj[idx]
                assert len(ws_proj_follow) == len(U_follow)
                follow = 'I added one new point after sorting'

            elif z_min < 0 and sorted_z[-1] < z_min:
                print('You found the exact solution')
                for k in range(i, num_iters):
                    ws[k] = ws[i - 1]
                return ws

            else:
                # count the number of z_i that are bigger than z_min
                u = 0  # indices and at least one z[k] is larger than z_min
                index_maxi = []
                for k in range(len(U)):
                    if z_min > sorted_z[k]:
                        u += 1
                    else:
                        index_maxi.append(argsort_z[k])
                g_maxi = g_i_lmo(sorted_z[u], ws_proj, L, index_maxi)
                g_mini = g_i_lmo(z_min, ws_proj, L, index_maxi)

                # case 1 : z_opt is lower than z_min (g(z_min) <= 0)
                if g_mini <= 0:
                    # Set all larger points to 0 : z_i >= z_min
                    sum = 0
                    for k in range(u, len(U)):
                        index_k = argsort_z[k]
                        sum += ws_proj[index_k]
                        ws_proj_follow[index_k] = 0.
                    # add the point corresponding to the LMO
                    if idx == -1: # new point
                        ws_proj_follow.append(- max(1 / L * z_min - sum, 0))
                        U_follow.append(P_idx)
                    else: # already in the active atoms
                        #print(idx)
                        ws_proj_follow[idx] += - (max(1 / L * z_min - sum + ws_proj[idx], 0) - ws_proj[idx])
                    follow = 'I removed points and added one possibly new'

                # case 2 : z_opt is in [z_min, z_u] g(z_min) > 0 and g_maxi < 0
                elif g_mini >= 0 and g_maxi <= 0:
                    # Set all larger points to 0 : z_i >= z_min
                    for k in range(u, len(U)):
                        index_k = argsort_z[k]
                        ws_proj_follow[index_k] = 0.
                        follow = 'I removed points'
                # case 3 : g_mini > 0 and g_maxi > 0
                else:
                    u_change = u
                    while g_mini >= 0 and g_maxi >= 0 and u_change + 2 <= len(U):
                        # modify the set index_maxi
                        index_u = argsort_z[u_change]
                        index_maxi.remove(index_u)
                        # compute the derivative on the left
                        g_mini = g_i_lmo(sorted_z[u_change], ws_proj, L, index_maxi)
                        # compute the derivative on the right
                        g_maxi = g_i_lmo(sorted_z[u_change + 1], ws_proj, L, index_maxi)
                        u_change += 1

                    if g_mini <= 0:
                        # the optimum is exactly in z_i, we delete points and update one
                        sum = 0
                        for k in range(u_change, len(U)):
                            index_k = argsort_z[k]
                            ws_proj_follow[index_k] = 0.
                            sum += ws_proj[index_k]
                        index_u = argsort_z[u_change - 1]
                        ws_proj_follow[index_u] += - (
                                    max((lam + grad_P[index_u]) / L - sum + ws_proj[index_u], 0) - ws_proj[index_u])
                        if ws_proj_follow[index_u] == 0:
                            ws_proj_follow[index_u] = 0.
                        follow = 'I removed points, and modify at most one'
                    elif g_mini >= 0 and g_maxi <= 0:
                        # the optimum is in the middle : we only delete some intermediary points
                        for k in range(u_change, len(U)):
                            index_k = argsort_z[k]
                            ws_proj_follow[index_k] = 0.
                        follow = 'I removed the points in between'

                    else:
                        # No points are set to zero, and the last one is updated
                        assert u_change == len(U) - 1, 'error somewhere'
                        index_u = argsort_z[u_change]
                        ws_proj_follow[index_u] += - (
                                max((lam + grad_P[index_u]) / L + ws_proj[index_u], 0) - ws_proj[index_u])
                        if ws_proj_follow[index_u] == 0:
                            ws_proj_follow[index_u] = 0.
                        follow = 'I updated the last point'
        ws[i] = np.array(U_follow).T @ np.array(ws_proj_follow)
        for k in reversed(range(len(U))):
            if ws_proj_follow[k] == 0.:
                del ws_proj_follow[k]
                del U_follow[k]
        U = U_follow.copy()
        ws_proj = ws_proj_follow.copy()
    return ws

























