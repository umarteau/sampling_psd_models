import math
import numpy as np
import torch as th

from scipy.sparse.linalg import cg as conj_grad

def nrmH(f, iH):
    return math.sqrt(f.T @ iH(f))


def pcg(A, b, x, invM = None, min_res = 1e-13, it=0, iter_max = 100):
    if invM == None:
        invM = lambda x: x

    r = b - A(x)
    c_r = r
    c0_r = r
    c_x = x
    z = invM(r)
    p = z

    while it < iter_max:
        Ap = A(p)
        alpha = (r.T @ z)/(p.T @ Ap)
        x = x + alpha*p

        rnew = b - A(x)
        #rnew = r - alpha*Ap

        if rnew.norm() < c_r.norm():
            c_x = x
            c_r = r

        if rnew.norm() <= min_res:
            break

        znew = invM(rnew)
        beta = (rnew.T @ znew)/(r.T @ z)
        p = znew + beta*p
        z = znew
        r = rnew
        it += 1

        #print(float(c0_err/rnew.norm()))

    return c_x, c_r, c_r.norm() / c0_r.norm()


def backtrack_step(F, data = None, iter_max = 100):
    ddd = dict()
    Q, l, f, L, it = data['iWtDelta'], data['newt_dec'], data['f'], data['L'], data['iter']
    M = Q.shape[0]

    def evalFt(t, level = 0):
        try:
            N = th.linalg.cholesky(th.eye(M,M, device = Q.device, dtype = Q.dtype) - t*Q)
            X1 = L @ N
            ddd[t] = F(X1, level = level)
            #print('step t={:.6E}, f={:.6E}'.format(float(t),float(f1)))
            return float(ddd[t]['f'])
        except BaseException:
            #print('error')
            return math.inf

    eyeI = th.eye(M,M, device = Q.device, dtype = Q.dtype)

    normQ = th.linalg.matrix_norm(Q, ord = 2)
    max_eigQ = th.linalg.matrix_norm(Q + normQ*eyeI, ord = 2) - normQ


    alpha = 0.1
    beta = 0.8

    t = 1.0/max(1.0, 1.01*max_eigQ)

    q = data['WDf'].T @ data['iWtDelta'].reshape(-1,1)

    while evalFt(t) > f - alpha*t*q:
        t *= beta

    evalFt(t, level = 2)
    data_new = ddd[t]
    data_new['iter'] += data['iter']
    data_new['step'] = t
    return data_new


def damped_newton_sc(F, L, data = None, min_newt_dec = 1e-3, max_constr_violation = 1e-12, it = 0, iter_max=100, verbose = True):

    if data is None:
        data = F(L, level = 2)
        data['iter'] += it
        verbose(data)

    while data['iter'] < iter_max and data['newt_dec'] > min_newt_dec:
            #data = max_step(F, data = data, iter_max = iter_max)
            data = backtrack_step(F, data = data, iter_max = iter_max)

            verbose(data)

    return data


def constrained_newton_method(funF, L, min_err = 1e-12, iter_max = 100, refinement = 30, newton_descent = damped_newton_sc, path_following_factor = 2, verbose=True):

    vrb = verbose
    if vrb == True:
        def verbose(data):
            if 'step' in data.keys():
                print("it:{:5d}\tobj:{:.6E}\tviol:{:.6E}\tnewt_dec:{:.6E}\ttau:{:.6E}\tt:{:.6E}".format(data['iter'],  float(data['f']/data['tau']), float(data['cnstr_viol']), float(data['newt_dec']), float(data['tau']), float(data['step'])))
            else:
                print("it:{:5d}\tobj:{:.6E}\tviol:{:.6E}\tnewt_dec:{:.6E}\ttau:{:.6E}".format(data['iter'],  float(data['f']/data['tau']), float(data['cnstr_viol']), float(data['newt_dec']), float(data['tau'])))
    elif vrb == False:
        def verbose(data):
            pass

    def F(L, tau, level):

        level = level + int(level >= 2)

        data = funF(L, tau, level)
        data['iter'] = int(level >= 2)
        data['tau'] = tau

        if level >= 2 and refinement > 0:
            q, r, rel_err = pcg(data['ext_WtHW'], data['ext_WDf'],
                       data['ext_iWtDelta'], invM = data['inv_ext_WtHW'], min_res = 1e-30, it=0, iter_max = refinement)

            #print(float(rel_err))
            M = L.shape[0]
            data['iWtDelta'] = q[0:M*M,:].reshape(M,M)
            data['ext_iWtDelta'] = q

            data['newt_dec'] = (data['WDf'].T @ data['iWtDelta'].reshape(-1,1)).abs().sqrt()

        return data

    G = lambda tau: lambda L, level: F(L, tau, level)

    tau = 1e-3

    if vrb:
        print('Search of the initial solution')

    #creating starting solution in the central path
    data = newton_descent(G(tau), L, min_newt_dec = 0.33, iter_max = iter_max, verbose = verbose)


    if vrb:
        print(' ')
        print('Path following')
    #putting tau to 2/min_err

    max_tau = 2.0/min_err

    while data['iter'] < iter_max and data['tau'] < max_tau:

        data['tau'] = data['tau']*path_following_factor
        print('chosen tau = ', data['tau'])
        data = newton_descent(G(data['tau']), data['L'], data = None,
                min_newt_dec = 0.38, it = data['iter'], iter_max = iter_max, verbose = verbose)

        # data = follow_tau(F, data = data, max_newt_dec = 10, iter_max = iter_max, verbose = verbose)
        # data = newton_descent(G(data['tau']), data['L'], data = data,
        #        min_newt_dec = 0.5, iter_max = iter_max, verbose = verbose)

    if vrb:
        print(' ')
        print('Second order convergence')
    return newton_descent(G(data['tau']), data['L'], data = data,
                          min_newt_dec = math.sqrt(2.0*min_err),
                          iter_max = iter_max + 10, verbose = verbose)




