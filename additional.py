import math
import numpy as np
import torch as th

import scipy.optimize as sco


def dijkstra_projection(PC, PD, r, max_iter = 5):
    x = r
    p = 0
    q = 0

    for i in range(0,max_iter):
        y = PD(x + p)
        p = x + p - y
        x = PC(y + q)
        q = y + q - x



def conj_grad(mmv, X0, B, cg_tolerance, max_iter):

    R = B - mmv(X0)
    X = X0

    m_eps = 1e-16

    P = R
    Rsold = th.sum(R.pow(2), dim=0)


    for i in range(max_iter):
        AP = mmv(P)
        alpha = Rsold / (th.sum(P * AP, dim=0) + m_eps)
        X.addmm_(P, th.diag(alpha))

        if (i + 1) % 5:
            R = B - mmv(X)
        else:
            R = R - th.mm(AP, th.diag(alpha))

        Rsnew = th.sum(R.pow(2), dim=0)
        if Rsnew.abs().max().sqrt() < cg_tolerance:
            break

        P = R + th.mm(P, th.diag(Rsnew / (Rsold + m_eps)))
        Rsold = Rsnew

    return X



def minimize_scalar_bounded(func, bounds, min_f = math.inf, xatol=1e-5, maxiter=500):

    maxfun = maxiter

    x1, x2 = bounds

    flag = 0

    sqrt_eps = np.sqrt(2.2e-16)
    golden_mean = 0.5 * (3.0 - np.sqrt(5.0))
    a, b = x1, x2
    fulc = a + golden_mean * (b - a)
    nfc, xf = fulc, fulc
    rat = e = 0.0
    x = xf
    fx = func(x)
    num = 1

    fu = np.inf

    ffulc = fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1


    while (np.abs(xf - xm) > (tol2 - 0.5 * (b - a)) or fx > min_f - sqrt_eps):
        golden = 1
        # Check for parabolic fit
        if np.abs(e) > tol1:
            golden = 0
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = np.abs(q)
            r = e
            e = rat

            # Check for acceptability of parabola
            if ((np.abs(p) < np.abs(0.5*q*r)) and (p > q*(a - xf)) and
                    (p < q * (b - xf))):
                rat = (p + 0.0) / q
                x = xf + rat
                step = '       parabolic'

                if ((x - a) < tol2) or ((b - x) < tol2):
                    si = np.sign(xm - xf) + ((xm - xf) == 0)
                    rat = tol1 * si
            else:      # do a golden-section step
                golden = 1

        if golden:  # do a golden-section step
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean*e
            step = '       golden'

        si = np.sign(rat) + (rat == 0)
        x = xf + si * max(np.abs(rat), tol1)
        fu = func(x)
        num += 1

        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if (fu <= fnfc) or (nfc == xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc, ffulc = x, fu

        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1

        if num >= maxfun:
            flag = 1
            break

    if np.isnan(xf) or np.isnan(fx) or np.isnan(fu):
        flag = 2

    fval = fx

    result = dict(fun=fval, status=flag, success=(flag == 0), a = a, b = b,
                  x=xf, nfev=num)

    return result


def max_step(F, data = None, iter_max = 100):
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
            print('error')
            return math.inf

    eyeI = th.eye(M,M, device = Q.device, dtype = Q.dtype)

    normQ = th.linalg.matrix_norm(Q, ord = 2)
    max_eigQ = th.linalg.matrix_norm(Q + normQ*eyeI, ord = 2) - normQ

    bb = True
    if 1.0/max_eigQ > 1 and l < (3.0 - math.sqrt(5.0))/2.0:
        if evalFt(1.0) < f:
            t = 1
            bb = False

    if bb:
        a = 0
        b = 1/max(1,max_eigQ)
        tol = (b - a)*0.1

        res = minimize_scalar_bounded(evalFt, (float(a), float(b)), min_f = float(f), xatol = float(tol), maxiter = iter_max - it)
        t = res['x']

    evalFt(t, level = 2)
    data_new = ddd[t]
    data_new['iter'] += data['iter']
    data_new['step'] = t
    return data_new




def follow_tau(F, data = None, max_newt_dec = 0.33, iter_max = 100, verbose = True):
    ddd = dict()
    L, it, tau = data['L'], data['iter'], data['tau']

    def evalFt(t):
        data = F(L, t, level = 2)
        ddd[t] = data
        #print('search tau={:.6E}, Dl={:.6E}'.format(float(t),float(l1 - max_newt_dec)))
        return data['newt_dec'] - max_newt_dec


    it += 1
    if evalFt(2.0*tau) < 0:
        v = 2.0*tau
    else:
        v, r = sco.brentq(evalFt,a = tau, b = 10*tau, maxiter = iter_max - it, rtol = 1e-2, full_output=True)
        it += r.function_calls

    if v <= tau:
        v = 1.01*tau
        evalFt(v)
        it = it + 1

    print('chosen tau = ', v)
    data = ddd[v]
    data['iter'] = it

    verbose(data)

    return data