import torch as th
import math
import unittest

th.set_default_dtype(th.float64)

### in this code, easy version of integration. We show

def sqdist(X, Y):
    if Y is None:
        norm = (X**2).sum(1)
        K = X @ X.T
        K *= -2
        K += norm[:, None]
        K += norm[None, :]
        return K
    else:
        normX = (X ** 2).sum(1)
        normY = (Y ** 2).sum(1)
        K = X @ Y.T
        K *= -2
        K += normX[:, None]
        K += normY[None, :]
        return K


def gaussKern(X, Y, gamma):
    sg = gamma.sqrt()
    if Y is None:
        K = sqdist(X*sg,None)
    else:
        K = sqdist(X * sg, Y * sg)
    K *= -1
    K.clamp_(min=-30, max=0)
    K.exp_()
    return K


def gaussIntegrate(X, gamma, a, b,mask = None):
    '''
    :param X: matrix of dimension N times d
    :param a: vector of dimension m times d
    :param b: vector of dimension m times d
    :param mask : boolean matrix of dimension N where M[i,j] should be set to 0 automatically outside the mask.
    :return: M of dimension N times m where M[i,j] = int_[a[j,:],b[j,:]] exp(-gamma*(x-X[i,:])**2) dx
    '''

    sg =gamma.sqrt()
    X = X*sg
    a = a*sg
    b = b*sg

    c = math.sqrt(math.pi)/2.0/sg
    if mask == None:
        return (c*(th.erf(b.unsqueeze(0) - X.unsqueeze(1)) - th.erf(a.unsqueeze(0) - X.unsqueeze(1)))).prod(2)
    else:
        Bx = b.unsqueeze(0) - X.unsqueeze(1)
        Ax = a.unsqueeze(0) - X.unsqueeze(1)
        Bx[mask[:,None,None]] = th.erf(Bx.masked_select[mask[:,None,None]])
        Ax[mask[:,None,None]] = th.erf(Ax.masked_select[mask[:,None,None]])
        Bx -= Ax
        Bx[~mask[:,None,None]] = 0
        res = (c*Bx).prod(2)
        return res

def gaussIntegrateProfiled(X, gamma, a, b,mask = None):
    '''
    :param X: matrix of dimension N times d
    :param a: vector of dimension m times d
    :param b: vector of dimension m times d
    :param mask : boolean matrix of dimension N where M[i,j] should be set to 0 automatically outside the mask.
    :return: M of dimension N times m where M[i,j] = int_[a[j,:],b[j,:]] exp(-gamma*(x-X[i,:])**2) dx
    '''

    sg =gamma.sqrt()
    X = X*sg
    a = a*sg
    b = b*sg

    with th.profiler.record_function('INTEGRATION_DOMAINS'):

        c = (math.sqrt(math.pi)/2.0/sg).prod().item()
        Bx = b.unsqueeze(0) - X.unsqueeze(1)
        Ax = a.unsqueeze(0) - X.unsqueeze(1)
    with th.profiler.record_function('APPLY_ERF'):
        if not(mask==None):
            Bx[mask[:,None,None]] = th.erf(Bx.masked_select[mask[:,None,None]])
            Ax[mask[:,None,None]] = th.erf(Ax.masked_select[mask[:,None,None]])
        else:
            Bx = th.erf(Bx)
            Ax = th.erf(Ax)
    with th.profiler.record_function('COMPUTE_DIFFERENCES_INTEGRALS'):
        Bx -= Ax
        if not(mask == None):
            Bx[~mask[:,None,None]] = 0
    with th.profiler.record_function('PRODUCT_ALONG_DIMENSIONS'):
        res = Bx.prod(2)
        res*=c
    return res

def integrate(A,X,gamma,Q,tol = None):
    """
    Usage


    :param g: g is a function that takes X of dimension N times d, gamma of dimension d and
    gives in output a matrix M of dimension N times m (with m >= 1), in particular
    must correspond to the close form of the following integral
    g(X,gamma)[i,:] = integral f(x) exp(-(gamma*(x-X[i,:])**2).sum()) dx
    for some function f that takes a d-dimensional vector in input and gives an m-dimensional vector in output

    :return:
    the output is a vector of dimension m that corresponds to
    model.integrate(g) = integral f(x) model(x) dx
    """

    B = gaussKern(X, X, gamma / 2).view(-1,1)

    if tol :
        mask = (B>tol)


    B *= A.view(-1,1)
    if tol:
        B= B.masked_select(mask).view(-1,1)


    d = X.size(1)
    Xm = 0.5*(X.view(-1,1,d) + X.view(1,-1,d)).view(-1,d)

    if tol:
        X_to_integrate = Xm.masked_select(mask.expand_as(Xm)).view(-1,d)
    else:
        X_to_integrate = Xm


    #need to compute integral
    a,b =Q[0,:].unsqueeze(0),Q[1,:].unsqueeze(0)
    G = gaussIntegrate(X_to_integrate, 2.0*gamma,a,b)

    return (G * B).sum(dim=0)


def integrateHypercube(A,X,gamma,a,b,tol = None):
    """
    Usage


    :param g: g is a function that takes X of dimension N times d, gamma of dimension d and
    gives in output a matrix M of dimension N times m (with m >= 1), in particular
    must correspond to the close form of the following integral
    g(X,gamma)[i,:] = integral f(x) exp(-(gamma*(x-X[i,:])**2).sum()) dx
    for some function f that takes a d-dimensional vector in input and gives an m-dimensional vector in output

    :return:
    the output is a vector of dimension m that corresponds to
    model.integrate(g) = integral f(x) model(x) dx
    """

    B = gaussKern(X, X, gamma / 2).view(-1,1)

    if tol :
        mask = (B>tol)


    B *= A.view(-1,1)
    if tol:
        B= B.masked_select(mask).view(-1,1)


    d = X.size(1)
    Xm = 0.5*(X.view(-1,1,d) + X.view(1,-1,d)).view(-1,d)

    if tol:
        X_to_integrate = Xm.masked_select(mask.expand_as(Xm)).view(-1,d)
    else:
        X_to_integrate = Xm


    #need to compute integral
    G = gaussIntegrate(X_to_integrate, 2.0*gamma,a,b)

    return (G * B).sum(dim=0)

def integrateProfiled(A,X,gamma,Q,tol = None):
    """
    Usage


    :param g: g is a function that takes X of dimension N times d, gamma of dimension d and
    gives in output a matrix M of dimension N times m (with m >= 1), in particular
    must correspond to the close form of the following integral
    g(X,gamma)[i,:] = integral f(x) exp(-(gamma*(x-X[i,:])**2).sum()) dx
    for some function f that takes a d-dimensional vector in input and gives an m-dimensional vector in output

    :return:
    the output is a vector of dimension m that corresponds to
    model.integrate(g) = integral f(x) model(x) dx
    """

    with th.profiler.record_function('Computing gaussian kernel'):

        B = gaussKern(X, X, gamma / 2).view(-1,1)

        if tol :
            mask = (B>tol)


        B *= A.view(-1,1)
        if tol:
            B= B.masked_select(mask).view(-1,1)

    with th.profiler.record_function('Computing matrix of centroids X_i + X_j'):
        d = X.size(1)
        Xhalf = 0.5*X
        Xm = (Xhalf.view(-1,1,d) + Xhalf.view(1,-1,d)).view(-1,d)

        if tol:
            X_to_integrate = Xm.masked_select(mask.expand_as(Xm)).view(-1,d)
        else:
            X_to_integrate = Xm


        #need to compute integral
        a,b =Q[0,:].unsqueeze(0),Q[1,:].unsqueeze(0)
    with th.profiler.record_function('Computation of the matrix G'):
        G = gaussIntegrateProfiled(X_to_integrate, 2.0*gamma,a,b)

    return (G * B).sum(dim=0)

def sampleProfiled(AA,X,g,Q, N, tol=1e-3,tolInt=None):
    '''
    :param N: numbers of i.i.d. samples to take from the model
    :param tol: (default 1e-3) tolerance of the sampling strategy
    :return: N i.i.d. samples distributed according to the model.
    '''

    with th.profiler.record_function('INITIAL_COMPUTATIONS'):


        BB = gaussKern(X, X, g / 2).view(-1, 1)

        if tolInt:
            mask = (BB > tolInt)

        BB *= AA.view(-1, 1)
        if tolInt:
            BB = BB.masked_select(mask).view(-1, 1)

        d = X.size(1)
        Xhalf = 0.5*X
        Xm =  (Xhalf.view(-1, 1, d) + Xhalf.view(1, -1, d)).view(-1, d)

        if tolInt:
            X_to_integrate = Xm.masked_select(mask.expand_as(Xm)).view(-1, d)
        else:
            X_to_integrate = Xm



        a, b = Q[0, :].view(1, -1), Q[1, :].view(1, -1)
        with th.profiler.record_function('INTEGRAL'):
            G = gaussIntegrate(X_to_integrate, 2.0 * g, a, b)
            integral = (G*BB).sum(dim=0).item()

        steps = th.ceil(th.log((Q[1, :] - Q[0, :]) / tol) / math.log(2)).clamp(min=0).type(th.LongTensor)
        direction_array = th.zeros(int(steps.sum().item()),dtype = th.int64)
        j = 0
        for i in range(steps.size(0)):
            s = steps[i].item()
            direction_array[j:j + s] = i
            j += s

        bins = th.tensor([N],dtype = th.int64)
        A = a
        B = b
        integrals = th.tensor([integral])

    for i0 in direction_array:
        i0 = int(i0)
        if integrals.size(0) < N:
            with th.profiler.record_function('DEFINE_NEW_HYPERCUBES'):

                A_half = A.clone()
                A_half[:, i0] = 0.5 * (B[:, i0] + A[:, i0])

                B_half = B.clone()
                B_half[:,i0] = 0.5*(B[:,i0]+ A[:,i0])
            with th.profiler.record_function('INTEGRAL'):
                G = gaussIntegrate(X_to_integrate, 2.0 * g, A, B_half)
                integrals_next = (G * BB).sum(dim=0)
            p = integrals_next/integrals


            bins_next = th.zeros(bins.size(),dtype =th.int64)
            for i,Nbin in enumerate(bins):
                bins_next[i] = int((th.rand(Nbin.item()) < p[i]).sum().item())
            bins = th.cat([bins_next,bins-bins_next])

            integrals = th.cat([integrals_next,integrals-integrals_next])
            A = th.cat([A,A_half])
            B = th.cat([B_half,B])
            mask_bins = (bins>0)
            bins = bins[mask_bins]
            A = A[mask_bins,:]
            B = B[mask_bins,:]
            integrals = integrals[mask_bins]

        else:
            with th.profiler.record_function('DEFINE_NEW_HYPERCUBES'):
                A_half = A.clone()
                A_half[:, i0] = 0.5 * (B[:, i0] + A[:, i0])

                B_half = B.clone()
                B_half[:, i0] = 0.5 * (B[:, i0] + A[:, i0])
            with th.profiler.record_function('INTEGRAL'):
                G = gaussIntegrate(X_to_integrate, 2.0 * g, A, B_half)
                integrals_next = (G * BB).sum(dim=0)
            p = integrals_next / integrals
            mask_bins = (th.rand(N) < p).type(th.LongTensor)
            integrals = mask_bins * integrals_next + (1-mask_bins)*(integrals-integrals_next)
            A = mask_bins[:,None] * A + (1-mask_bins)[:,None]*A_half
            B = mask_bins[:,None] * B_half + (1-mask_bins)[:,None]*B
    if integrals.size() == N:
        samples = A + (B-A)*th.rand(N,d)
    else:
        lsamples = []
        for i in range(A.size(0)):
            lsamples.append((A[i,:].view(1,-1) + (B[i,:]-A[i,:])[None,:]*th.rand(bins[i],d)))
        samples = th.cat(lsamples)

    return samples[th.randperm(N),:]

n = 50
d = 5
Q = th.ones(2,d,dtype = th.float64)
Q[0,:] *=-1
X = 2*th.rand(n,d) - 1
gamma = 0.1+th.rand(1,d)

k=1
aux = th.rand(n,k)
A = aux@aux.T
N = 100


class MyTestCase(unittest.TestCase):
    def test_cost_integrate(self):

        with th.profiler.profile(
                activities=[
                    th.profiler.ProfilerActivity.CPU
                ]) as profiler:
            integrateProfiled(A, X, gamma, Q, tol=1e-14)
        print(profiler.key_averages().table(sort_by = 'cpu_time_total',row_limit = 10))
        # sorted bu
        return None
    def test_cost_sample(self):
        with th.profiler.profile(
                activities=[
                    th.profiler.ProfilerActivity.CPU
                ]) as profiler:
            sampleProfiled(A, X, gamma, Q, N, tol=1e-3, tolInt=None)
        print(profiler.key_averages().table(sort_by='cpu_time_total', row_limit=10))
        # sorted bu
        return None



if __name__ == '__main__':
    unittest.main()