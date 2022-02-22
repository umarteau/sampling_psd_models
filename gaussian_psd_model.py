import torch as th
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
from joblib import Parallel,delayed

import scipy.optimize as sco

from cvxprog import constrained_newton_method


def gcd(lst):
    u = lst[0]
    for k in lst:
        u = math.gcd(u, k)
    return u


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


class bs_matrix:
    """
    Models a matrix which is of the form 1_{otimes_left} \otimes X \otimes 1_{otimes_right}
    """
    def __init__(self, X, otimes_left=1, otimes_right=1, int_a=None, int_b=None):
        self.X = X
        self.d = X.shape[1]
        self.comprN = X.shape[0]
        self.totalN = X.shape[0] * otimes_left * otimes_right
        self.otimes_left = otimes_left
        self.otimes_right = otimes_right

        if int_a == None:
            int_a = -math.inf * th.ones(1, X.shape[1], dtype=X.dtype, device = X.device)
        self.int_a = int_a

        if int_b == None:
            int_b = math.inf * th.ones(1, X.shape[1], dtype=X.dtype, device = X.device)
        self.int_b = int_b

    def copy(self):
        return bs_matrix(self.X,
                         otimes_left=self.otimes_left, otimes_right=self.otimes_right,
                         int_a=self.int_a, int_b=self.int_b)

    def kron_left(self, N, D=1):
        assert (int(self.otimes_left * N / D) == self.otimes_left * N / D)
        return bs_matrix(self.X,
                         otimes_left=int(self.otimes_left * N / D), otimes_right=self.otimes_right,
                         int_a=self.int_a, int_b=self.int_b)

    def kron_right(self, N, D=1):
        assert int(self.otimes_right * N / D) == self.otimes_right * N / D
        return bs_matrix(self.X,
                         otimes_left=self.otimes_left, otimes_right=int(self.otimes_right * N / D),
                         int_a=self.int_a, int_b=self.int_b)

    def decompress(self, out_otimes_left=1, out_otimes_right=1):
        l = int(self.otimes_left / out_otimes_left)
        r = int(self.otimes_right / out_otimes_right)
        assert int(l) == l
        assert int(r) == r
        return bs_matrix(
            self.X.repeat([l, 1]).repeat_interleave(r, dim=0),
            otimes_left=out_otimes_left, otimes_right=out_otimes_right,
            int_a=self.int_a, int_b=self.int_b)

    def binary_op(self, Y, op):
        gcd_left = math.gcd(self.otimes_left, Y.otimes_left)
        gcd_right = math.gcd(self.otimes_right, Y.otimes_right)

        decX = self.decompress(out_otimes_left=gcd_left, out_otimes_right=gcd_right)
        decY = self.decompress(out_otimes_left=gcd_left, out_otimes_right=gcd_right)

        return bs_matrix(op(decX.X, decY.X),
                         otimes_left=gcd_left, otimes_right=gcd_right,
                         int_a=th.maximum(self.int_a, Y.int_a), int_b=th.minimum(self.int_b, Y.int_b)
                         )

    def __add__(self, other):
        return self.binary_op(other, th.add)

    def __mul__(self, M):
        b = self.copy()
        b.X = b.X * M
        return b


def create_structured_matrix_from_X(X,**kwargs):
    ss = {}
    for k in kwargs.keys():
        ss[k] = bs_matrix(X[:, kwargs[k]])
    return structured_matrix(ss)

class structured_matrix(dict):

    def __init__(self, sZ):
        d = 0
        N = 0
        for k in sZ:
            if isinstance(sZ[k], bs_matrix):
                self[k] = sZ[k]
            else:
                self[k] = bs_matrix(th.tensor(sZ[k]))

            if N == 0:
                N = self[k].totalN
            assert self[k].totalN == N

            d = d + self[k].d

        self.N = N
        self.total_d = d

    def to_bs_matrix(self):
        vr = list(self.keys())
        vr.sort()
        gcd_left = gcd([self[k].otimes_left for k in vr])
        gcd_right = gcd([self[k].otimes_right for k in vr])
        smallN = int(self.N / gcd_left / gcd_right)

        dt = self[vr[0]].X.dtype
        dv = self[vr[0]].X.device
        X = th.zeros(smallN, self.total_d, dtype=dt, device=dv)
        int_a = th.zeros(1, self.total_d, dtype=dt, device=dv)
        int_b = th.zeros(1, self.total_d, dtype=dt, device=dv)
        d = 0
        for k in vr:
            bs = self[k].decompress(out_otimes_left=gcd_left, out_otimes_right=gcd_right)
            d1 = d + bs.d
            X[:, d:d1] = bs.X
            int_a[:, d:d1] = bs.int_a
            int_b[:, d:d1] = bs.int_b
            d = d1

        return bs_matrix(X, otimes_left=gcd_left, otimes_right=gcd_right, int_a=int_a, int_b=int_b)

    def subset(self, vars):
        return structured_matrix({k: self[k] for k in vars})

    def rename(self, kwargs):
        """
        Usage

        >>> g.rename(x = 'y', z = 'x')

        :param kwargs:
        :return:
        """

        assert(len(set(kwargs.values())) == len(kwargs.values()))
        assert(set(self.X.keys()).difference(kwargs.keys()).intersection(kwargs.values()) == {})


        for k in kwargs.keys():
            u = self.X.pop(k)
            self.X[kwargs[k]] = u

            u = self.gamma.pop(k)
            self.gamma[kwargs[k]] = u





###### UTILITIES


def reduce(B,X):
    gl = gcd([X[k].otimes_left for k in X.keys()])
    gr = gcd([X[k].otimes_right for k in X.keys()])
    N = B.shape[0]
    rB = B.reshape((gl, int(N / gl / gr), gr, gl, int(N / gl / gr), gr)).sum(dim=(0, 2, 3, 5))
    rX = {}
    for k in X.keys():
        rX[k] = X[k].kron_left(1, D=gl)
        rX[k] = X[k].kron_right(1, D=gr)
    return rB, structured_matrix(rX)

def reduce1(B1,X):
    """
    Same operation as reduce but for a rank1 model
    :param B1:
    :param X:
    :return:
    """
    gl = gcd([X[k].otimes_left for k in X.keys()])
    gr = gcd([X[k].otimes_right for k in X.keys()])
    N = B1.shape[0]
    rB1 = B1.reshape((gl, int(N / gl / gr), gr)).sum(dim=(0, 2))
    rX = {}
    for k in X.keys():
        rX[k] = X[k].kron_left(1, D=gl)
        rX[k] = X[k].kron_right(1, D=gr)
    return rB1, structured_matrix(rX)


def kron(a, b):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast.
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = th.Size(th.tensor(a.shape[-2:]) * th.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    return res.reshape(siz0 + siz1)

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

    c = (math.sqrt(math.pi)/2.0/sg).prod().item()
    if mask == None:
        return c*(((th.erf(b.unsqueeze(0) - X.unsqueeze(1)) - th.erf(a.unsqueeze(0) - X.unsqueeze(1)))).prod(2))
    else:
        Bx = b.unsqueeze(0) - X.unsqueeze(1)
        Ax = a.unsqueeze(0) - X.unsqueeze(1)
        Bx[mask[:,None,None]] = th.erf(Bx.masked_select[mask[:,None,None]])
        Ax[mask[:,None,None]] = th.erf(Ax.masked_select[mask[:,None,None]])
        Bx -= Ax
        Bx[~mask[:,None,None]] = 0
        return c*((Bx).prod(2))



###### Factory method for Gaussian PSD Models



def CreateGaussianPsdModel(B, X, gamma, **kwargs):
    """
    Usage
        CreateGaussianPsdModel(B, X, g, x=[0,1,2], y=[3], z=[4,5])

    :param B:
    :param X:
    :param gamma:
    :param kwargs:
    :return:
    """
    strX = create_structured_matrix_from_X(X, **kwargs)
    gg = th.tensor(gamma, dtype=X.dtype, device=X.device).view(1,-1)
    strGamma = create_structured_matrix_from_X(gg, **kwargs)
    return GaussianPsdModel(B, strX, strGamma)



### Gaussian PSD Models



class GaussianPsdModel:

    def __init__(self, B, X, gamma):
        rB, rX = reduce(B, X)
        self.B = rB
        self.X = rX
        self.gamma = gamma
        self.dtype = list(self.X.values())[0].X.dtype
        self.dev = list(self.X.values())[0].X.device
        self.variables = {k: rX[k].d for k in rX.keys()}

    def marginalize(self, **kwargs):
        """
        Usage
        >>> xyz = CreateGaussianPsdModel(B,X, g, x=[0,1,2], y=[3], z=[4,5])
        >>> x = xyz.marginalize(y = [], z = [])

        :param kwargs:
        :return:
        """

        sel = kwargs.keys()

        assert set(sel).issubset(set(self.variables.keys()))

        X = self.X.subset(sel).to_bs_matrix().decompress().X
        gamma = self.gamma.subset(sel).to_bs_matrix().X

        B = gaussKern(X, X, gamma / 2)
        B *= (math.pi / (2 * gamma)).sqrt().prod()
        B *= self.B

        oth = set(self.X.keys()).difference(set(sel))
        if oth == {}:
            return B.sum().sum()
        else:
            return GaussianPsdModel(B, self.X.subset(oth), self.gamma.subset(oth))

    def integrate_hypercube(self, a, b):
        gg = lambda X, gamma: gaussIntegrate(X, gamma, a, b)
        return self.integrate(gg)

    def normalization_constant(self):
        return self.integrate()

    def normalize(self):
        self.B.mul_(1.0/self.normalization_constant())

    def integrate(self, g=None):
        """
        Usage

        >>> model = CreateGaussianPsdModel(B,X, gamma, x=[0,1,2])
        >>> v = model.integrate(g)

        :param g: g is a function that takes X of dimension N times d, gamma of dimension d and
        gives in output a matrix M of dimension N times m (with m >= 1), in particular
        must correspond to the close form of the following integral
        g(X,gamma)[i,:] = integral f(x) exp(-(gamma*(x-X[i,:])**2).sum()) dx
        for some function f that takes a d-dimensional vector in input and gives an m-dimensional vector in output

        :return:
        the output is a vector of dimension m that corresponds to
        model.integrate(g) = integral f(x) model(x) dx
        """


        X = self.X.to_bs_matrix().decompress().X
        gamma = self.gamma.to_bs_matrix().X

        B = gaussKern(X, X, gamma / 2)
        B *= self.B

        if g is None:
            return (B.view(-1,1)).sum(dim=0) * (math.pi/(2.0*gamma)).sqrt().prod()
        else:
            d = X.shape[1]
            G = g( 0.5*(X.view(-1,1,d) + X.view(1,-1,d)).view(-1,d) , 2.0*gamma)

            return (G * B.view(-1,1)).sum(dim=0)

    def mean(self):
        X = self.X.to_bs_matrix().decompress().X
        gamma = self.gamma.to_bs_matrix().X
        B = gaussKern(X, X, gamma / 2)
        B *= self.B
        return X.T @ B.sum(1)

    def cov(self):
        assert False, "cov is not implemented yet"


    def sample(self, N, tol=1e-3):
        '''
        :param N: numbers of i.i.d. samples to take from the model
        :param tol: (default 1e-3) tolerance of the sampling strategy
        :return: N i.i.d. samples distributed according to the model.
        '''

        dtype = self.dtype
        device = self.dev
        x0 = self.mean()
        d = self.dimensions()

        on = th.ones(1, d, device=device, dtype = dtype)
        qq = math.log(self.normalization_constant())
        c = 1
        while True:
            c = 10 * c
            a = x0.t() - c * on
            b = x0.t() + c * on
            v = self.integrate_hypercube(a, b)
            if math.log(v) - qq >= math.log(1.0 - tol):
                break

        steps = math.ceil(d * (math.log(2 * c / tol) / math.log(2)))

        A = a.repeat(N, 1)
        B = b.repeat(N, 1)

        for i in range(0, steps):
            j = i % d
            A1 = A.clone(); A2 = A.clone()
            B1 = B.clone(); B2 = B.clone()
            B1[:, j] = (B[:, j] + A[:, j]) / 2.0
            A2[:, j] = B1[:, j]


            z = th.rand(N, 1, device=A.device, dtype = dtype)*v

            v1 = self.integrate_hypercube(A1, B1).reshape(N,1)

            A = A * (z == v1) + A1 * (z < v1) + A2 * (z > v1)
            B = B * (z == v1) + B1 * (z < v1) + B2 * (z > v1)
            v = v * (z == v1) + v1 * (z < v1) + (v-v1) * (z > v1)

        V = (B - A) * th.rand(N, d, device=A.device) + A

        return V[th.randperm(N), :]


    def __call__(self, *args, **kwargs):
        """
        Usage
        >>> XY = GaussianPsdModel(B, X, gamma, x = [0,1], y = [2,3,4])
        >>> X = XY(y = [1,0,-1])

        X corresponds to the model where we fixed y=[1, 0, -1].

        :param kwargs:
        :return:
        """



        assert len(args) <= 1 and (len(args) == 0) == (len(kwargs) > 0), 'either X contains a list of points or it is empty and some variable assignments are present'

        if len(args) == 1 and th.is_tensor(args[0]):
            Z = self.X.to_bs_matrix().decompress().X
            g = self.gamma.to_bs_matrix().decompress().X
            KXZ = gaussKern(args[0], Z, g)
            return ((KXZ @ self.B) * KXZ).sum(dim=1, keepdim=True)

        if len(kwargs) == 0:
            return self

        assert set(kwargs.keys()).issubset(set(self.variables.keys()))

        Xnew = structured_matrix(kwargs).to_bs_matrix().X.to(dtype = self.dtype, device = self.dev)
        selX = self.X.subset(kwargs.keys()).to_bs_matrix().decompress().X
        selg = self.gamma.subset(kwargs.keys()).to_bs_matrix().decompress().X

        K = gaussKern(Xnew, selX, selg)

        oth = set(self.X.keys()).difference(set(kwargs.keys()))
        if len(oth) == 0:
            return (K.matmul(self.B) * K).sum(dim=1).T
        else:
            return GaussianPsdModel(self.B * K.view(1, -1) * K.view(-1, 1), self.X.subset(oth), self.gamma.subset(oth))


    def __mul__(self, G2):
        # Let A with variables x, y and B with variables y, z
        # then A * B has variables x, y, z, where the y is combined properly

        varX = set(self.X.keys())
        varY = set(G2.X.keys())
        vc = varX.intersection(varY)

        N = self.X.N
        M = G2.X.N
        sZ = {}
        ng = {}

        for k in vc:
            g1 = self.gamma[k].X
            g2 = G2.gamma[k].X
            sZ[k] = self.X[k].kron_right(M) * (g1/(g1+g2)) + G2.X[k].kron_left(N) * (g2/(g1+g2))
            ng[k] = self.gamma[k] + G2.gamma[k]


        for k in varX.difference(varY):
            sZ[k] = self.X[k].kron_right(M)
            ng[k] = self.gamma[k]

        for k in varY.difference(varX):
            sZ[k] = G2.X[k].kron_left(N)
            ng[k] = G2.gamma[k]

        newX = structured_matrix(sZ)
        newgamma = structured_matrix(ng)

        X1 = self.X.subset(vc).to_bs_matrix().X
        X2 = G2.X.subset(vc).to_bs_matrix().X
        g1 = self.gamma.subset(vc).to_bs_matrix().X
        g2 = G2.gamma.subset(vc).to_bs_matrix().X
        K = gaussKern(X1, X2, g1*g2/(g1+g2)).view(-1,1)

        C = kron(self.B, G2.B) * K * K.T

        return GaussianPsdModel(C, newX, newgamma)



    def dimensions(self, **kwargs):
        assert set(kwargs.keys()).issubset(set(self.variables.keys()))

        if len(kwargs.keys()) == 0:
            return sum(self.variables.values())
        else:
            return sum([self.variables[k] for k in kwargs.keys()])

    def rename_vars(self, **kwargs):
        """
        Usage

        >>> g.rename(x = 'y', z = 'x')

        :param kwargs:
        :return:
        """

        assert set(kwargs.keys()).issubset(set(self.variables.keys()))

        self.rX.rename(**kwargs)
        self.B.rename(**kwargs)
        self.gamma.rename(**kwargs)



    def compress(self, M):
        assert False, "compress is not implemented yet"


def CreateGaussianPsdModel1(B, X, gamma,Q, **kwargs):
    """
    Usage
        CreateGaussianPsdModel(B, X, g, x=[0,1,2], y=[3], z=[4,5])

    :param B:
    :param X:
    :param gamma:
    :param kwargs:
    :return:
    """
    strX = create_structured_matrix_from_X(X, **kwargs)
    gg = th.tensor(gamma, dtype=X.dtype, device=X.device).view(1,-1)
    strGamma = create_structured_matrix_from_X(gg, **kwargs)
    return GaussianPsdModel1(B, strX, strGamma,Q=Q)



class GaussianPsdModel1:
    def __init__(self, B1, X, gamma,Q = None):
        rB1, rX = reduce1(B1, X)
        self.B1 = rB1
        self.B = self.B1[:,None]*self.B1[None,:]
        #B1 has size th.tensor([n,])
        self.X = rX
        self.gamma = gamma
        self.dtype = list(self.X.values())[0].X.dtype
        self.dev = list(self.X.values())[0].X.device
        self.variables = {k: rX[k].d for k in rX.keys()}
        self.Q = Q
        self.Lip = None

    def compute_lip(self):
        if self.Lip is None:
            X = self.X.to_bs_matrix().decompress().X

            gamma = self.gamma.to_bs_matrix().X

            K = gaussKern(X,None,gamma)
            alpha = self.B1.view(-1,1)
            self.Lip = (alpha.T@K@alpha).sqrt().item()
        else:
            pass




    def marginalize(self, **kwargs):
        """
        Usage
        >>> xyz = CreateGaussianPsdModel(B,X, g, x=[0,1,2], y=[3], z=[4,5])
        >>> x = xyz.marginalize(y = [], z = [])

        :param kwargs:
        :return:
        TODO
        """

        assert False


    def integrate_hypercube(self, a, b,tol = None):
        gg = lambda X, gamma: gaussIntegrate(X, gamma, a, b)
        return self.integrate(gg,tol = tol)

    def normalization_constant(self):
        return self.integrate()

    def normalize(self):
        self.B.mul_(1.0/self.normalization_constant())

    def integrate(self, g=None,tol = None):
        """
        Usage

        >>> model = CreateGaussianPsdModel(B,X, gamma, x=[0,1,2])
        >>> v = model.integrate(g)

        :param g: g is a function that takes X of dimension N times d, gamma of dimension d and
        gives in output a matrix M of dimension N times m (with m >= 1), in particular
        must correspond to the close form of the following integral
        g(X,gamma)[i,:] = integral f(x) exp(-(gamma*(x-X[i,:])**2).sum()) dx
        for some function f that takes a d-dimensional vector in input and gives an m-dimensional vector in output

        :return:
        the output is a vector of dimension m that corresponds to
        model.integrate(g) = integral f(x) model(x) dx
        """


        X = self.X.to_bs_matrix().decompress().X

        gamma = self.gamma.to_bs_matrix().X

        B = gaussKern(X, X, gamma / 2).view(-1,1)

        if tol :
            mask = (B>tol)


        B *= self.B.view(-1,1)
        if tol:
            B= B.masked_select(mask).view(-1,1)



        if g is None and self.Q is None:
            return B.sum(dim=0) * (math.pi/(2.0*gamma)).sqrt().prod()
        else:
            d = X.size(1)
            Xhalf = 0.5*X
            Xm = (Xhalf.view(-1,1,d) + Xhalf.view(1,-1,d)).view(-1,d)

            if tol:
                X_to_integrate = Xm.masked_select(mask.expand_as(Xm)).view(-1,d)
            else:
                X_to_integrate = Xm

            if g is None:
                #need to compute integral
                a,b = self.Q[0,:].unsqueeze(0),self.Q[1,:].unsqueeze(0)
                g = lambda XX,ggamma: gaussIntegrate(XX,ggamma,a,b)

            G = g(X_to_integrate, 2.0*gamma)


            return (G * B).sum(dim=0)

    def mean(self):
        X = self.X.to_bs_matrix().decompress().X
        gamma = self.gamma.to_bs_matrix().X
        B = gaussKern(X, X, gamma / 2)
        B *= self.B
        return X.T @ B.sum(1)

    def cov(self):
        assert False, "cov is not implemented yet"


    def sample(self, N, tol=1e-3,adative = None,tolInt=None):
        '''
        :param N: numbers of i.i.d. samples to take from the model
        :param tol: (default 1e-3) tolerance of the sampling strategy
        :return: N i.i.d. samples distributed according to the model.
        '''

        dtype = self.dtype
        device = self.dev
        d = self.dimensions()
        qq = math.log(self.normalization_constant())
        if self.Q is None:
            Q = th.ones(2,d)
            Q[0,:] *=-1
            Q *= math.inf

        else:
            Q = self.Q

        a,b = Q[0,:].view(1,-1),Q[1,:].view(1,-1)


        mask = (a == -math.inf)+(b == math.inf)
        if mask.sum().item() > 0:
            tol = tol/2
            x0 = self.mean().view(1,-1)[mask]
            on = th.ones(x0.size(), device=device, dtype=dtype)

            c = 0.1
            while True:
                c*=10
                a[mask] = x0 - c * on
                b[mask] = x0+c*on
                v = self.integrate_hypercube(a, b)
                if math.log(v) - qq >= math.log(1.0 - tol):
                    break
        else:
            v = self.integrate_hypercube(a, b)

        # in the non adaptive case
        Qnorm = th.log(b-a).sum()
        steps = math.ceil((Qnorm + d * math.log(1 / tol) )/ math.log(2))

        A = a.repeat(N, 1)
        B = b.repeat(N, 1)

        steps = th.ceil(th.log((b - a).view(-1) / tol) / math.log(2)).clamp(min=0).type(th.LongTensor)
        direction_array = th.zeros(int(steps.sum().item()), dtype=th.int64)
        j = 0
        for i in range(steps.size(0)):
            s = steps[i].item()
            direction_array[j:j + s] = i
            j += s

        for j in direction_array:
            #j = i % d
            A1 = A.clone(); A2 = A.clone()
            B1 = B.clone(); B2 = B.clone()
            B1[:, j] = (B[:, j] + A[:, j]) / 2.0
            A2[:, j] = B1[:, j]


            z = th.rand(N, 1, device=A.device, dtype = dtype)*v

            v1 = self.integrate_hypercube(A1, B1,tol = tolInt).reshape(N,1)

            A = A * (z == v1) + A1 * (z < v1) + A2 * (z > v1)
            B = B * (z == v1) + B1 * (z < v1) + B2 * (z > v1)
            v = v * (z == v1) + v1 * (z < v1) + (v-v1) * (z > v1)

        V = (B - A) * th.rand(N, d, device=A.device) + A

        return V[th.randperm(N), :]

    def sample_from_hypercube(self, Ntotal, tol=1e-3,adaptive = None,tolInt=None,Q = None,n_jobs = 1,Nmax = None):
        '''
        :param N: numbers of i.i.d. samples to take from the model
        :param tol: (default 1e-3) tolerance of the sampling strategy
        :return: N i.i.d. samples distributed according to the model.
        '''
        if not(adaptive is None):
            self.compute_lip()


        if Q is None:
            Q = self.Q
        a,b = Q[0,:].view(1,-1),Q[1,:].view(1,-1)

        X = self.X.to_bs_matrix().decompress().X
        g = self.gamma.to_bs_matrix().X
        AA = self.B

        BB = gaussKern(X, X, g / 2).view(-1, 1)
        if tolInt:
            mask = (BB > tolInt)
        BB *= AA.view(-1, 1)
        if tolInt:
            BB = BB.masked_select(mask).view(-1, 1)

        d = X.size(1)
        Xhalf = 0.5 * X
        Xm = (Xhalf.view(-1, 1, d) + Xhalf.view(1, -1, d)).view(-1, d)

        if tolInt:
            X_to_integrate = Xm.masked_select(mask.expand_as(Xm)).view(-1, d)
        else:
            X_to_integrate = Xm




        G = gaussIntegrate(X_to_integrate, 2.0 * g, a, b)
        integral = (G * BB).sum(dim=0).item()


        Qnorm = (Q[1, :] - Q[0, :]).prod().item()
        gamma_min = g.min().item()

        if adaptive == 'tv':
            new_tol = tol * integral/Qnorm/self.Lip**2/d/(math.sqrt(8*gamma_min))
            print(f'adaptive side of hypercube computed to have tv distance bounded by {tol} : {new_tol}')
            tol = new_tol
        if adaptive == 'hellinger':
            new_tol = tol * math.sqrt(integral)/math.sqrt(Qnorm)/self.Lip/d/(math.sqrt(2*gamma_min))
            print(f'adaptive side of hypercube computed to have hellinger distance bounded by {tol} : {new_tol}')
            tol = new_tol

        steps = th.ceil(th.log((Q[1, :] - Q[0, :]) / tol) / math.log(2)).clamp(min=0).type(th.LongTensor)
        min_steps = steps.min().item()
        steps -= min_steps

        direction_array = th.zeros(int(steps.sum().item()),dtype = th.int64)
        j = 0
        for i in range(steps.size(0)):
            s = steps[i].item()
            direction_array[j:j + s] = i
            j += s
        direction_array = direction_array[th.randperm(len(direction_array))]
        direction_array = th.cat([th.tensor(list(range(d))*min_steps).type(th.LongTensor),direction_array])
        def small_sample(N):
            print(f'computing {N} samples')
            bins = th.tensor([N],dtype = th.int64)
            A = a
            B = b
            integrals = th.tensor([integral])

            for i0 in direction_array:
                i0 = int(i0)
                if integrals.size(0) < N:
                #if True

                    A_half = A.clone()
                    A_half[:, i0] = 0.5 * (B[:, i0] + A[:, i0])

                    B_half = B.clone()
                    B_half[:,i0] = 0.5*(B[:,i0]+ A[:,i0])


                    G = gaussIntegrate(X_to_integrate, 2.0 * g, A, B_half)
                    integrals_next = (G * BB).sum(dim=0)

                    mask_zeros_1 = (integrals < 1e-14)
                    mask_zeros = (integrals_next < 1e-14)
                    integrals_next[mask_zeros] = 0.

                    p = integrals_next/integrals

                    p[mask_zeros_1] = 0.5


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
                    A_half = A.clone()
                    A_half[:, i0] = 0.5 * (B[:, i0] + A[:, i0])

                    B_half = B.clone()
                    B_half[:, i0] = 0.5 * (B[:, i0] + A[:, i0])

                    G = gaussIntegrate(X_to_integrate, 2.0 * g, A, B_half)
                    integrals_next = (G * BB).sum(dim=0)

                    mask_zeros_1 = (integrals < 1e-14)
                    mask_zeros = (integrals_next < 1e-14)
                    integrals_next[mask_zeros] = 0.

                    p = integrals_next/integrals
                    p[mask_zeros_1] = 0.5

                    #p = integrals_next / integrals
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
            return samples

        if Nmax is None:
            samples = small_sample(Ntotal)

        elif n_jobs == 1:
            q = Ntotal//Nmax
            r = Ntotal%Nmax
            if r == 0:
                q-= 1

            samples = []
            for k in range(q):
                samples.append(small_sample(Nmax))
            samples.append(small_sample(Ntotal - q*Nmax))
            samples = th.cat(samples)
        else:
            q = Ntotal // Nmax
            r = Ntotal % Nmax
            if r == 0:
                q -= 1
            l_q = [Nmax for k in range(q)] + [Ntotal - q * Nmax]

            samples = Parallel(n_jobs=n_jobs)(delayed(small_sample)(n_q) for n_q in l_q)
            samples = th.cat(samples)

        return samples[th.randperm(Ntotal),:]



    def g(self, *args, **kwargs):
        """
        Usage
        >>> XY = GaussianPsdModel(B, X, gamma, x = [0,1], y = [2,3,4])
        >>> X = XY(y = [1,0,-1])

        X corresponds to the model where we fixed y=[1, 0, -1].

        :param kwargs:
        :return:
        """



        assert len(args) <= 1 and (len(args) == 0) == (len(kwargs) > 0), 'either X contains a list of points or it is empty and some variable assignments are present'

        if len(args) == 1 and th.is_tensor(args[0]):
            Z = self.X.to_bs_matrix().decompress().X
            g = self.gamma.to_bs_matrix().decompress().X
            KXZ = gaussKern(args[0], Z, g)
            return (KXZ @ self.B1)

        if len(kwargs) == 0:
            return self

        assert set(kwargs.keys()).issubset(set(self.variables.keys()))

        Xnew = structured_matrix(kwargs).to_bs_matrix().X.to(dtype = self.dtype, device = self.dev)
        selX = self.X.subset(kwargs.keys()).to_bs_matrix().decompress().X
        selg = self.gamma.subset(kwargs.keys()).to_bs_matrix().decompress().X

        K = gaussKern(Xnew, selX, selg)

        oth = set(self.X.keys()).difference(set(kwargs.keys()))
        if len(oth) == 0:
            return K.matmul(self.B1).T
        else:
            return GaussianPsdModel(self.B * K.view(1, -1) * K.view(-1, 1), self.X.subset(oth), self.gamma.subset(oth))






    def __call__(self, *args, **kwargs):
        """
        Usage
        >>> XY = GaussianPsdModel(B, X, gamma, x = [0,1], y = [2,3,4])
        >>> X = XY(y = [1,0,-1])

        X corresponds to the model where we fixed y=[1, 0, -1].

        :param kwargs:
        :return:
        """



        assert len(args) <= 1 and (len(args) == 0) == (len(kwargs) > 0), 'either X contains a list of points or it is empty and some variable assignments are present'

        if len(args) == 1 and th.is_tensor(args[0]):
            Z = self.X.to_bs_matrix().decompress().X
            g = self.gamma.to_bs_matrix().decompress().X
            KXZ = gaussKern(args[0], Z, g)
            return ((KXZ @ self.B) * KXZ).sum(dim=1, keepdim=True)

        if len(kwargs) == 0:
            return self

        assert set(kwargs.keys()).issubset(set(self.variables.keys()))

        Xnew = structured_matrix(kwargs).to_bs_matrix().X.to(dtype = self.dtype, device = self.dev)
        selX = self.X.subset(kwargs.keys()).to_bs_matrix().decompress().X
        selg = self.gamma.subset(kwargs.keys()).to_bs_matrix().decompress().X

        K = gaussKern(Xnew, selX, selg)

        oth = set(self.X.keys()).difference(set(kwargs.keys()))
        if len(oth) == 0:
            return (K.matmul(self.B) * K).sum(dim=1).T
        else:
            return GaussianPsdModel(self.B * K.view(1, -1) * K.view(-1, 1), self.X.subset(oth), self.gamma.subset(oth))


    def __mul__(self, G2):
        # Let A with variables x, y and B with variables y, z
        # then A * B has variables x, y, z, where the y is combined properly
        # have to do two cases : rank 1 psd or full psd

        varX = set(self.X.keys())
        varY = set(G2.X.keys())
        vc = varX.intersection(varY)

        N = self.X.N
        M = G2.X.N
        sZ = {}
        ng = {}

        for k in vc:
            g1 = self.gamma[k].X
            g2 = G2.gamma[k].X
            sZ[k] = self.X[k].kron_right(M) * (g1/(g1+g2)) + G2.X[k].kron_left(N) * (g2/(g1+g2))
            ng[k] = self.gamma[k] + G2.gamma[k]


        for k in varX.difference(varY):
            sZ[k] = self.X[k].kron_right(M)
            ng[k] = self.gamma[k]

        for k in varY.difference(varX):
            sZ[k] = G2.X[k].kron_left(N)
            ng[k] = G2.gamma[k]

        newX = structured_matrix(sZ)
        newgamma = structured_matrix(ng)

        X1 = self.X.subset(vc).to_bs_matrix().X
        X2 = G2.X.subset(vc).to_bs_matrix().X
        g1 = self.gamma.subset(vc).to_bs_matrix().X
        g2 = G2.gamma.subset(vc).to_bs_matrix().X
        K = gaussKern(X1, X2, g1*g2/(g1+g2)).view(-1,1)

        C = kron(self.B, G2.B) * K * K.T

        return GaussianPsdModel(C, newX, newgamma)



    def dimensions(self, **kwargs):
        assert set(kwargs.keys()).issubset(set(self.variables.keys()))

        if len(kwargs.keys()) == 0:
            return sum(self.variables.values())
        else:
            return sum([self.variables[k] for k in kwargs.keys()])

    def rename_vars(self, **kwargs):
        """
        Usage

        >>> g.rename(x = 'y', z = 'x')

        :param kwargs:
        :return:
        """

        assert set(kwargs.keys()).issubset(set(self.variables.keys()))

        self.rX.rename(**kwargs)
        self.B.rename(**kwargs)
        self.gamma.rename(**kwargs)



    def compress(self, M):
        assert False, "compress is not implemented yet"


def add_reg(M,reg):
    M+= reg*M.trace()*th.eye(M.size(0))
    pass

cm = plt.cm.get_cmap('RdYlBu_r')

def scatter_color(fig, ax0, vmax, Xtest, ptest):
    sc = ax0.scatter(Xtest[:, 0], Xtest[:, 1], c=ptest, vmin=0, vmax=vmax, cmap=cm)
    fig.colorbar(sc, ax=ax0)


def my_cholesky(M,eps = 1e-12):
    if eps !=0:
        add_reg(M,eps)
    return M.cholesky().T


def prepareHellinger(X_train,q_train,gamma,mu_train = None,Ny = None,tol = 1e-14,Q = None,m_compression =None,empirical = False,keep_data=True):
    n = X_train.size(0)
    d = X_train.size(1)

    device = q_train.device
    dtype = q_train.dtype

    if th.is_tensor(gamma) == False:
        new_gamma = gamma*th.ones(1,d,device = device, dtype = dtype)
        gamma = new_gamma
    if th.is_tensor(Ny):
        m = Ny.size(0)
    else:
        if (Ny is None) or Ny >= n:
            Ny = X_train
            m = X_train.size(0)
        else:
            m = Ny
            Ny = X_train[np.random.choice(n,size = m,replace = False),:]
    if mu_train is None:
        mu_train = th.ones(q_train.size(),device = device,dtype = dtype)

    if m_compression:
        Knm = gaussKern(X_train, Ny, gamma)
        Knm /= mu_train.sqrt()[:, None]
        b = q_train / mu_train.sqrt()
        index, _ = chase_projection(Knm, b, m_compression)
        m = m_compression
        Ny = Ny[index,:]

    Kmm = gaussKern(Ny, None, gamma)
    T = my_cholesky(Kmm)


    if empirical == False:
        Imm = gaussKern(Ny,None,0.5*gamma)
        if tol:
            mask = (Imm > tol).view(-1,1)

        if Q is None:
            Imm *= (math.pi / (2.0 * gamma)).sqrt().prod()
        else:
            Nym = 0.5 * (Ny.view(-1, 1, d) + Ny.view(1, -1, d)).view(-1, d)
            if tol:
                NyI = Nym.masked_select(mask.expand_as(Nym)).view(-1, d)
            else:
                NyI = Nym


            a, b = Q[0, :].unsqueeze(0),Q[1, :].unsqueeze(0)
            G = gaussIntegrate(NyI, 2.0*gamma, a, b)
            if tol:
                Imm[mask.view(m,m)] *=G.view(-1)
            else:
                Imm *= G.view(m,m)
        triI = my_cholesky(Imm)
        triI = ((triI.T).triangular_solve(T, upper=True, transpose=True)[0]).T


    Kmn = gaussKern(Ny, X_train,gamma)
    Tmn = Kmn.triangular_solve(T, upper=True, transpose=True)[0]  / math.sqrt(n)

    if empirical :
        triI = (Tmn/(mu_train[None,:].sqrt())).T
    gradient = Tmn @ ((q_train/mu_train).view(n, 1) ) / math.sqrt(n)



    dico_train = {'Ny': Ny,'triCov': triI,'triK' : T,'grad':gradient,'gamma':gamma}

    if keep_data:
        dico_train['X'] = X_train
        dico_train['q_train'] = q_train
        dico_train['mu_train'] = mu_train
        dico_train['Q'] = Q

    return dico_train

def findSolutionHellinger(dico_train,la,X_test= None, q_test = None,\
                          mu_test=None, plot=False, plot_title='',keep_data = False):


    Ny,triI,T,gradient,gamma = dico_train['Ny'],dico_train['triCov'],dico_train['triK'],dico_train['grad'],dico_train['gamma']

    device = T.device
    dtype = T.dtype

    if X_test is None:
        n_test = None
    else:
        n_test = X_test.size(0)
        Kmn_test = gaussKern(Ny, X_test, gamma)
        Tmn_test = Kmn_test.triangular_solve(T, upper=True, transpose=True)[0]
        if mu_test is None:
            mu_test = th.ones(n_test,device = device,dtype = dtype)



    n,m = gradient.size(0),T.size(0)


    Cov = triI.T @triI
    try:
        la_list =[l for l in la]
    except:
        la_list =[la]


    dico_test = {'score' : {},'lambda_list': la_list,'score_best': math.inf}

    for l in la_list:
        L = (Cov + l*th.eye(m,dtype = Cov.dtype,device = Cov.device)).cholesky()
        alpha = gradient.cholesky_solve(L)
        if X_test is None:
            dico_test['alpha_best'] = alpha.clone()
            dico_test['lambda_best'] = l
            break

        q_test_approx = (Tmn_test.T @ alpha).view(n_test)
        score = ((q_test - q_test_approx).pow(2)/mu_test).sum() / n_test

        dico_test['score'][str(l)] = score

        if score < dico_test['score_best']:
            dico_test['score_best'] = score.item()
            dico_test['lambda_best'] = l
            dico_test['alpha_best'] = alpha.clone()
            dico_test['q_best'] = q_test_approx.clone()


    if plot and not(X_test is None):
        fig, ax = plt.subplots(1)
        ax.semilogx(la_list, dico_test['score_list'])
        ax.invert_xaxis()
        ax.set_title('score')
        ax.set_xlabel('lambda')
        fig.suptitle(plot_title)
        plt.show(block=False)

    elif plot:
        print("No test, so no plot, only trained")

    dico_test['alpha_best'] = dico_test['alpha_best'].triangular_solve(T,upper = True,transpose=False)[0]
    dico_train['alpha'] = dico_test['alpha_best'].clone()
    dico_train['la'] = dico_test['lambda_best']
    if keep_data:
        dico_test['X_test'] = X_test
        dico_test['mu_test'] = mu_test
        dico_test['p_test'] = q_test

    return dico_train,dico_test



def step(G,mask,index):
    #in place modification of the gram matrix
    # selection of largest index
    index_mask_1 = index.masked_select(mask)
    values,indices = G[0,index_mask_1].max(0)
    try:
        i0 = index_mask_1[indices[0]]
    except:
        i0 = index_mask_1[indices.item()]
    mask[i0] = False
    C0 = G[:,i0][:,None]
    G -= C0@C0.T
    index_masked = index.masked_select(mask)
    norms = th.sqrt(G[index_masked,index_masked])
    norms[norms == 0] = 1
    G[:,index_masked] /= norms
    G[index_masked,:] /= norms[:,None]
    return G,mask,index
def chase_projection(A,b,m):
    n = A.size(1)
    d  = A.size(0)
    G= th.cat([b[:,None],A],dim = 1)
    G = G.T @ G
    d = th.sqrt(G.diag())
    G[:,1:] /= d[1:]
    G[1:,] /= d[1:, None]
    mask = th.tensor([False] + [True]*n)
    index = th.LongTensor(list(range(n+1)))
    score = [G[0,0].item()]
    for k in range(m):
        G,mask,index = step(G,mask,index)
        score.append(G[0,0].item())

    mask[0] = True
    selected = index.masked_select(mask == False)-1
    return selected,score




def findBestHellinger(X_train,q_train,gamma,la,Ny = None,X_test=None,q_test= None,Q =None,\
                             m_compression = None,mu_train = None,mu_test =None,tol = 1e-14,\
                      empirical = False,keep_data= True,dir = None,save = None,retrain = False):

    try:
        gamma_list =[g for g in gamma]
    except:
        gamma_list = [gamma]

    dico_test = {'score': {}, 'gamma_list': gamma_list, 'score_best': math.inf}

    if keep_data:
        dico_test['X_test'] = X_test
        dico_test['mu_test'] = mu_test
        dico_test['p_test'] = q_test
    for g in gamma_list:

        dico_train_gamma = prepareHellinger(X_train, q_train, g, mu_train=mu_train,
                                                       Ny=Ny,tol=tol, Q=Q,m_compression=m_compression,empirical = empirical,keep_data = keep_data)

        dico_train_gamma,dico_test_gamma = findSolutionHellinger(dico_train_gamma, la, X_test, q_test=q_test,  mu_test=mu_test ,keep_data = False)

        if X_test is None:
            dico_test['alpha_best'] = dico_test_gamma['alpha_best'].clone()
            dico_test['gamma_best'] = g
            dico_test['lambda_best'] = dico_test_gamma['lambda_best']
            break

        dico_test['score'][str(g)] = dico_test_gamma['score']
        score = dico_test_gamma['score_best']


        if score < dico_test['score_best']:
            dico_test['score_best'] = dico_test_gamma['score_best']
            dico_test['lambda_best'] = dico_test_gamma['lambda_best']
            dico_test['gamma_best'] = g
            dico_test['alpha_best'] = dico_test_gamma['alpha_best'].clone()
            dico_test['q_best'] = dico_test_gamma['q_best'].clone()
            dico_train = dico_train_gamma.copy()




    gamma_ref,lambda_ref,score_ref = dico_test['gamma_best'],dico_test['lambda_best'],dico_test['score_best']
    print(f'Final score for gamma {gamma_ref}, lambda {lambda_ref} : {score_ref}')

    if retrain:
        Xtot = th.cat([X_train,X_test])
        qtot = th.cat([q_train,q_test])
        mutot = th.cat([mu_train,mu_test])
        Nyfinal = dico_train['Ny']
        dico_train = prepareHellinger(Xtot, qtot, gamma_ref, mu_train=mutot,
                                            Ny=Ny, tol=tol, Q=Q, m_compression=m_compression, empirical=empirical,
                                            keep_data=keep_data)

        dico_train,_ = findSolutionHellinger(dico_train, lambda_ref, None, q_test=None,
                                                                  mu_test=None, keep_data=False)

    if not(save is None):
        if dir is None:
            dir = ''
        else:
            dir = f'{dir}/'

        with open(f'{dir}test_{save}.pickle', 'wb') as handle:
            pickle.dump(dico_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{dir}train_{save}.pickle', 'wb') as handle:
            pickle.dump(dico_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dico_train,dico_test


def find_Ny_from_sqrt(X_train,q_train,gamma,la,NyIntermediate,X_test=None,q_test= None,Q =None,\
                             mu_train = None,mu_test =None,tol = 1e-14,\
                      empirical = False,keep_data= True,dir = None,save = None,retrain = False,Nmax = None,n_jobs = 1):
    dico_train,_ = findBestHellinger(X_train, q_train.abs().sqrt(), gamma, la, Ny=NyIntermediate, X_test=X_test, q_test=q_test.abs().sqrt(),
                                            Q=Q,  mu_train=mu_train, mu_test=mu_test,tol = tol, \
                      empirical=empirical, keep_data=False, dir=dir, save=save, retrain=True)

    alpha, X, g = dico_train['alpha'], dico_train['Ny'], dico_train['gamma']
    d = X_train.size(1)
    g = g * th.ones(1, d)
    p = CreateGaussianPsdModel1(alpha, X, g, Q, x=list(range(d)))
    X_samples = p.sample_from_hypercube(NyIntermediate,tol  = 1e-2,tolInt = 1e-12,Nmax = Nmax,n_jobs = n_jobs)
    return X_samples

###############################
# MLE Estimation of Gaussian PSD model, from iid samples
###############################













def mle_from_samples(X = None, Ny = None, la = None, gamma = None, **kwargs):

    N = X.shape[0]
    d = X.shape[1]
    dtype = X.dtype
    device = X.device


    mm = X.mean(0).reshape(-1,1)
    cov = (X.T @ X)/ N - mm @ mm.T
    if th.is_tensor(Ny):
        M = Ny.shape[0]
    else:
        M = Ny
        s, U = th.linalg.eigh(cov)
        Ny =  mm.T + th.randn(M, d, dtype=dtype, device=device) @ th.diag(s.sqrt()) @ U.T

    if gamma is None:
        gamma = th.linalg.norm(cov, p = 2) * N**(-1/(1+d))
    elif not th.is_tensor(gamma):
        gamma = gamma*th.ones(1,d, dtype=dtype, device=device)

    gamma = gamma.reshape(1,d)

    vecI = th.eye(M, M, dtype = dtype, device = device).reshape(-1,1)
    vecKNy = th.eye(M, M, dtype = dtype, device = device).reshape(-1,1)
    LNy = th.linalg.cholesky(gaussKern(Ny, Ny, gamma) + 1e-13*th.eye(M, M, dtype = dtype, device = device))
    KXNyT, _ = th.triangular_solve(gaussKern(X, Ny, gamma).T, LNy, upper=False, transpose = False)
    KXNy = KXNyT.T
    cgamma = (math.pi/(2.0*gamma)).sqrt().prod()
    KS = cgamma * gaussKern(Ny, Ny, gamma / 2.0)
    Sigma0, _ = th.triangular_solve((KS + KS.T) / 2.0, LNy, upper=False, transpose = False)
    Sigma1, _ = th.triangular_solve(Sigma0.T, LNy, upper=False, transpose = False)
    Sigma = (Sigma1 + Sigma1.T)/2.0
    ss = th.tensor(Sigma, dtype=dtype, device=device).reshape(-1,1)


    def evalPSDmodel(x):
        A = x.reshape(M,M)
        return ((KXNy @ A) * KXNy).sum(dim=1, keepdim=True)

    def formCorrMatr(u):
        return (KXNy.T @ (u * KXNy)).reshape(-1,1)

    def evalPSDmodelZ(x):
        A = x.reshape(M,M)
        return th.vstack([((KXNy @ A) * KXNy).sum(dim=1, keepdim=True), ss.T @ x])

    def formCorrMatrZ(u):
        return (KXNy.T @ (u[0:-1,:] * KXNy)).reshape(-1,1) + u[-1:,:]*ss

    x0 = th.eye(M,M, dtype=dtype, device=device)
    x0 = x0 / (x0.reshape(-1,1).T @ ss)
    x0 = th.linalg.cholesky(x0)


    def funF(L, tau, level = 3):
        A = L @ L.T


        #correct L
        y = A.reshape(-1,1)
        y = y - ss * (((y.T @ ss) - 1)/(ss.T @ ss))

        L = th.linalg.cholesky(y.reshape(M,M))


        def B(x):
            X = x.reshape(M,M)
            X = (X + X.T) / 2.0
            return (A.T @ X @ A).reshape(-1,1)

        def W(x):
            X = x.reshape(M,M)
            X = (X + X.T) / 2.0
            return (L.T @ X @ L).reshape(-1,1)

        def Wt(x):
            X = x.reshape(M,M)
            X = (X + X.T) / 2.0
            return (L @ X @ L.T).reshape(-1,1)

        def iW(x):
            X = x.reshape(M,M)
            X = (X + X.T) / 2.0
            return th.triangular_solve(
                th.triangular_solve(X, L, upper = False, transpose = True).solution.T,
                        L, upper = False, transpose = True).reshape(-1,1)

        def iWt(x):
            X = x.reshape(M,M)
            X = (X + X.T) / 2.0
            return th.triangular_solve(
                th.triangular_solve(X, L, upper = False).solution.T,
                        L, upper = False).reshape(-1,1)

        Z = KXNy @ L
        r = (Z * Z).sum(dim=1, keepdim=True)

        data = dict()

        #############

        vcstr = ss.T @ A.reshape(-1,1) - 1
        c_err = 1e-3
        lbv = c_err - vcstr**2
        funv = tau * (-r.log().sum() + la*vecKNy.T @ A.reshape(-1,1) - 0.5*lbv.log()) - 2*L.diag().log().sum() + vecI.T @ A.reshape(-1,1)

        data['f'] = funv
        data['cnstr_viol'] = abs(vcstr)
        data['L'] = L

        if level == 0:
            return data

        #############


        w = tau * (-formCorrMatr(1.0/r) + la*vecKNy - ss*vcstr/lbv)  + vecI
        Wz = W(w) - th.eye(M, M, dtype = dtype, device = device).reshape(-1,1)

        data['WDf'] = Wz
        data['ext_WDf'] = th.vstack([Wz, th.zeros(1, 1, dtype = dtype, device = device)])
        if level == 1:
            return data

        ##################

        zz = evalPSDmodel(B(ss))
        cc = lbv**2/(c_err + vcstr**2)
        V = th.vstack([r**2/tau, cc/tau]).reshape(-1).diag() + th.vstack([th.hstack([(Z @ Z.T)**2, zz]), th.hstack([zz.T, ss.T @ B(ss)])])
        V = (V + V.T)/2.0

        F = th.linalg.cholesky(V + 1e-13*V.norm('fro')*th.eye(N+1, N+1, dtype = dtype, device = device))

        iFBz, _ = th.triangular_solve(evalPSDmodelZ(B(w) - A.reshape(-1,1)), F, upper = False, transpose = False)

        Ws = W(ss)
        iFBs, _ = th.triangular_solve(evalPSDmodelZ(B(ss)), F, upper = False, transpose = False)


        num = Ws.T @ Wz - iFBs.T @ iFBz
        den = Ws.T @ Ws - iFBs.T @ iFBs
        newt_dec2 = Wz.T @ Wz - iFBz.T @ iFBz - (num / den)*num

        # if newt_dec2 < -1e-10:
        #     print('negative newt_dec2 =', newt_dec2)
        #     raise Exception


        iFt_iFBz, _ = th.triangular_solve(iFBz, F, upper = False, transpose = True)
        isBinvHz = Wz - W(formCorrMatrZ(iFt_iFBz))

        iFt_iFBs, _ = th.triangular_solve(iFBs, F, upper = False, transpose = True)
        isBinvHs = Ws - W(formCorrMatrZ(iFt_iFBs))

        Q = isBinvHz - (num/den)*isBinvHs

        data['iWtDelta'] = Q.reshape(M,M)
        data['ext_iWtDelta'] = th.vstack([Q, num/den])
        data['newt_dec'] = newt_dec2.abs().sqrt()

        if level == 2:
            return data

        #########################

        def ext_WtHW(u):
            x, z = u[0:-1].reshape(-1,1), u[-1:].reshape(-1,1)
            Ws = W(ss)
            return th.vstack([
                tau * W(formCorrMatr(evalPSDmodel(Wt(x))/r**2)) + ((Ws.T @ x)/cc)*Ws + x + z*Ws,
                Ws.T @ x
            ])

        def invT(x):
            return x - W(formCorrMatrZ(th.cholesky_solve(evalPSDmodelZ(Wt(x)), F, upper = False)))

        def inv_ext_WtHW(u):
            x, z = u[0:-1].reshape(-1,1), u[-1:].reshape(-1,1)
            iTs = invT(ss)
            y = ((x.T @ iTs) - z)/(ss.T @ iTs)
            return th.vstack([invT(x) - y*iTs, y])


        data['ext_WtHW'] = ext_WtHW
        data['inv_ext_WtHW'] = inv_ext_WtHW

        return data


    #
    #
    # def simpleF(L):
    #     A = L @ L.T
    #     Z = KXNy @ L
    #     r = (Z * Z).sum(dim=1, keepdim=True)
    #
    #     f = -r.log().sum() + la*vecKNy.T @ A.reshape(-1,1)  + 1.0/la**2*0.5*(ss.T @ A.reshape(-1,1) - 1)**2 #- la**2*2*L.diag().log().sum()
    #     dx = -formCorrMatr(1.0/r) + la*vecKNy + 1.0/la**2*ss*(ss.T @ A.reshape(-1,1) - 1) #- la**2*th.linalg.inv(A).reshape(-1,1)
    #
    #     return f, dx


    data = constrained_newton_method(funF, x0, min_err=1e-3, iter_max=10000)
    L = data['L']


    # eyeI = th.eye(M,M, dtype=dtype, device=device)
    #
    # L = x0
    # for i in range(0,1000):
    #     f, dx = simpleF(L)
    #     #dxs = (dx - ss * (dx.T @ ss)/(ss.T @ ss)).reshape(M, M)
    #     print(f)
    #
    #     D = L.T @ dx.reshape(M,M) @ L
    #     D = (D + D.T) / 2.0
    #
    #     # s, U = th.linalg.eigh(D)
    #     # s, I = th.sort(s)
    #     # irt = range(0,5)
    #     # D = U[:,I[irt]] @ th.diag(s[I[irt]]) @ U[:,I[irt]].T
    #
    #     normD = th.linalg.matrix_norm(D, ord = 2)
    #     max_eigD = th.linalg.matrix_norm(D + normD*eyeI, ord = 2) - normD
    #     min_eigD = th.linalg.matrix_norm(-D + normD*eyeI, ord = 2) - normD
    #
    #     #print('min_eig:', min_eigD)
    #     tmax = float(0.99/min_eigD)
    #
    #
    #     dd = dict()
    #     def minf(t):
    #         S = L @ th.linalg.solve(eyeI + t*D, L.T)
    #         S = (S + S.T)/2
    #         Lnew = th.linalg.cholesky(S)
    #         f, dx = simpleF(Lnew)
    #         dd[t] = (f, Lnew, dx)
    #         return float(f)
    #
    #     fx = math.inf
    #     cc = 1.0
    #     while fx > f:
    #         res = sco.minimize_scalar(minf, bounds=[0.0,abs(cc*tmax)],method='bounded',options={'disp':0, 'xatol':1e-3*cc*tmax})
    #         tt = res.x
    #         fx = res.fun
    #         cc *= 0.8
    #     f, L, dx = dd[tt]


    L, _ = th.triangular_solve(L, LNy, upper = False, transpose = True)

    A = L @ L.T

    p = CreateGaussianPsdModel(A, Ny, gamma, **kwargs)
    p.normalize()

    return p


def mle_from_samples_slow(X = None, Ny = None, la = None, gamma = None, **kwargs):

    N = X.shape[0]
    d = X.shape[1]
    dtype = X.dtype
    device = X.device


    mm = X.mean(0).reshape(-1,1)
    cov = (X.T @ X)/ N - mm @ mm.T
    if th.is_tensor(Ny):
        M = Ny.shape[0]
    else:
        M = Ny
        s, U = th.linalg.eigh(cov)
        Ny =  mm.T + th.randn(M, d, dtype=dtype, device=device) @ th.diag(s.sqrt()) @ U.T

    if gamma is None:
        gamma = th.linalg.norm(cov, p = 2) * N**(-1/(1+d))
    elif not th.is_tensor(gamma):
        gamma = gamma*th.ones(1,d, dtype=dtype, device=device)

    gamma = gamma.reshape(1,d)

    vecI = th.eye(M, M, dtype = dtype, device = device).reshape(-1,1)
    vecKNy = th.eye(M, M, dtype = dtype, device = device).reshape(-1,1)
    LNy = th.linalg.cholesky(gaussKern(Ny, Ny, gamma) + 1e-13*th.eye(M, M, dtype = dtype, device = device))
    KXNyT, _ = th.triangular_solve(gaussKern(X, Ny, gamma).T, LNy, upper=False, transpose = False)
    KXNy = KXNyT.T
    cgamma = (math.pi/(2.0*gamma)).sqrt().prod()
    KS = cgamma * gaussKern(Ny, Ny, gamma / 2.0)
    Sigma0, _ = th.triangular_solve((KS + KS.T) / 2.0, LNy, upper=False, transpose = False)
    Sigma1, _ = th.triangular_solve(Sigma0.T, LNy, upper=False, transpose = False)
    Sigma = (Sigma1 + Sigma1.T)/2.0
    ss = th.tensor(Sigma, dtype=dtype, device=device).reshape(-1,1)

    x0 = th.eye(M,M, dtype=dtype, device=device)
    x0 = x0 / (x0.reshape(-1,1).T @ ss)
    x0 = th.linalg.cholesky(x0)



    def simpleF(L, tau):
        A = L @ L.T
        Z = KXNy @ L
        r = (Z * Z).sum(dim=1, keepdim=True)

        f = -r.log().sum() + 0.5*la*A.norm('fro')**2  + 1.0/la**2*0.5*(ss.T @ A.reshape(-1,1) - 1)**2 + 0.5*tau*A.norm('fro')**2 - tau*2*L.diag().log().sum()
        dx = -(KXNy.T @ ((1.0/r) * KXNy)).reshape(-1,1) + la*A.reshape(-1,1) + 1.0/la**2*ss*(ss.T @ A.reshape(-1,1) - 1) + tau*A.reshape(-1,1) - tau*th.linalg.inv(A).reshape(-1,1)

        return f, dx


    eyeI = th.eye(M,M, dtype=dtype, device=device)

    L = x0
    ttD = math.inf
    for i in range(0,10000):
        tau = 1.0/math.sqrt(1.0 + i)

        f, dx = simpleF(L, tau)

        D = L.T @ dx.reshape(M,M) @ L
        D = (D + D.T)/2.0

        normD = th.linalg.matrix_norm(D, ord = 2)
        min_eigD = th.linalg.matrix_norm(-D + normD*eyeI, ord = 2) - normD

        tmax_D = min(10*ttD, float(1.0/min_eigD))

        ddD = dict()
        def val_D(t):
            S = L @ th.linalg.solve(eyeI + t*D, L.T)
            S = (S + S.T)/2
            Lnew = th.linalg.cholesky(S)
            f, dx = simpleF(Lnew, tau)
            ddD[t] = (f, Lnew, dx)
            return float(f)



        # s, U = th.linalg.eigh(D)
        # s, I = th.sort(s)
        # if abs(s[I[0]]) > abs(s[I[-1]]):
        #     u = U[:,I[0]].reshape(-1,1)
        #     s = -1
        # else:
        #     u = U[:,I[-1]].reshape(-1,1)
        #     s = 1
        #
        # if s == 1:
        #     tmax_D = 1
        # else:
        #     tmax_D = float(2*L.norm('fro'))
        #
        # A = L @ L.T
        # ddD = dict()
        # def val_D(t):
        #     v = L @ u
        #     S = A - s*t*v @ v.T
        #     Lnew = th.linalg.cholesky(S)
        #     f, dx = simpleF(Lnew)
        #     ddD[t] = (f, Lnew, dx)
        #     return float(f)


        fx = math.inf
        cc = 1.0
        while fx > f:
            res = sco.minimize_scalar(val_D, bounds=[0.0,abs(cc*tmax_D)],method='bounded',options={'disp':0, 'xatol':1e-3*cc*tmax_D})
            ttD = res.x
            fx = res.fun
            cc *= 0.8

        fnew, L, dx = ddD[ttD]

        # if s == -1:
        #     print('it={:5d}\tf={:.6E}\tt={:.6E} inverse'.format(i,float(f),float(ttD)))
        # else:
        #     print('it={:5d}\tf={:.6E}\tt={:.6E} direct'.format(i,float(f),float(ttD)))

        print('it={:5d}\tf={:.6E}\tt={:.6E}\ttmax={:.6E} inverse'.format(i,float(f),float(ttD), float(tmax_D)))



    L, _ = th.triangular_solve(L, LNy, upper = False, transpose = True)

    A = L @ L.T

    p = CreateGaussianPsdModel(A, Ny, gamma, **kwargs)
    p.normalize()

    return p




def mle_from_samples_acc(X = None, Ny = None, la = None, gamma = None, **kwargs):

    N = X.shape[0]
    d = X.shape[1]
    dtype = X.dtype
    device = X.device


    mm = X.mean(0).reshape(-1,1)
    cov = (X.T @ X)/ N - mm @ mm.T
    if th.is_tensor(Ny):
        M = Ny.shape[0]
    else:
        M = Ny
        s, U = th.linalg.eigh(cov)
        Ny =  mm.T + th.randn(M, d, dtype=dtype, device=device) @ th.diag(s.sqrt()) @ U.T

    if gamma is None:
        gamma = th.linalg.norm(cov, p = 2) * N**(-1/(1+d))
    elif not th.is_tensor(gamma):
        gamma = gamma*th.ones(1,d, dtype=dtype, device=device)

    gamma = gamma.reshape(1,d)

    vecI = th.eye(M, M, dtype = dtype, device = device).reshape(-1,1)
    vecKNy = th.eye(M, M, dtype = dtype, device = device).reshape(-1,1)
    LNy = th.linalg.cholesky(gaussKern(Ny, Ny, gamma) + 1e-13*th.eye(M, M, dtype = dtype, device = device))
    KXNyT, _ = th.triangular_solve(gaussKern(X, Ny, gamma).T, LNy, upper=False, transpose = False)
    KXNy = KXNyT.T
    cgamma = (math.pi/(2.0*gamma)).sqrt().prod()
    KS = cgamma * gaussKern(Ny, Ny, gamma / 2.0)
    Sigma0, _ = th.triangular_solve((KS + KS.T) / 2.0, LNy, upper=False, transpose = False)
    Sigma1, _ = th.triangular_solve(Sigma0.T, LNy, upper=False, transpose = False)
    Sigma = (Sigma1 + Sigma1.T)/2.0
    ss = th.tensor(Sigma, dtype=dtype, device=device).reshape(-1,1)

    x0 = th.eye(M,M, dtype=dtype, device=device)
    x0 = x0 / (x0.reshape(-1,1).T @ ss)
    x0 = th.linalg.cholesky(x0)



    def simpleF(L, tau):
        A = L @ L.T
        Z = KXNy @ L
        r = (Z * Z).sum(dim=1, keepdim=True)

        f = -r.log().sum() + 0.5*la*A.norm('fro')**2  + 1.0/la**2*0.5*(ss.T @ A.reshape(-1,1) - 1)**2 + 0.5*tau*A.norm('fro')**2 - tau*2*L.diag().log().sum()
        dx = -(KXNy.T @ ((1.0/r) * KXNy)).reshape(-1,1) + la*A.reshape(-1,1) + 1.0/la**2*ss*(ss.T @ A.reshape(-1,1) - 1) + tau*A.reshape(-1,1) - tau*th.linalg.inv(A).reshape(-1,1)

        return f, dx


    eyeI = th.eye(M,M, dtype=dtype, device=device)

    L = x0
    ttD = math.inf
    for i in range(0,10000):
        tau = 1.0/math.sqrt(1.0 + i)

        f, dx = simpleF(L, tau)

        D = L.T @ dx.reshape(M,M) @ L
        D = (D + D.T)/2.0

        normD = th.linalg.matrix_norm(D, ord = 2)
        min_eigD = th.linalg.matrix_norm(-D + normD*eyeI, ord = 2) - normD

        tmax_D = min(10*ttD, float(1.0/min_eigD))

        ddD = dict()
        def val_D(t):
            S = L @ th.linalg.solve(eyeI + t*D, L.T)
            S = (S + S.T)/2
            Lnew = th.linalg.cholesky(S)
            f, dx = simpleF(Lnew, tau)
            ddD[t] = (f, Lnew, dx)
            return float(f)



        # s, U = th.linalg.eigh(D)
        # s, I = th.sort(s)
        # if abs(s[I[0]]) > abs(s[I[-1]]):
        #     u = U[:,I[0]].reshape(-1,1)
        #     s = -1
        # else:
        #     u = U[:,I[-1]].reshape(-1,1)
        #     s = 1
        #
        # if s == 1:
        #     tmax_D = 1
        # else:
        #     tmax_D = float(2*L.norm('fro'))
        #
        # A = L @ L.T
        # ddD = dict()
        # def val_D(t):
        #     v = L @ u
        #     S = A - s*t*v @ v.T
        #     Lnew = th.linalg.cholesky(S)
        #     f, dx = simpleF(Lnew)
        #     ddD[t] = (f, Lnew, dx)
        #     return float(f)


        fx = math.inf
        cc = 1.0
        while fx > f:
            res = sco.minimize_scalar(val_D, bounds=[0.0,abs(cc*tmax_D)],method='bounded',options={'disp':0, 'xatol':1e-3*cc*tmax_D})
            ttD = res.x
            fx = res.fun
            cc *= 0.8

        fnew, L, dx = ddD[ttD]

        # if s == -1:
        #     print('it={:5d}\tf={:.6E}\tt={:.6E} inverse'.format(i,float(f),float(ttD)))
        # else:
        #     print('it={:5d}\tf={:.6E}\tt={:.6E} direct'.format(i,float(f),float(ttD)))

        print('it={:5d}\tf={:.6E}\tt={:.6E}\ttmax={:.6E} inverse'.format(i,float(f),float(ttD), float(tmax_D)))



    L, _ = th.triangular_solve(L, LNy, upper = False, transpose = True)

    A = L @ L.T

    p = CreateGaussianPsdModel(A, Ny, gamma, **kwargs)
    p.normalize()

    return p