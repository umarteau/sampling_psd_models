import unittest

from gaussian_psd_model import *
import torch as th
import sympy as sy


class MyTestCase(unittest.TestCase):

    def test_check_kern(self):
        X1 = th.randn(3,5)
        X2 = th.randn(4,5)

        gamma = th.tensor(range(1,6), dtype=th.float64).view(1,-1)
        K = gaussKern(X1,X2, gamma)

        for i in range(3):
            for j in range(4):
                self.assertLessEqual(th.abs(K[i,j] - math.exp(-(gamma*(X1[i,:] - X2[j,:])**2).sum())), 1e-6)

    def test_create(self):
        A = th.randn(3,20)
        Bp = A.T @ A
        Xp = th.randn(20,7)
        gp = th.rand(1,7) + 0.2
        p = CreateGaussianPsdModel(Bp, Xp, gp, x=[0,1], y=[2,3,4], z=[5,6])
        return A, Bp, Xp, gp, p


    def test_evaluate(self):

        A, Bp, Xp, gp, p = self.test_create()

        for i in range(0,10):
            v = th.randn(1,7)
            val = p(x=v[:,0:2], y=v[:,2:5], z=v[:,5:7])

            k = gaussKern(Xp, v, gp)
            self.assertLessEqual(th.abs(val - k.T @ (Bp @ k)), 1e-6)


    def test_marginalize(self):
        A, Bp, Xp, gp, p = self.test_create()
        yy = p.marginalize(x = [], z = [])

        cc = (math.pi/(2*gp[:,[0,1,5,6]])).sqrt().prod()
        yyB = cc * Bp * gaussKern(Xp[:,[0,1,5,6]],Xp[:,[0,1,5,6]],gp[:,[0,1,5,6]]/2)
        self.assertLessEqual(th.norm(yy.B - yyB), 1e-6)
        self.assertLessEqual(th.norm(yy.X.to_bs_matrix().X - Xp[:,[2,3,4]]), 1e-6)

    def test_multiply(self):
        A, Bp, Xp, gp, p = self.test_create()

        A = th.randn(4,4)
        Bq = A.T @ A
        Xq = th.randn(4,8)
        gq = th.rand(1,8)+0.3
        q = CreateGaussianPsdModel(Bq, Xq, gq, y=[0,1,2], z=[3,4], w=[5,6,7])

        hh = p * q

        for i in range(0,10):
            v = th.randn(1,10)
            vx = v[:,0:2]; vy=v[:,2:5]; vz=v[:,5:7]; vw=v[:,7:10]
            self.assertLessEqual(th.norm(
                hh(x=vx, y=vy, z=vz, w=vw) - p(x=vx, y=vy, z=vz) * q(y=vy, z=vz, w=vw)
            ), 1e-3)


        self.assertLessEqual(th.norm(Xp[:,0:2] - (p * q).marginalize(y =[], z=[], w=[]).X.to_bs_matrix().X), 1e-6 )

    def test_integrate(self):

        for j in range(0,10):
            th.set_default_dtype(th.double)
            a = 2*th.rand(1,2)-1
            b = 5*th.randn(1,2)
            X = 2*th.randn(2,2)
            g = 2*th.rand(2,1) + 0.2


            p = CreateGaussianPsdModel(th.tensor([[1,-1],[-1,1]]), X, g, x=[0], y = [1])


            # int_(-inf,inf) int_(-inf,inf)  (a[0,0]*x + b[0,0]).sin() * (a[0,1]*y+b[0,1]).sin() * ((-g[0]*(x-X[0,0])**2 - g[1]*(y-X[0,1])**2).exp() - (-g[0]*(x-X[1,0])**2 - g[1]*(y-X[1,1])**2).exp())**2   dx dy
            closed_form_integral = lambda a, b, c, d, g, q, z, w, h, k: -(math.pi*math.exp(-(q*(4*g*(g*(z**2+h**2)+q*(w**2+k**2))+a**2)+c**2*g)/(8*g*q))*(2*math.sin((c*(w+k)+2*d)/2)*math.exp(g*h*z+k*q*w)*math.sin((a*(z+h)+2*b)/2)-math.exp((g*(z**2+h**2)+q*(w**2+k**2))/2)*(math.sin(c*w+d)*math.sin(a*z+b)+math.sin(a*h+b)*math.sin(c*k+d))))/(2*math.sqrt(g)*math.sqrt(q))

            vv = closed_form_integral(a[0,0], b[0,0],a[0,1], b[0,1],g[0],g[1],X[0,0], X[0,1], X[1,0], X[1,1])

            def gg(X, g):
                return ((math.pi / g).sqrt() * (a*X + b).sin() * (-a**2/(4*g)).exp()).prod(1).reshape(-1,1)

            self.assertLessEqual(th.abs(vv - p.integrate(gg)), 1e-6)

    def test_normalization_constant(self):
        _, _, _, _, p = self.test_create()


        dtype = p.dtype
        device = p.dev
        d = p.dimensions()

        on = th.ones(1, d, device=device, dtype = dtype)

        vv = p.integrate_hypercube(-math.inf * on, math.inf * on)

        self.assertLessEqual(th.abs(vv - p.normalization_constant()), 1e-6)

    def test_normalize(self):
        _, _, _, _, p = self.test_create()

        p.normalize()

        self.assertLessEqual(th.abs(p.normalization_constant() - 1), 1e-6)

    def test_sample(self):
        _, _, _, _, p = self.test_create()

        p.normalize()

        for j in range(0,10):
            a = 0.1*th.rand(1,p.dimensions())-0.05
            b = 6*th.randn(1,p.dimensions())

            def gg(X, g):
                return ((math.pi / g).sqrt() * (a*X + b).sin() * (-a**2/(4*g)).exp()).prod(1).reshape(-1,1)

            vv = p.integrate(gg)

            samples = p.sample(1000)

            qq = (samples * a + b).sin().prod(1).mean()

            print(vv)
            print(qq)

            self.assertLessEqual(th.abs(vv - qq), 1e-2)

    def test_mle_estimate(self):

        th.set_default_dtype(th.double)
        device = 'cpu'

        X = 2*th.randn(2,2, device = device)
        g = 2*th.rand(2,1, device = device) + 0.2

        p = CreateGaussianPsdModel(th.tensor([[1.0,-1.0],[-1.0,1.0]], device = device), X, g, x=[0], y = [1])

        X = p.sample(10000)

        m = mle_from_samples(X, 100, 1e-1, 0.5, x=[0], y = [1])

        X = p.sample(10000)
        v = (p(X) - m(X)).abs()
        print(v.mean(), v.max())


if __name__ == '__main__':
    unittest.main()
