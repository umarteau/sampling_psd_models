import unittest

import time
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
        n=20
        d=1

        Bp1 = th.randn(n)


        Xp = th.randn(n,d)
        gp = th.rand(1,d) + 0.2
        Q = th.ones(2,d)
        Q[0,:] *= -1
        Q = None
        p = CreateGaussianPsdModel1(Bp1, Xp, gp,Q, x=list(range(d)))
        return Bp1, Xp, gp, p

    def test_integrate(self):

        for j in range(0,10):
            th.set_default_dtype(th.double)
            a = 2*th.rand(1,2)-1
            b = 5*th.randn(1,2)
            X = 2*th.randn(2,2)
            g = 2*th.rand(2,1) + 0.2


            p = CreateGaussianPsdModel1(th.tensor([1,-1]), X, g,None, x=[0], y = [1])


            # int_(-inf,inf) int_(-inf,inf)  (a[0,0]*x + b[0,0]).sin() * (a[0,1]*y+b[0,1]).sin() * ((-g[0]*(x-X[0,0])**2 - g[1]*(y-X[0,1])**2).exp() - (-g[0]*(x-X[1,0])**2 - g[1]*(y-X[1,1])**2).exp())**2   dx dy
            closed_form_integral = lambda a, b, c, d, g, q, z, w, h, k: -(math.pi*math.exp(-(q*(4*g*(g*(z**2+h**2)+q*(w**2+k**2))+a**2)+c**2*g)/(8*g*q))*(2*math.sin((c*(w+k)+2*d)/2)*math.exp(g*h*z+k*q*w)*math.sin((a*(z+h)+2*b)/2)-math.exp((g*(z**2+h**2)+q*(w**2+k**2))/2)*(math.sin(c*w+d)*math.sin(a*z+b)+math.sin(a*h+b)*math.sin(c*k+d))))/(2*math.sqrt(g)*math.sqrt(q))

            vv = closed_form_integral(a[0,0], b[0,0],a[0,1], b[0,1],g[0],g[1],X[0,0], X[0,1], X[1,0], X[1,1])

            def gg(X, g):
                return ((math.pi / g).sqrt() * (a*X + b).sin() * (-a**2/(4*g)).exp()).prod(1).reshape(-1,1)

            self.assertLessEqual(th.abs(vv - p.integrate(gg)), 1e-6)

    def test_integrate_2(self):

        for j in range(0,10):
            th.set_default_dtype(th.double)
            Q = th.rand(2, 1)
            Q[0,:] *= 2
            Q[0,:] -=1
            Q[1,:] *=5



            X = 2*th.randn(2,1)-1
            g = 2*th.rand(1,1) + 0.2

            B1 = th.tensor([1,-1])


            p = CreateGaussianPsdModel1(B1, X, g,Q, x=[0])


            # int_(-inf,inf) int_(-inf,inf)  (a[0,0]*x + b[0,0]).sin() * (a[0,1]*y+b[0,1]).sin() * ((-g[0]*(x-X[0,0])**2 - g[1]*(y-X[0,1])**2).exp() - (-g[0]*(x-X[1,0])**2 - g[1]*(y-X[1,1])**2).exp())**2   dx dy
            closed_form_integral = lambda a, b, g, x1, x2 : math.sqrt(math.pi)/2/math.sqrt(2*g)*((math.erf(math.sqrt(2*g)*(b-x1))-math.erf(math.sqrt(2*g)*(a-x1)))  -2 * math.exp(-g*(x1-x2)**2/2)*(math.erf(math.sqrt(2*g)*(b-(x1+x2)/2))-math.erf(math.sqrt(2*g)*(a-(x1+x2)/2))) + (math.erf(math.sqrt(2*g)*(b-x2))-math.erf(math.sqrt(2*g)*(a-x2))) )


            vv = closed_form_integral(Q[0,0], Q[1,0], g[0],X[0,0],  X[1,0])

            self.assertLessEqual(th.abs(vv - p.integrate(tol=1e-14)), 1e-6)
    def test_integrate_speed(self):

        for j in range(3):
            th.set_default_dtype(th.double)

            n = 2000
            d = 20

            Q = th.rand(2, d)
            Q[0,:] *= 2
            Q[0,:] -=1
            Q[1,:] *=5




            X = 2*th.randn(n,d)-1
            g = 2*th.rand(1,d) + 0.2

            B1 = th.tensor(th.rand(n))


            p = CreateGaussianPsdModel1(B1, X, g,Q, x=list(range(d)))


            # int_(-inf,inf) int_(-inf,inf)  (a[0,0]*x + b[0,0]).sin() * (a[0,1]*y+b[0,1]).sin() * ((-g[0]*(x-X[0,0])**2 - g[1]*(y-X[0,1])**2).exp() - (-g[0]*(x-X[1,0])**2 - g[1]*(y-X[1,1])**2).exp())**2   dx dy



            st = time.time()
            p.integrate(tol=1e-14)
            print(f"with tolerance  {time.time() - st}")

            st = time.time()
            p.integrate()
            print(f"with nothing{time.time()-st}")

            assert True

    def test_normalization_constant(self):
        _, _, _, p = self.test_create()


        dtype = p.dtype
        device = p.dev
        d = p.dimensions()

        on = th.ones(1, d, device=device, dtype = dtype)

        vv = p.integrate_hypercube(-math.inf * on, math.inf * on)

        self.assertLessEqual(th.abs(vv - p.normalization_constant()), 1e-6)

    def test_normalize(self):
        _, _, _, p = self.test_create()

        p.normalize()

        self.assertLessEqual(th.abs(p.normalization_constant() - 1), 1e-6)

    def test_sample(self):
        _, _, _, p = self.test_create()

        p.normalize()

        for j in range(0,10):
            a = 0.1*th.rand(1,p.dimensions())-0.05
            b = 6*th.randn(1,p.dimensions())

            def gg(X, g):
                return ((math.pi / g).sqrt() * (a*X + b).sin() * (-a**2/(4*g)).exp()).prod(1).reshape(-1,1)

            vv = p.integrate(gg)

            samples = p.sample(1000,tol=1e-5,tolInt=1e-14)
            qq = (samples * a + b).sin().prod(1).mean()

            print(vv)
            print(qq)

            self.assertLessEqual(th.abs(vv - qq), 1e-2)






if __name__ == '__main__':
    unittest.main()
