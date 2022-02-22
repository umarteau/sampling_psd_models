import numpy as np
import matplotlib.pyplot as plt
from gaussian_psd_model import findBestHellinger as findSol
from gaussian_psd_model import *
import pickle
import time
import math


th.set_default_dtype(th.float64)
cm = plt.cm.get_cmap('RdYlBu_r')

class gridSampler:
    def __init__(self,p,Q,n_side = None,n = None):
        d = Q.size(1)
        if n_side is None:
            n_side = int(math.pow(n,1/d))
        n = math.pow(n_side,d)

        a = Q[0,:]
        b = Q[1,:]
        steps = (b - a) / n_side
        gridTensors = th.meshgrid([th.arange(a[i],b[i],steps[i]) for i in range(d)])
        gridTensor = th.cat([gt.unsqueeze(-1) for gt in gridTensors],dim=-1).view(-1,d)
        p_values = p(gridTensor)
        integral = p_values.sum().item()
        p_values /= integral
        self.gridTensor = gridTensor
        self.probas = p_values
        self.steps = steps
        self.a = a
        self.b = b
        self.n = n
        self.n_side = n_side
        self.d = d
    def sample(self,N):
        indices = th.multinomial(self.probas,N,replacement = True)
        vecs = self.gridTensor[indices,:]
        return vecs + self.steps[None,:]*th.rand(N,self.d)
        
        






d = 2


Q = th.tensor([[-3.,-3.],[3.,3.]])

def mu_sampler(n):
    return Q[0,:][None,:] + (Q[1,:]-Q[0,:])[None,:]*th.rand(n,d)

def mu_density(x):
    return th.ones(x.size(0))/(Q[1,:]-Q[0,:]).prod()


n_grid_side = 200
n_grid = n_grid_side**2

xx_grid = np.linspace(Q[0,0],Q[1,0],n_grid_side)
yy_grid = np.linspace(Q[0,1],Q[1,1],n_grid_side)

xxx_grid,yyy_grid = np.meshgrid(xx_grid,yy_grid)
XX_grid,YY_grid = th.from_numpy(xxx_grid.flatten()),th.from_numpy(yyy_grid.flatten())

X_grid = th.cat([XX_grid[:,None],YY_grid[:,None]],dim =1 )


def plot_contour(fig,ax0,vmax,pntest):
    sc = ax0.contourf(xxx_grid, yyy_grid, pntest.reshape(n_grid_side,n_grid_side), vmin=0, vmax=vmax, levels=35, cmap=cm)
    fig.colorbar(sc, ax=ax0)

def scatter_color(fig, ax0, vmax, Xtest, ptest):
    sc = ax0.scatter(Xtest[:, 0], Xtest[:, 1], c=ptest, vmin=0, vmax=vmax, cmap=cm)
    fig.colorbar(sc, ax=ax0)


def p_fun(x):
    d = 2
    s = th.cat([0.7*th.ones(d).unsqueeze(0), th.tensor([2,0.3]).unsqueeze(0), th.tensor([0.7,0.7]).unsqueeze(0)],dim=0)

    mu = th.cat([-th.ones(1, d), th.ones(1, d), th.ones(1, d)], 0)
    alpha = th.tensor([0.3, 0.4, 0.4])

    X = ((x.unsqueeze(0)-mu.unsqueeze(1))/math.sqrt(2)/s.unsqueeze(1))
    res = -X.pow(2).sum(-1)
    res = res.exp()
    res = res*alpha[:,None]
    return res.sum(0)

def p_sample(N,Q):
    a = Q[0,:]
    b = Q[1,:]

    d = 2
    s = th.cat([0.7*th.ones(d).unsqueeze(0), th.tensor([2,0.3]).unsqueeze(0), th.tensor([0.7,0.7]).unsqueeze(0)],dim=0)

    mu = th.cat([-th.ones(1, d), th.ones(1, d), th.ones(1, d)], 0)
    alpha = th.tensor([0.3, 0.4, 0.4])

    c = (2 * math.pi * s.pow(2)).sqrt().prod(dim=-1)
    alpha_bis = alpha*c

    Naux = 0
    l = []
    while Naux < N:

        index_samples = th.multinomial(alpha_bis,N,replacement = True)
        s_samples = s[index_samples,:]
        mu_samples = mu[index_samples,:]
        new_samples = mu_samples + s_samples*th.randn(N,d)
        mask = (((new_samples < a[None,:])+(new_samples > b[None,:])).sum(1) == 0)
        new_samples = new_samples[mask,:]
        N_new_samples  = new_samples.size(0)
        Naux += N_new_samples
        l.append(new_samples)

    samples = th.cat(l)[:N,:]

    return samples

#plot density

fig,ax = plt.subplots(1,1)
fig.suptitle('Probability to learn')
p_grid = p_fun(X_grid)
plot_contour(fig,ax,p_grid.max(),p_grid)
plt.show(block= False)


# Learning

def learn_model(n,m,name = 'experiment_1'):
    true_name = f'{name}_n_{n}_m_{m}'
    try:
        with open(f'test_{true_name}.pickle', 'rb') as handle:
            dico_test = pickle.load(handle)
        with open(f'train_{true_name}.pickle', 'rb') as handle:
            dico_train = pickle.load(handle)
        return dico_train,dico_test
    except:
        proportion = 0.5
        n_train = int(proportion*n)
        n_test = n-n_train

        X_train = mu_sampler(n_train)
        X_test = mu_sampler(n_test)
        mu_train = mu_density(X_train)
        mu_test = mu_density(X_test)

        p_train = p_fun(X_train)
        p_test = p_fun(X_test)


        #gammaList = [1,0.5,0.3,0.25,0.2,0.1,0.05]
        #gammaList = [0.1,0.2,0.4,0.6,0.8,1]
        gammaList = [0.008,0.01,0.05,0.1,0.2,0.5,1,2,3,5,10]
        laList = [1,1e-1,1e-2,1e-3,1e-4,1e-6,1e-7,1e-8,1e-9]
        #laList = [1e-6]
        empirical =True
        #save = None

        dico_train,dico_test = findSol(X_train,p_train,gammaList,laList,Ny = m,X_test=X_test\
                                                                                    ,p_test= p_test,Q =Q,\
                                     m_compression = m,mu_train = mu_train,mu_test =mu_test,tol = 1e-14,empirical = empirical,save = true_name,retrain = False)
        return dico_train,dico_test

name = 'experiment_1'
n = 10000
m= 90


dico_train,dico_test = learn_model(n,m,name)

alpha,X,g= dico_test['alpha_best'],dico_train['Ny'],dico_test['gamma_best']
g = g*th.ones(1,d)
p = CreateGaussianPsdModel1(alpha, X, g,Q, x=list(range(d)))
p_pred_grid = p(X_grid)
m = X.size(0)


fig,ax = plt.subplots(1,2)
plot_contour(fig,ax[0],p_grid.max(),p_grid)
ax[0].set_title('Probability')
plot_contour(fig,ax[1],p_grid.max(),p_pred_grid)
ax[1].set_title('Predicted')
fig.suptitle(f'One shot prediction from {m} random nystrom samples and with {n} points')
plt.show(block = False)

#########
# Sampling
######

Nsamples = 1000
Nmax = 100
adaptive = None
tol = 1e-5
v = p.sample_from_hypercube(Nsamples,tol =tol,tolInt = 1e-13,adaptive = adaptive,n_jobs = 6,Nmax=Nmax)
plt.figure()
plt.scatter(v[:,0],v[:,1])
plt.title('learnt_samples')
plt.show(block = False)


grid_sampler = gridSampler(p_fun,Q,n=n)
samples_bad = grid_sampler.sample(Nsamples)

plt.figure()
plt.scatter(samples_bad[:,0],samples_bad[:,1])
plt.title('grid_samples')
plt.show(block = False)

samples_true = p_sample(Nsamples,Q)

plt.figure()
plt.scatter(samples_true[:,0],samples_true[:,1])
plt.title('true_samples')
plt.show(block =False)


def MMD(X1,X2,eta,delta=0.9):
    g = eta*th.ones(1,d)
    n = X1.size(0)
    m = X2.size(0)
    t1 = gaussKern(X1,X1,g).sum().sum().item()/n**2
    t2 = gaussKern(X2,X2,g).sum().sum().item()/(n*m)
    t3 = gaussKern(X1,X2,g).sum().sum().item()/m**2
    error_1 = math.log(2/delta)/n + math.sqrt(math.log(2/delta)/n)
    error_2 = math.log(2 / delta) / m + math.sqrt(math.log(2 / delta) / m)
    error = error_1+error_2

    return math.sqrt(t1 + t2 - 2*t3),error
eta = 1
mmd_good,error_good = MMD(samples_true,v,eta)
mmd_bad,error_bad = MMD(samples_bad,samples_true,eta)

print(f'mmd good  between : {mmd_good-error_good,mmd_good+error_good}, mmd bad : {mmd_bad-error_bad,mmd_bad+error_bad}')
plt.show()