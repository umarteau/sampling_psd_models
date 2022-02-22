import torch as th
import numpy as np
import matplotlib.pyplot as plt
from gaussian_psd_model import findBestHellinger as findSol
from gaussian_psd_model import *
import pickle
import time
import math

th.set_default_dtype(th.float64)

cm = plt.cm.get_cmap('RdYlBu_r')


n_test = 200
d = 2


Q = th.tensor([[-3.,-3.],[3.,3.]])

def mu_sampler(n):
    return Q[0,:][None,:] + (Q[1,:]-Q[0,:])[None,:]*th.rand(n,d)

def mu_density(x):
    return th.ones(x.size(0))/(Q[1,:]-Q[0,:]).prod()


xx_test = np.linspace(Q[0,0],Q[1,0],n_test)
yy_test = np.linspace(Q[0,1],Q[1,1],n_test)

xxx_test,yyy_test = np.meshgrid(xx_test,yy_test)
XX_test,YY_test = th.from_numpy(xxx_test.flatten()),th.from_numpy(yyy_test.flatten())

X_test = th.cat([XX_test[:,None],YY_test[:,None]],dim =1 )


def plot_contour(fig,ax0,vmax,pntest):
    sc = ax0.contourf(xxx_test, yyy_test, pntest.reshape(n_test,n_test), vmin=0, vmax=vmax, levels=35, cmap=cm)
    fig.colorbar(sc, ax=ax0)

def scatter_color(fig, ax0, vmax, Xtest, ptest):
    sc = ax0.scatter(Xtest[:, 0], Xtest[:, 1], c=ptest, vmin=0, vmax=vmax, cmap=cm)
    fig.colorbar(sc, ax=ax0)


def p_fun(x):
    d = 2
    s = th.tensor([0.7, 0.6, 0.7])
    mu = th.cat([-th.ones(1, d), th.ones(1, d), th.ones(1, d)], 0)
    alpha = th.tensor([0.3, -0.4, 0.4])

    X = ((x.unsqueeze(0)-mu.unsqueeze(1))/math.sqrt(2)/s[:,None,None])
    res = -X.pow(2).sum(-1)
    res = res.exp()
    res = res*alpha[:,None]
    return res.sum(0)


#plot density

fig,ax = plt.subplots(1,1)
fig.suptitle('Probability to learn')
p_n_test = p_fun(X_test)
plot_contour(fig,ax,p_n_test.max(),p_n_test)
plt.show(block = False)


# Learning

n=2000
m =50


X_train = mu_sampler(n)
mu_train = mu_density(X_train)
mu_test = mu_density(X_test)

p_train = p_fun(X_train)
p_test = p_fun(X_test)

#gammaList = [1,0.5,0.3,0.25,0.2,0.1,0.05]
#gammaList = [0.1,0.2,0.4,0.6,0.8,1]
gammaList = [0.008,0.01,0.05,0.1,0.2,0.5,1]
laList = [10000,1000,100,20,10,1,1e-1,1e-2,1e-3,1e-4,1e-6,1e-7,1e-8,1e-9]
#laList = [1e-6]
empirical =True
#save = None
save = 'experiment_1'




dico_train,dico_test = findSol(X_train,p_train,gammaList,laList,Ny = m,X_test=X_test\
                                                                            ,p_test= p_test,Q =Q,\
                             m_compression = None,mu_train = mu_train,mu_test =mu_test,tol = 1e-14,empirical = empirical,save = save)

with open(f'test_{save}.pickle', 'rb') as handle:
    dico_test = pickle.load(handle)

with open(f'train_{save}.pickle', 'rb') as handle:
    dico_train = pickle.load(handle)

p_pred = dico_test['q_best'].pow(2)




fig,ax = plt.subplots(1,2)
plot_contour(fig,ax[0],p_test.max(),p_test)
ax[0].set_title('Probability')
plot_contour(fig,ax[1],p_test.max(),p_pred)
ax[1].set_title('Predicted')
fig.suptitle(f'One shot prediction from {m} random nystrom samples and with {n} points')
plt.show(block = False)

#########
# Sampling
######

adaptive = 'hellinger'
alpha,X,g= dico_test['alpha_best'],dico_train['Ny'],dico_test['gamma_best']
g = g*th.ones(1,d)
p = CreateGaussianPsdModel1(alpha, X, g,Q, x=list(range(d)))
tol = 1e-3
st = time.time()
v = p.sample_from_hypercube_profiled(1,tol =tol,tolInt = 1e-14,adaptive = adaptive)
print(f'time for intelignet sampling {time.time()-st}')
plt.figure()
plt.scatter(v[:,0],v[:,1])
plt.show(block = False)

