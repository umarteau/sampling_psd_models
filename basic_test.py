import torch as th
import numpy as np
import matplotlib.pyplot as plt
from gaussian_psd_model import findBestHellinger as findSol
import pickle


th.set_default_dtype(th.float64)

cm = plt.cm.get_cmap('RdYlBu_r')


n_test = 100

xx_test = np.linspace(-20,20,n_test)
yy_test = np.linspace(-20,20,n_test)

xxx_test,yyy_test = np.meshgrid(xx_test,yy_test)
XX_test,YY_test = th.from_numpy(xxx_test.flatten()),th.from_numpy(yyy_test.flatten())

X_test = th.cat([XX_test[:,None],YY_test[:,None]],dim =1 )


def plot_contour(fig,ax0,vmax,pntest):
    sc = ax0.contourf(xxx_test, yyy_test, pntest.reshape(n_test,n_test), vmin=0, vmax=vmax, levels=35, cmap=cm)
    fig.colorbar(sc, ax=ax0)

def scatter_color(fig, ax0, vmax, Xtest, ptest):
    sc = ax0.scatter(Xtest[:, 0], Xtest[:, 1], c=ptest, vmin=0, vmax=vmax, cmap=cm)
    fig.colorbar(sc, ax=ax0)

# dimension
d = 2

#integration domain : square of diameter 1

Q = th.ones(2,d)
Q[0,:]*=-1
Q*= 20

A0 = th.tensor([[-20]*d])
B0 = th.tensor([[20]*d])

def mu0_sampler(n,max_n = None,n_jobs = None):
    return 40*th.rand(n,d)-20

def mu0(x):
    return 1*th.ones(x.size()[:-1])/1600


def V(x,b=0.05):
    x1 = x[...,0]
    x2 = x[...,1]
    return 1e-2*x1.pow(2) + (x2+b*x1.pow(2)-100*b).pow(2)

beta_obj = 1


def p_beta(x,beta=1):
    Vx = V(x)
    Vx -= Vx.min()
    Vx *= -beta
    Vx.exp_()
    return Vx


def proba_stupid(x):
    mu1 = 0
    mu2 = 2
    sigma1 = 0.5
    sigma2 = 0.1
    return th.exp(-(x-mu1).pow(2)/(2*sigma1**2 )) + th.exp(-(x-mu2).pow(2)/(2*sigma2**2 ))

def fun_plot():
    fig,ax = plt.subplots(1,1)
    fig.suptitle('Learning with non-negative coefficients')
    xtest = th.linspace(-2,3,200)
    ptest = proba_stupid(xtest)
    gauss1 = 0.5*th.exp(-(xtest-2).pow(2)/(2*0.08**2 ))
    gauss2 = 0.5*th.exp(-(xtest-2).pow(2)/(2*0.3**2 ))
    plt.plot(xtest,ptest,color = 'g',lw=3)
    plt.plot(xtest,gauss1,linestyle = '--', color = 'b',label = '$\sigma = 0.08$',lw = 3)
    plt.plot(xtest,gauss2,linestyle = '--', color = 'r',label = '$\sigma = 0.3$',lw =3)
    plt.legend()
    fig.savefig('hihi.png')
    plt.show()
    pass


fig,ax = plt.subplots(1,1)
fig.suptitle('Probability to learn')
p_n_test = p_beta(X_test,beta_obj)
plot_contour(fig,ax,p_n_test.max(),p_n_test)
plt.show(block = False)


###############################
# First try
#############################

#score_ref,alpha_ref,Ny_ref,q_ref,lambda_ref,gamma_ref,score_gamma = findBestHellinger(X_train,p_train,gamma,la,Ny = None,X_test=None,p_test= None,Q =None,\
#                             m_compression = None,mu_train = None,mu_test =None,tol = 1e-14)

n=2000
m =1000


X_train = mu0_sampler(n)
mu_train = th.ones(n)/1600
mu_test = th.ones(X_test.size(0))/1600

beta = 0.1
p_fun = lambda x : p_beta(x,beta = beta)
p_train = p_fun(X_train)
p_test = p_fun(X_test)

#gammaList = [1,0.5,0.3,0.25,0.2,0.1,0.05]
#gammaList = [0.1,0.2,0.4,0.6,0.8,1]
gammaList = [0.008,0.01,0.05,0.1,0.2,0.5,1]
laList = [10000,1000,100,20,10,1,1e-1,1e-2,1e-3,1e-4,1e-6,1e-7,1e-8,1e-9]
#laList = [1e-6]
empirical =True




dico_train,dico_test = findSol(X_train,p_train,gammaList,laList,Ny = m,X_test=X_test\
                                                                            ,p_test= p_test,Q =Q,\
                             m_compression = None,mu_train = mu_train,mu_test =mu_test,tol = 1e-14,empirical = empirical,save = 'first_try')

with open('test_first_try.pickle', 'rb') as handle:
    dico_test = pickle.load(handle)

p_pred = dico_test['q_best'].pow(2)

fig,ax = plt.subplots(1,2)
plot_contour(fig,ax[0],p_test.max(),p_test)
ax[0].set_title('Probability')
plot_contour(fig,ax[1],p_test.max(),p_pred)
ax[1].set_title('Predicted')
fig.suptitle(f'One shot prediction from {m} random nystrom samples, SIMPLE case')
plt.show()


from test_tony_data_hell.solving_system import chase_projection

Knm = gaussian_kernel(X_train,C,sigma = sigma)
Knm /= qb_train[:,None]

b = q_train/qb_train

mbis = 100

i,score = chase_projection(Knm,b,mbis)
print(score[-1])

score,alpha,C,q_test_predicted,l,sigma,score_list = solve_big_problem_simple(X_train,q_train,X_test,q_test,gaussian_kernel,sigma_list,lambda_list,\
                             qb_train = qb_train,qb_test =None,m=m,X_nystrom = C[i,:])

fig,ax = plt.subplots(1,2)
plot_contour(fig,ax[0],q_test.pow(2).max(),q_test.pow(2))
ax[0].set_title('Probability')
plot_contour(fig,ax[1],q_test.pow(2).max(),q_test_predicted.pow(2))
ax[1].set_title('Predicted')
fig.suptitle(f'After reducing to {mbis} points')
plt.show(block = False)