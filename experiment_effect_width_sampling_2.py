from gaussian_psd_model import findBestHellinger as findSol
from gaussian_psd_model import *
import pickle
import time
import math



th.set_default_dtype(th.float64)
cm = plt.cm.get_cmap('Blues')

n_grid = 300
d = 2

Q = th.ones(2,d)
Q[0,:] *=-1
Q *=20


def mu_sampler(n):
    return Q[0, :][None, :] + (Q[1, :] - Q[0, :])[None, :] * th.rand(n, d)


def mu_density(x):
    return th.ones(x.size(0)) / (Q[1, :] - Q[0, :]).prod()


def scatter_color(fig, ax0, vmax, Xtest, ptest):
    sc = ax0.scatter(Xtest[:, 0], Xtest[:, 1], c=ptest, vmin=0, vmax=vmax, cmap=cm)
    fig.colorbar(sc, ax=ax0)

xx_grid = np.linspace(Q[0,0],Q[1,0],n_grid)
yy_grid = np.linspace(Q[0,1],Q[1,1],n_grid)

xxx_grid,yyy_grid = np.meshgrid(xx_grid,yy_grid)
xx_grid,yy_grid = th.from_numpy(xxx_grid.flatten()),th.from_numpy(yyy_grid.flatten())

X_grid = th.cat([xx_grid[:,None],yy_grid[:,None]],dim =1 )


def plot_contour(fig,ax0,vmax,pntest):
    sc = ax0.contourf(xxx_grid, yyy_grid, pntest.reshape(n_grid,n_grid), vmin=0, vmax=vmax, levels=35, cmap=cm)
    #fig.colorbar(sc, ax=ax0)





def V(x,b=0.05):
    x1 = x[...,0]
    x2 = x[...,1]
    return 1e-2*x1.pow(2) + (x2+b*x1.pow(2)-100*b).pow(2)

def f(x):
    return x[...,0].pow(2)+x[...,1].pow(2)


beta_obj = 1

def p_beta(x,beta=1):
    Vx = V(x)
    Vx -= Vx.min()
    Vx *= -beta
    Vx.exp_()
    return Vx

p_fun = lambda x : p_beta(x,beta=beta_obj)



def learn_model(n,m,name = 'experiment_1',advancedNy = False,m_intermediate = None,Nmax = None,overwrite = False):
    true_name = f'{name}_n_{n}_m_{m}_d_{d}'
    try:
        if overwrite == True:
            print('overwrite is true, overwriting model')
            raise ValueError
        with open(f'models/test_{true_name}.pickle', 'rb') as handle:
            dico_test = pickle.load(handle)
        with open(f'models/train_{true_name}.pickle', 'rb') as handle:
            dico_train = pickle.load(handle)

        gamma_ref, lambda_ref, score_ref = dico_test['gamma_best'], dico_test['lambda_best'], dico_test['score_best']
        print(f'Final score for gamma {gamma_ref}, lambda {lambda_ref} : {score_ref}')
        return dico_train,dico_test
    except:
        proportion = 0.5
        n_train = int(proportion*n)
        n_test = n-n_train

        X_train = mu_sampler(n_train)
        X_test = mu_sampler(n_test)
        mu_train = mu_density(X_train)
        mu_test = mu_density(X_test)

        q_train = p_fun(X_train).sqrt()
        q_test = p_fun(X_test).sqrt()


        #gammaList = [1,0.5,0.3,0.25,0.2,0.1,0.05]
        #gammaList = [0.1,0.2,0.4,0.6,0.8,1]
        gammaList = [0.01,0.05,0.1,0.2,0.5,1,2,4,5,10]
        laList = [1,1e-1,1e-2,1e-3,1e-4,1e-6,1e-7,1e-8,1e-9]
        #laList = [1e-6]
        empirical =True
        #save = None
        if advancedNy:
            n_jobs = 5
            if Nmax is None:
                Nmax = m_intermediate//n_jobs

            Ny = find_Ny_from_sqrt(X_train, q_train, gammaList, laList, m_intermediate, X_test=X_test, q_test=q_test,\
                                   Q=Q, \
                              mu_train=mu_train, mu_test=mu_test, tol=1e-14, \
                              empirical=True, retrain=True, Nmax=Nmax, n_jobs=n_jobs)
            dico_train, dico_test = findSol(X_train, q_train, gammaList, laList, Ny=Ny, X_test=X_test \
                                            , q_test=q_test, Q=Q, \
                                            m_compression=m, mu_train=mu_train, mu_test=mu_test, tol=1e-14, \
                                            empirical=empirical, dir='models', save=f'{true_name}', retrain=True)
        else:
            dico_train,dico_test = findSol(X_train,q_train,gammaList,laList,Ny = m,X_test=X_test\
                                                                                    ,p_test= q_test,Q =Q,\
                                     m_compression = None,mu_train = mu_train,mu_test =mu_test,tol = 1e-14,\
                                       empirical = empirical,dir = 'models',save = f'{true_name}',retrain = True)
        return dico_train,dico_test




name = f'experiment_5_warped_beta_{beta_obj}'
#overwrite = True
overwrite = False
n = 100000
m_intermediate = 300
m = 300
doubles_max = 100*100*1000*5
advancedNy = True
# advancedNy = False
eta = 2

Nmax = int(doubles_max/m_intermediate/m_intermediate/d)


dico_train, dico_test = learn_model(n, m, name, advancedNy=advancedNy, m_intermediate=m_intermediate, Nmax=Nmax,
                                    overwrite=overwrite)

alpha, X, g = dico_train['alpha'], dico_train['Ny'], dico_train['gamma']
g = g * th.ones(1, d)

p = CreateGaussianPsdModel1(alpha, X, g, Q, x=list(range(d)))
tol=1e-3
n_jobs = 5
Nsamples = 1000
doubles_max = 100*100*1000*5

#tol_list = [14,7,4]
tol_list = [7,4,0.1]
def get_sample_tols(tol_list):
    res = {}
    for tol in tol_list:

        sample_name = f'{name}samples_{Nsamples}_n_{n}_m_{m}_d_{d}_tol_{tol}.pickle'

        try:
            with open(f'samples/{sample_name}', 'rb') as handle:
                if overwrite:
                    print('here')
                    raise ValueError
                dico_samples = pickle.load(handle)
        except:
            Nmax = int(doubles_max / m ** 2 / d)
            print('haere')
            samples_model = p.sample_from_hypercube(Nsamples, tol=tol, tolInt=1e-13, adaptive=None, n_jobs=n_jobs,
                                                      Nmax=Nmax)
            print('here')
            dico_samples = {'model': samples_model}

            with open(f'samples/{sample_name}', 'wb') as handle:
                pickle.dump(dico_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

        samples_model = dico_samples['model']
        res[tol] = samples_model
    return(res)


dico_samples = get_sample_tols(tol_list)
m = X.size(0)


#plot density



delta = Q[1,:] - Q[0,:]

major_xticks = th.arange(Q[0,0],Q[1,0] + delta[0].item()/10,delta[0].item()/4)
major_yticks = th.arange(Q[0,1],Q[1,1] + delta[1].item()/10,delta[1].item()/4)
labelsize = 5
markersize = 2
# size of the grid
alpha = 1
tols = tol_list
version = 3

fig,ax = plt.subplots(2,2)

#first plot the distribution
p_n_grid = p(X_grid)
plot_contour(fig,ax[0,0],p_n_grid.max(),p_n_grid)
ax[0,0].set_title(None)
ax[0,0].set_xticks(major_xticks)
ax[0,0].set_yticks(major_yticks)
ax[0,0].tick_params(axis='x', which='both', bottom=None, top=None, labelbottom=False, labeltop=False)
ax[0,0].tick_params(axis='y', which='both', left=None, right=None, labelleft=False, labelright=False)


def make_points(tol,ax,title = None):
    sides = delta/((th.ceil(th.log(delta/tol)/math.log(2)).type(th.int64)*math.log(2)).exp())
    samples = dico_samples[tol]
    minor_xticks = th.arange(Q[0,0],Q[1,0]+0.1,sides[0].item())
    minor_yticks = th.arange(Q[0,1],Q[1,1]+0.1,sides[1].item())

    ax.set_xticks(major_xticks)
    ax.set_yticks(major_yticks)
    ax.set_xticks(minor_xticks,minor =True)
    ax.set_yticks(minor_yticks,minor = True)

    ax.scatter(samples[:,0],samples[:,1],marker = '+',label = rf'$\rho = {tol}$',s=markersize)

    ax.set_xlim(Q[0,0],Q[1,0])
    ax.set_ylim(Q[0,1],Q[1,1])
    ax.set_title(title)
    ax.legend(loc = 1)
    #size of the labels under the plot
    ax.tick_params(axis = 'x',which = 'both',bottom = None,top = None,labelbottom = False,labeltop=False)
    ax.tick_params(axis = 'y',which = 'both',left = None,right = None,labelleft = False,labelright = False)
    if tol>1:
        ax.grid(which = 'both',alpha = alpha)

make_points(tols[0],ax[0,1])
make_points(tols[1],ax[1,0])
make_points(tols[2],ax[1,1])

plt.tight_layout()
fig.savefig(f'plots/tau_effect_{version}.pdf' , format='pdf', dpi=1000)




