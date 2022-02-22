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
        p_values = p(gridTensor).view(-1)
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
        
        






d = 5


Q = th.tensor([[-1.]*d,[1.]*d])

def mu_sampler(n):
    return Q[0,:][None,:] + (Q[1,:]-Q[0,:])[None,:]*th.rand(n,d)

def mu_density(x):
    return th.ones(x.size(0))/(Q[1,:]-Q[0,:]).prod()







def scatter_color(fig, ax0, vmax, Xtest, ptest):
    sc = ax0.scatter(Xtest[:, 0], Xtest[:, 1], c=ptest, vmin=0, vmax=vmax, cmap=cm)
    fig.colorbar(sc, ax=ax0)


def make_p_fun(d):
    X_p =th.ones(2,d)
    X_p[1,:] *=-1
    gamma_p = 4*0.25*th.ones(1,d)/d
    alpha_p = th.tensor([1,-1],dtype = th.float64)
    Q_p = th.ones(2,d)
    Q_p[0,:] *=-1
    p_obj = CreateGaussianPsdModel1(alpha_p, X_p, gamma_p,Q_p, x=list(range(d)))
    return p_obj





def plot_2():

    p_obj = make_p_fun(2)
    n_grid_side = 200

    xx_grid = np.linspace(Q[0, 0], Q[1, 0], n_grid_side)
    yy_grid = np.linspace(Q[0, 1], Q[1, 1], n_grid_side)

    xxx_grid, yyy_grid = np.meshgrid(xx_grid, yy_grid)
    xx_grid, yy_grid = th.from_numpy(xxx_grid.flatten()), th.from_numpy(yyy_grid.flatten())

    X_grid = th.cat([xx_grid[:, None], yy_grid[:, None]], dim=1)


    def plot_contour(fig, ax0, vmax, pntest):
        sc = ax0.contourf(xxx_grid, yyy_grid, pntest.reshape(n_grid_side, n_grid_side), vmin=0, vmax=vmax, levels=35,
                          cmap=cm)
        fig.colorbar(sc, ax=ax0)


    p_grid = p_obj(X_grid)

    fig, ax = plt.subplots(1, 1)

    ax.set_title('target')
    plot_contour(fig, ax, p_grid.max(), p_grid)
    plt.show(block=False)
    tol = 1e-4
    Nmax = 200
    v = p_obj.sample_from_hypercube(200, tol=tol, tolInt=1e-13, adaptive=None, n_jobs=1, Nmax=Nmax)
    fig, ax = plt.subplots(1, 1)
    ax.scatter(v[:, 0], v[:, 1])
    plt.show(block=True)
    plt.show()
    pass





def p_fun(x):

    s = th.cat([0.7*th.ones(d).unsqueeze(0), th.tensor([2]*(d//2) + [0.3]*(d-d//2)).unsqueeze(0), 0.7*th.ones(d).unsqueeze(0)],dim=0)
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

    s = th.cat([0.7 * th.ones(d).unsqueeze(0), th.tensor([2] * (d // 2) + [0.3] * (d - d // 2)).unsqueeze(0),0.7 * th.ones(d).unsqueeze(0)], dim=0)
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

p_obj = make_p_fun(5)


# Learning

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

        q_train = p_obj.g(X_train)
        q_test = p_obj.g(X_test)


        #gammaList = [1,0.5,0.3,0.25,0.2,0.1,0.05]
        #gammaList = [0.1,0.2,0.4,0.6,0.8,1]
        gammaList = [0.008,0.01,0.05,0.1,0.2,0.5,1,2,4,5,10]
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



def compute_everything(nmm_list,name,Nsamples,doubles_max,tol = 1e-2,Nbatches = 5,\
                       advancedNy=True,overwrite_indic = None,n_jobs = 1):
    for k in nmm_list:
        n,m_intermediate,m = k
        print(f'starting iteration with {n}, {m_intermediate} and {m}')
        if overwrite_indic == 'model':
            overwrite = True
        else:
            overwrite = False

        Nmax = int(doubles_max/m_intermediate**2/d)
        dico_train,dico_test = learn_model(n,m,name,advancedNy=advancedNy,m_intermediate = m_intermediate,Nmax = Nmax,overwrite = overwrite)
        alpha,X,g= dico_train['alpha'],dico_train['Ny'],dico_train['gamma']
        g = g*th.ones(1,d)
        p_model = CreateGaussianPsdModel1(alpha, X, g,Q, x=list(range(d)))

        #########
        # Sampling
        ######
        print('sampling...')
        #overwrite= True

        sample_name = f'{name}samples_{Nsamples}_n_{n}_m_{m}_d_{d}.pickle'


        try:
            with open(f'samples/{sample_name}', 'rb') as handle:
                if overwrite:
                    raise ValueError
                dico_samples = pickle.load(handle)
        except:

            Nmax = int(doubles_max / m ** 2 / d)
            samples_model = p_model.sample_from_hypercube(Nsamples, tol=tol, tolInt=1e-13, adaptive=None, n_jobs=n_jobs, Nmax=Nmax)
            Nmax = int(doubles_max / 2 ** 2 / d)
            if Nmax >= Nsamples:
                Nmax = None
            samples_true = p_obj.sample_from_hypercube(Nsamples,tol = 1e-5,tolInt =1e-13,Nmax =Nmax)
            grid_sampler = gridSampler(p_obj, Q, n=n)
            samples_bad = grid_sampler.sample(Nsamples)
            samples_random = mu_sampler(Nsamples)
            dico_samples = {'model':samples_model,'true' : samples_true,'grid':samples_bad,'random' : samples_random}

            with open(f'samples/{sample_name}', 'wb') as handle:
                pickle.dump(dico_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

        samples_model = dico_samples['model']
        samples_true = dico_samples['true']
        samples_bad = dico_samples['grid']
        samples_random = dico_samples['random']
        print(samples_model.size())
        print(samples_true.size())
        print(samples_bad.size())
        print(samples_random.size())
        if overwrite_indic== 'mmd':
            overwrite = True
        #Nsamples = 10000
        #overwrite = True



        def MMD(X1,X2,eta,Nbatches):
            g = eta*th.ones(1,d)
            n = X1.size(0)
            m = X2.size(0)
            n1 = n//Nbatches
            m2 = m//Nbatches
            res = []
            for i in range(Nbatches):
                X11 = X1[i*n1:(i+1)*n1,:]
                X22 = X2[i * m2:(i + 1) * m2, :]
                print('start_MMD')
                t1 = gaussKern(X11,X11,g).sum().sum().item()/n1**2
                print('first_gauss_kernel')
                t2 = gaussKern(X22,X22,g).sum().sum().item()/m2**2
                print('second_gauss_kernel')
                t3 = gaussKern(X11,X22,g).sum().sum().item()/(n1*m2)
                print('third_gauss_kernel')
                res.append(math.sqrt(t1 + t2 - 2*t3))
            res = th.tensor(res)
            return res
        eta_list = [0.1,0.2,0.5,1,2,5,10]
        for eta in eta_list:
            mmd_name = f'mmd_results/{name}_mmd_samples_{Nsamples}_n_{n}_m_{m}_d_{d}_eta_{eta}.pickle'

            try:
                with open(mmd_name, 'rb') as handle:
                    if overwrite:
                        raise ValueError
                    dico_mmd = pickle.load(handle)
            except:
                mmd_model= MMD(samples_true,samples_model,eta,Nbatches)
                mmd_bad = MMD(samples_bad,samples_true,eta,Nbatches)
                mmd_random = MMD(samples_random, samples_true, eta,Nbatches)
                nOneBatch = Nsamples//Nbatches
                mmd_true = MMD(samples_true,\
                               th.cat([samples_true[nOneBatch:nOneBatch*Nbatches,:],samples_true[:nOneBatch]]),eta,Nbatches)
                dico_mmd = {'mmd_good':mmd_model,'mmd_bad':mmd_bad,\
                            'mmd_random' : mmd_random,'mmd_true':mmd_true}
                with open(mmd_name, 'wb') as handle:
                    pickle.dump(dico_mmd, handle, protocol=pickle.HIGHEST_PROTOCOL)
            mmd_model,mmd_bad,mmd_random = dico_mmd['mmd_good'],dico_mmd['mmd_bad'],dico_mmd['mmd_random']
            mean_model = mmd_model.mean().item()
            error_model = (mmd_model - mean_model).pow(2).mean().sqrt().item()
            mean_bad = mmd_bad.mean().item()
            error_bad = (mmd_bad - mean_bad).pow(2).mean().sqrt().item()
            mean_random = mmd_random.mean().item()
            error_random = (mmd_random - mean_random).pow(2).mean().sqrt().item()
            print(f'for eta = {eta} : mmd good  between : {mean_model-error_model,mean_model+error_model},\
             mmd bad : {mean_bad-error_bad,mean_bad+error_bad}, mmd random : {mean_random-error_random,mean_random+error_random}')


#name = 'experiment_3_test'
name = 'experiment_6_true'
nmm_list = [[1000,50,50],[3600,50,50],[10000,50,50],[36000,50,50],[100000,50,50]]
#nmm_list = [[1000,50,50],[3600,100,100],[10000,150,150],[36000,200,200],[100000,300,300]]
#nmm_list = [[1000,100,50],[10000,100,50],[100000,100,50]]
overwrite_indic= None
#advancedNy = False
d=5
doubles_max = 100*100*1000*5
Nbatches = 5
Nsamples = 50000
n_jobs = 5

compute_everything(nmm_list,name,Nsamples,doubles_max,tol = 1e-2,Nbatches = Nbatches,\
                       advancedNy=True,overwrite_indic = overwrite_indic,n_jobs = n_jobs)