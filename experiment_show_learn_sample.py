import numpy as np
import matplotlib.pyplot as plt
from gaussian_psd_model import findBestHellinger as findSol
from gaussian_psd_model import *
import pickle
import time
import math

th.set_default_dtype(th.float64)
cm = plt.cm.get_cmap('Blues')


class gridSampler:
    def __init__(self, p, Q, n_side=None, n=None):
        d = Q.size(1)
        if n_side is None:
            n_side = int(math.pow(n, 1 / d))
        n = math.pow(n_side, d)

        a = Q[0, :]
        b = Q[1, :]
        steps = (b - a) / n_side
        gridTensors = th.meshgrid([th.arange(a[i], b[i], steps[i]) for i in range(d)])
        gridTensor = th.cat([gt.unsqueeze(-1) for gt in gridTensors], dim=-1).view(-1, d)
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

    def sample(self, N):
        indices = th.multinomial(self.probas, N, replacement=True)
        vecs = self.gridTensor[indices, :]
        return vecs + self.steps[None, :] * th.rand(N, self.d)


d = 2

Q = th.tensor([[-3.] * d, [3.] * d])


def mu_sampler(n):
    return Q[0, :][None, :] + (Q[1, :] - Q[0, :])[None, :] * th.rand(n, d)


def mu_density(x):
    return th.ones(x.size(0)) / (Q[1, :] - Q[0, :]).prod()


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


def p_sample(N, Q):
    a = Q[0, :]
    b = Q[1, :]

    s = th.cat([0.7 * th.ones(d).unsqueeze(0), th.tensor([2] * (d // 2) + [0.3] * (d - d // 2)).unsqueeze(0),
                0.7 * th.ones(d).unsqueeze(0)], dim=0)
    mu = th.cat([-th.ones(1, d), th.ones(1, d), th.ones(1, d)], 0)
    alpha = th.tensor([0.3, 0.4, 0.4])

    c = (2 * math.pi * s.pow(2)).sqrt().prod(dim=-1)
    alpha_bis = alpha * c

    Naux = 0
    l = []
    while Naux < N:
        index_samples = th.multinomial(alpha_bis, N, replacement=True)
        s_samples = s[index_samples, :]
        mu_samples = mu[index_samples, :]
        new_samples = mu_samples + s_samples * th.randn(N, d)
        mask = (((new_samples < a[None, :]) + (new_samples > b[None, :])).sum(1) == 0)
        new_samples = new_samples[mask, :]
        N_new_samples = new_samples.size(0)
        Naux += N_new_samples
        l.append(new_samples)

    samples = th.cat(l)[:N, :]

    return samples


# Learning

def learn_model(n, m, name='experiment_1', advancedNy=False, m_intermediate=None, Nmax=None, overwrite=False):
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
        return dico_train, dico_test
    except:
        proportion = 0.5
        n_train = int(proportion * n)
        n_test = n - n_train

        X_train = mu_sampler(n_train)
        X_test = mu_sampler(n_test)
        mu_train = mu_density(X_train)
        mu_test = mu_density(X_test)

        q_train = p_fun(X_train).sqrt()
        q_test = p_fun(X_test).sqrt()

        # gammaList = [1,0.5,0.3,0.25,0.2,0.1,0.05]
        # gammaList = [0.1,0.2,0.4,0.6,0.8,1]
        gammaList = [0.008, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 4, 5, 10]
        laList = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-7, 1e-8, 1e-9]
        # laList = [1e-6]
        empirical = True
        # save = None
        if advancedNy:
            n_jobs = 5
            if Nmax is None:
                Nmax = m_intermediate // n_jobs

            Ny = find_Ny_from_sqrt(X_train, q_train, gammaList, laList, m_intermediate, X_test=X_test, q_test=q_test, \
                                   Q=Q, \
                                   mu_train=mu_train, mu_test=mu_test, tol=1e-14, \
                                   empirical=True, retrain=True, Nmax=Nmax, n_jobs=n_jobs)
            dico_train, dico_test = findSol(X_train, q_train, gammaList, laList, Ny=Ny, X_test=X_test \
                                            , q_test=q_test, Q=Q, \
                                            m_compression=m, mu_train=mu_train, mu_test=mu_test, tol=1e-14, \
                                            empirical=empirical, dir='models', save=f'{true_name}', retrain=True)
        else:
            dico_train, dico_test = findSol(X_train, q_train, gammaList, laList, Ny=m, X_test=X_test \
                                            , q_test=q_test, Q=Q, \
                                            m_compression=None, mu_train=mu_train, mu_test=mu_test, tol=1e-14, \
                                            empirical=empirical, dir='models', save=f'{true_name}', retrain=True)
        return dico_train, dico_test


name = 'experiment_12'
# overwrite = True
overwrite = False
n = 100000
m_intermediate = 1000
m = 200
Nmax = 60
advancedNy = True
# advancedNy = False


dico_train, dico_test = learn_model(n, m, name, advancedNy=advancedNy, m_intermediate=m_intermediate, Nmax=Nmax,
                                    overwrite=overwrite)

alpha, X, g = dico_train['alpha'], dico_train['Ny'], dico_train['gamma']
g = g * th.ones(1, d)
p = CreateGaussianPsdModel1(alpha, X, g, Q, x=list(range(d)))

m = X.size(0)


def plothere():
    if d == 2:
        n_grid_side = 300

        xx_grid = np.linspace(Q[0, 0], Q[1, 0], n_grid_side)
        yy_grid = np.linspace(Q[0, 1], Q[1, 1], n_grid_side)

        xxx_grid, yyy_grid = np.meshgrid(xx_grid, yy_grid)
        xx_grid, yy_grid = th.from_numpy(xxx_grid.flatten()), th.from_numpy(yyy_grid.flatten())

        X_grid = th.cat([xx_grid[:, None], yy_grid[:, None]], dim=1)

        def plot_contour(fig, ax0, vmax, pntest):
            sc = ax0.contourf(xxx_grid, yyy_grid, pntest.reshape(n_grid_side, n_grid_side), vmin=0, vmax=vmax,
                              levels=35,
                              cmap=cm)
            fig.colorbar(sc, ax=ax0)

        p_pref_grid = p(X_grid)
        p_grid = p_fun(X_grid)
        tol = 1e-4
        Nmax = 200
        v = p.sample_from_hypercube(1000, tol=tol, tolInt=1e-13, adaptive=None, n_jobs=5, Nmax=Nmax)

        fig, ax = plt.subplots(3, 1,figsize=(5,15))

        ax[0].set_title('target_density')
        plot_contour(fig, ax[0], p_grid.max(), p_grid)
        ax[1].set_title('learned_density')
        plot_contour(fig, ax[1], p_grid.max(), p_pref_grid)
        ax[2].scatter(v[:, 0], v[:, 1],marker = '+')
        fig.savefig('plots/experiment_learning.pdf',format = 'pdf',dpi=1000)
        plt.tight_layout()
        plt.show(block=False)

        fig, ax = plt.subplots(1, 1)
        ax.scatter(v[:, 0], v[:, 1])
        plt.show(block=True)

    else:
        pass

plothere()


#########
# Sampling
######
print('sampling...')
# overwrite= True
Nsamples = 10000
Nmax = 1000

sample_name = f'{name}samples_{Nsamples}_n_{n}_m_{m}_d_{d}.pickle'

try:
    with open(f'samples/{sample_name}', 'rb') as handle:
        if overwrite:
            raise ValueError
        dico_samples = pickle.load(handle)
except:
    adaptive = None
    tol = 1e-4
    v = p.sample_from_hypercube(Nsamples, tol=tol, tolInt=1e-13, adaptive=adaptive, n_jobs=6, Nmax=Nmax)
    samples_true = p_sample(Nsamples, Q)
    grid_sampler = gridSampler(p_fun, Q, n=n)
    samples_bad = grid_sampler.sample(Nsamples)
    samples_random = mu_sampler(Nsamples)
    dico_samples = {'model': v, 'true': samples_true, 'grid': samples_bad, 'random': samples_random}

    with open(f'samples/{sample_name}', 'wb') as handle:
        pickle.dump(dico_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Nsamples = 10000
# overwrite = True

v = dico_samples['model'][:Nsamples, :]
samples_true = dico_samples['true'][:Nsamples, :]
samples_bad = dico_samples['grid'][:Nsamples, :]
samples_random = dico_samples['random'][:Nsamples, :]


def plotthere():
    if d == 2:
        nshow = 1000
        vv = v[:nshow, :]
        plt.scatter(vv[:, 0], vv[:, 1])
        plt.show(block=False)
    else:
        pass


# _ = plotthere()

