import numpy as np
import matplotlib.pyplot as plt
from gaussian_psd_model import findBestHellinger as findSol
from gaussian_psd_model import *
import pickle
import time
import math

th.set_default_dtype(th.float64)
cm = plt.cm.get_cmap('RdYlBu_r')




d = 10

Q = th.tensor([[-1.] * d, [1.] * d])


def mu_sampler(n):
    th.random.seed()
    return Q[0, :][None, :] + (Q[1, :] - Q[0, :])[None, :] * th.rand(n, d)


def mu_density(x):
    return th.ones(x.size(0)) / (Q[1, :] - Q[0, :]).prod()


def scatter_color(fig, ax0, vmax, Xtest, ptest):
    sc = ax0.scatter(Xtest[:, 0], Xtest[:, 1], c=ptest, vmin=0, vmax=vmax, cmap=cm)
    fig.colorbar(sc, ax=ax0)


def make_p_fun(d):
    X_p = th.ones(2, d)
    X_p[1, :] *= -1
    gamma_p = 4 * 0.25 * th.ones(1, d) / d
    alpha_p = th.tensor([1, -1], dtype=th.float64)
    Q_p = th.ones(2, d)
    Q_p[0, :] *= -1
    p_obj = CreateGaussianPsdModel1(alpha_p, X_p, gamma_p, Q_p, x=list(range(d)))
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
    s = th.cat([0.7 * th.ones(d).unsqueeze(0), th.tensor([2] * (d // 2) + [0.3] * (d - d // 2)).unsqueeze(0),
                0.7 * th.ones(d).unsqueeze(0)], dim=0)
    mu = th.cat([-th.ones(1, d), th.ones(1, d), th.ones(1, d)], 0)
    alpha = th.tensor([0.3, 0.4, 0.4])

    X = ((x.unsqueeze(0) - mu.unsqueeze(1)) / math.sqrt(2) / s.unsqueeze(1))
    res = -X.pow(2).sum(-1)
    res = res.exp()
    res = res * alpha[:, None]
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


p_obj = make_p_fun(d)


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

        q_train = p_obj.g(X_train)


        q_test = p_obj.g(X_test)

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


def compute_model_evaluations(nmm_list, name, n_evs, doubles_max,  Nbatches=5, \
                       advancedNy=True, overwrite_indic=None, base = 'uniform',distance = 'hellinger',n_jobs = 1):
    for k in nmm_list:
        n, m_intermediate, m = k
        print(f'starting iteration with {n}, {m_intermediate} and {m}')
        if overwrite_indic == 'model' or overwrite_indic == 'all':
            overwrite = True
        else:
            overwrite = False

        Nmax = int(doubles_max / m_intermediate)
        dico_train, dico_test = learn_model(n, m, name, advancedNy=advancedNy, m_intermediate=m_intermediate, Nmax=Nmax,
                                            overwrite=overwrite)
        alpha, X, g = dico_train['alpha'], dico_train['Ny'], dico_train['gamma']
        g = g * th.ones(1, d)
        p_model = CreateGaussianPsdModel1(alpha, X, g, Q, x=list(range(d)))

        evs_name = f'{name}evaluations_n_{n}_m_{m}_d_{d}_base_{base}.pickle'

        try:
            with open(f'model_evaluations/{evs_name}', 'rb') as handle:
                if overwrite:
                    raise ValueError
                dico_evs = pickle.load(handle)
                base_evs = dico_evs['model']
                base_evs_obj = dico_evs['objective']
                n_evs_previous = base_evs.size(0)
        except:
            n_evs_previous = 0
            base_evs = th.zeros(0,)
            base_evs_obj = th.zeros(0,)
        base_evs_big = th.stack([base_evs,base_evs_obj],dim=1)
        n_to_compute = n_evs - n_evs_previous

        if n_to_compute > 0:
            if base == 'uniform':
                X_evs = mu_sampler(n_to_compute)
            else:
                ''' TODO '''
            def small_evaluate(bounds):
                return th.cat([p_model(X_evs[bounds[0]:bounds[1]]),p_obj(X_evs[bounds[0]:bounds[1]])],dim=1)

            Nmax = int(doubles_max / m )
            q,r = n_to_compute//Nmax,n_to_compute%Nmax
            if r == 0:
                q -=1
            l_bounds = [(i*Nmax,(i+1)*Nmax) for i in range(q)] + [(q*Nmax,n_to_compute)]
            res_evs_big = Parallel(n_jobs=n_jobs)(delayed(small_evaluate)(bounds) for bounds in l_bounds)

            res_evs_big.append(base_evs_big)
            res_evs_big = th.cat(res_evs_big,dim =0)
            res_evs = res_evs_big[:,0]
            res_evs_obj = res_evs_big[:,1]

        else:
            res_evs = base_evs
            res_evs_obj = base_evs_obj
        res_evs = res_evs.clamp(min=0.)
        res_evs_obj = res_evs_obj.clamp(min = 0.)
        res_evs /= p_model.integrate().item()
        res_evs_obj /= p_obj.integrate().item()
        dico_evs = {'model' : res_evs,'objective' : res_evs_obj}

        if n_to_compute > 0 :
            with open(f'model_evaluations/{evs_name}', 'wb') as handle:
                pickle.dump(dico_evs, handle, protocol=pickle.HIGHEST_PROTOCOL)

        ################## look

        if overwrite_indic == 'dist' or overwrite_indic == 'all':
            overwrite = True

        # Nsamples = 10000
        # overwrite = True

        #########
        ###########
        if distance == 'hellinger':
            def dist(p1,p2,Nbatches):
                nn = p1.size(0)
                n_one_batch = nn//Nbatches
                dd = th.tensor([(p1[i*n_one_batch:(i+1)*n_one_batch].sqrt()-p2[i*n_one_batch:(i+1)*n_one_batch].sqrt()).pow(2).mean().sqrt().item() for i in range(Nbatches)])

                return dd
        elif distance == 'tv':
            def dist(p1,p2,Nbatches):
                nn = p1.size(0)
                n_one_batch = nn // Nbatches
                dd = th.tensor([(p1[i * n_one_batch:(i + 1) * n_one_batch] - p2[i * n_one_batch:(
                                                                                                                   i + 1) * n_one_batch]).abs().pow(
                    2).mean().item() for i in range(Nbatches)])
                return dd

        d_name = f'{distance}_results/{name}_distance_{distance}_n_evals_{n_evs}_n_batch_{Nbatches}_n_{n}_m_{m}_d_{d}.pickle'

        try:
            with open(d_name, 'rb') as handle:
                if overwrite:
                    raise ValueError
                dico_distance = pickle.load(handle)
        except:
            d_model = dist(res_evs, res_evs_obj, Nbatches)
            dico_distance = {'distance_model': d_model}
            with open(d_name, 'wb') as handle:
                pickle.dump(dico_distance, handle, protocol=pickle.HIGHEST_PROTOCOL)
        d_model = dico_distance['distance_model']
        mean_model = d_model.mean().item()
        error_model = (d_model - mean_model).pow(2).mean().sqrt().item()
        print(f'for our model, the distance to the target measure is between : {mean_model - error_model, mean_model + error_model}')
    pass


# name = 'experiment_3_test'
name = 'experiment_12_test'
#nmm_list = [[1000, 50, 50], [3600, 50, 50], [10000, 50, 50], [36000, 50, 50], [100000, 50, 50]]
#nmm_list = [[1000, 50, 50], [3600, 50, 50], [10000, 50, 50], [36000, 50, 50], [100000, 50, 50]]
# nmm_list = [[1000,50,50],[3600,100,100],[10000,150,150],[36000,200,200],[100000,300,300]]
nmm_list = []
nlist = [2000,5000,10000,50000,100000,500000]
mlist = [50,100,200,500,1000]
mmmlist = []
for n in nlist:
    for m in mlist:
        mmmlist.append([n,m,m])
overwrite_indic = None
advancedNy = False
doubles_max = 100 * 100 * 1000 * 5
Nbatches = 5
n_jobs = 5
n_evs = 5000000


compute_model_evaluations(mmmlist, name, n_evs, doubles_max, Nbatches=Nbatches, \
                   advancedNy=advancedNy, overwrite_indic=overwrite_indic)