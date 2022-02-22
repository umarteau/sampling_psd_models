from gaussian_psd_model import findBestHellinger as findSol
from gaussian_psd_model import *
import pickle
import math

th.set_default_dtype(th.float64)
cm = plt.cm.get_cmap('Blues')



##################
# Getting the info for the MMD plot
#################


nm_list = [[1000,50],[3600,50],[10000,50],[36000,50],[100000,50]]
#nm_list = [[1000,50],[3600,100],[10000,150],[36000,200],[100000,300]]
#nm_list = [[1000,50],[10000,50],[100000,50]]
eta_list= [0.1,0.2,0.5,1,2,10]
#name_basic = 'experiment_3_test'
name_basic = 'experiment_6_true'
Nsamples = 50000
#d=2
d=5
islog = False
dico_eta = {}
for eta in eta_list:
    names_eta = []
    x = []
    y_good,yerr_good = [],[]
    y_bad,yerr_bad = [],[]
    y_random,yerr_random = [],[]
    y_true,yerr_true = [],[]
    for i in range(len(nm_list)):
        n,m = nm_list[i]
        x.append(n)
        neta = f'mmd_results/{name_basic}_mmd_samples_{Nsamples}_n_{n}_m_{m}_d_{d}_eta_{eta}.pickle'
        with open(neta, 'rb') as handle:
           dico_mmd = pickle.load(handle)
           mmd_good, mmd_bad, mmd_random,mmd_true = dico_mmd['mmd_good'], dico_mmd['mmd_bad'], dico_mmd['mmd_random'],dico_mmd['mmd_true']
        if islog:
            mmd_good = mmd_good.log()
            mmd_bad = mmd_bad.log()
            mmd_random = mmd_random.log()
            mmd_true = mmd_true.log()

        mean_good = mmd_good.mean().item()
        sd_good = math.sqrt((mmd_good - mean_good).pow(2).mean().item())

        mean_bad = mmd_bad.mean().item()
        sd_bad = math.sqrt((mmd_bad - mean_bad).pow(2).mean().item())

        mean_random = mmd_random.mean().item()
        sd_random = math.sqrt((mmd_random - mean_random).pow(2).mean().item())

        mean_true = mmd_true.mean().item()
        sd_true = math.sqrt((mmd_true - mean_true).pow(2).mean().item())

        if islog:
            mean_good = math.exp(mean_good)
            mean_bad = math.exp(mean_bad)
            mean_random = math.exp(mean_random)
            mean_true = math.exp(mean_true)
            sd_good = math.exp(sd_good)
            sd_bad = math.exp(sd_bad)
            sd_random = math.exp(sd_random)
            sd_true = math.exp(sd_true)


        y_good.append(mean_good)

        yerr_good.append(sd_good)
        y_bad.append(mean_bad)
        yerr_bad.append(sd_bad)
        y_random.append(mean_random)
        yerr_random.append(sd_random)
        y_true.append(mean_true)
        yerr_true.append(sd_true)
    dico_eta[eta] = [x,y_good,yerr_good,y_bad,yerr_bad,y_random,yerr_random,y_true,yerr_true]

capsize = 3
for eta in eta_list:


    x,y_good,yerr_good,y_bad,yerr_bad,y_random,yerr_random,y_true,yerr_true= dico_eta[eta]
    fig,ax = plt.subplots(1,1)
    ax.errorbar(x,y_good,yerr=yerr_good,capsize=capsize,label = 'PSD model')
    ax.errorbar(x,y_bad,yerr=yerr_bad,capsize=capsize,label = 'grid')
    ax.errorbar(x,y_random,yerr=yerr_random,capsize = capsize,label = 'uniform')
    ax.errorbar(x, y_true, yerr=yerr_true, capsize=capsize, label='noise level')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('evaluation points')
    ax.set_ylabel('MMD distance $d_{\eta}(p,p_*)$')
    ax.set_title(f'MMD distance for $\eta = {eta}$ in dimension $d = {d}$')
    ax.legend(loc=1)
    #fig.savefig('plots/MMD_d_2.pdf',format = 'pdf',dpi = 1000)
    plt.show(block=False)


##################
# Other plots : objective, learnt and samples
#################


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
    alpha = th.tensor([0.08, -0.4, 0.4])

    X = ((x.unsqueeze(0)-mu.unsqueeze(1))/math.sqrt(2)/s[:,None,None])
    res = -X.pow(2).sum(-1)
    res = res.exp()
    res = res*alpha[:,None]
    return res.sum(0)



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





name = 'experiment_2_true'
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

sample_name = f'{name}samples_{Nsamples}_n_{n}_m_{m}_d_{d}.pickle'

try:
    with open(f'samples/{sample_name}', 'rb') as handle:
        if overwrite:
            raise ValueError
        dico_samples = pickle.load(handle)
except:
    Nmax = int(doubles_max / m ** 2 / d)
    samples_model = p.sample_from_hypercube(Nsamples, tol=tol, tolInt=1e-13, adaptive=None, n_jobs=n_jobs,
                                                  Nmax=Nmax)

    dico_samples = {'model': samples_model}

    with open(f'samples/{sample_name}', 'wb') as handle:
        pickle.dump(dico_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

samples_model = dico_samples['model']
m = X.size(0)
savename = 'experiment_learning_4'

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
            #fig.colorbar(sc, ax=ax0)

        p_pref_grid = p(X_grid)
        p_grid = p_fun(X_grid)

        fontsize = 20
        fig, ax = plt.subplots(2, 2,figsize=(10,8))

        ax[0,0].set_title('Ground truth',fontsize =fontsize)
        plot_contour(fig, ax[0,0], p_grid.max(), p_grid)
        ax[0,0].tick_params(axis='x', which='both', bottom=None, top=None, labelbottom=False, labeltop=False)
        ax[0,0].tick_params(axis='y', which='both', left=None, right=None, labelleft=False, labelright=False)
        ax[0,1].set_title('Approximation',fontsize=fontsize)
        plot_contour(fig, ax[0,1], p_grid.max(), p_pref_grid)
        ax[0, 1].tick_params(axis='x', which='both', bottom=None, top=None, labelbottom=False, labeltop=False)
        ax[0, 1].tick_params(axis='y', which='both', left=None, right=None, labelleft=False, labelright=False)
        ax[1,0].scatter(samples_model[:, 0], samples_model[:, 1],marker = '+')
        ax[1,0].set_title('Samples',fontsize=fontsize)
        ax[1, 0].tick_params(axis='x', which='both', bottom=None, top=None, labelbottom=False, labeltop=False)
        ax[1, 0].tick_params(axis='y', which='both', left=None, right=None, labelleft=False, labelright=False)
        x, y_good, yerr_good, y_bad, yerr_bad, y_random, yerr_random, y_true, yerr_true = dico_eta[eta]
        lw = 2
        ax[1,1].errorbar(x, y_good, yerr=yerr_good, capsize=capsize, label='PSD model',lw=lw)
        ax[1,1].errorbar(x, y_bad, yerr=yerr_bad, capsize=capsize, label='grid',lw = lw)
        ax[1,1].errorbar(x, y_random, yerr=yerr_random, capsize=capsize, label='uniform',lw=lw)
        ax[1,1].errorbar(x, y_true, yerr=yerr_true, capsize=capsize, label='noise level',lw=lw)
        ax[1,1].set_xscale('log')
        ax[1,1].set_yscale('log')
        ax[1,1].set_xlabel('$n$')
        ax[1,1].set_title(f'MMD distance for $d = 5$',fontsize = fontsize)
        ax[1,1].legend(loc=1)
        ax[1,1].set_ylim(ymax=1)

        plt.tight_layout()
        fig.savefig(f'plots/{savename}.pdf',format = 'pdf',dpi=1000)
        plt.show(block=False)

    else:
        pass

plothere()




