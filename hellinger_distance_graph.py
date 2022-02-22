from gaussian_psd_model import findBestHellinger as findSol
from gaussian_psd_model import *
import pickle
import math

th.set_default_dtype(th.float64)
cm = plt.cm.get_cmap('Blues')


################################
# first part of the plot
############################


nlist = [2000,5000,10000,50000,100000,500000]
mlist = [50,100,200,500,1000]
n_evs = 5000000
distance = 'hellinger'
Nbatches = 5
d= 10


name_basic = 'experiment_12_test'
islog = False

dico_m = {}
for m in mlist:
    x = []
    y_model,yerr_model = [],[]
    for n in nlist:
        x.append(n)
        filename = f'{distance}_results/{name_basic}_distance_{distance}_n_evals_{n_evs}_n_batch_{Nbatches}_n_{n}_m_{m}_d_{d}.pickle'

        with open(filename, 'rb') as handle:
            dico_distance = pickle.load(handle)
            d_model = dico_distance['distance_model']

        if islog:
            d_model = d_model.log()

        mean_model = d_model.mean().item()
        sd_model = math.sqrt((d_model - mean_model).pow(2).mean().item())

        if islog:
            mean_model = math.exp(mean_model)
            sd_model = math.exp(sd_model)



        y_model.append(mean_model)
        yerr_model.append(sd_model)

    dico_m[m] = [x,y_model,yerr_model]

################################
# second part of the plot
############################

nlist = [2000,5000,10000,50000,100000,500000]
mlist = [50,100,200,500,1000]
n_evs = 5000000
distance = 'hellinger'
Nbatches = 5
d = 2

name_basic = 'experiment_13_test'
islog = False

dico_m_2 = {}
for m in mlist:
    x = []
    y_model, yerr_model = [], []
    for n in nlist:
        x.append(n)
        filename = f'{distance}_results/{name_basic}_distance_{distance}_n_evals_{n_evs}_n_batch_{Nbatches}_n_{n}_m_{m}_d_{d}.pickle'

        with open(filename, 'rb') as handle:
            dico_distance = pickle.load(handle)
            d_model = dico_distance['distance_model']

        if islog:
            d_model = d_model.log()

        mean_model = d_model.mean().item()
        sd_model = math.sqrt((d_model - mean_model).pow(2).mean().item())

        if islog:
            mean_model = math.exp(mean_model)
            sd_model = math.exp(sd_model)

        y_model.append(mean_model)
        yerr_model.append(sd_model)

    dico_m_2[m] = [x, y_model, yerr_model]




capsize = 3
savefile = 'experiment_hellinger_distance_1'
lw = 2

def plothere():
    if d == 2:
        fig, ax = plt.subplots(1,2, figsize=(10, 4))
        fontsize = 20
        for m,value in dico_m.items():
            x, y_model, yerr_model = value
            ax[0].errorbar(x, y_model, yerr=yerr_model, capsize=capsize, label=f'$m=${m}',lw=lw)
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[0].set_xlabel('$n$')
        ax[0].set_title(f'Hellinger distance, $d=10$',fontsize = fontsize)
        ax[0].legend(loc=1)
        #ax[1,1].set_ylim(ymax=1)

        for m,value in dico_m_2.items():
            x, y_model, yerr_model = value
            ax[1].errorbar(x, y_model, yerr=yerr_model, capsize=capsize, label=f'$m=${m}',lw=lw)
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[1].set_xlabel('$n$')
        ax[1].set_title(f'Hellinger distance, $d=2$',fontsize = fontsize)
        ax[1].legend(loc=1)
        #ax[1,1].set_ylim(ymax=1)

        plt.tight_layout()
        fig.savefig(f'plots/{savefile}.pdf',format = 'pdf',dpi=1000)
        plt.show(block=False)

    else:
        pass

plothere()




