import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600
import os
import numpy as np

results_dir = os.path.join(os.getcwd(),'HingePlaceMRI_Results/Sec-5-1-4')
results_dirs = [os.path.join(results_dir,'DiffTarget%d_ver2'%(i+1)) for i in range(4)]
Isafe = [np.linspace(4,5,5), np.linspace(10,13,5), np.linspace(8,10,5),np.linspace(6,7.5,5)]
rel_inc = np.array([np.load(os.path.join(directory,'HPvsSP_RelInc.npy')) for directory in results_dirs])
rel_inc_HP2 = np.array([np.load(os.path.join(directory,'HPvsSP2_RelInc.npy')) for directory in results_dirs])
marker = ['^', 'o']
linestyle = ['dashed', 'dotted', 'dotted', (0,(3,1,1,1,1,1))]
color = ['C1', 'C2', 'C3', 'C4']
Itot_mul = [2,8]

for i in range(rel_inc.shape[0]):
    for ii in range(rel_inc.shape[1]):
        plt.plot(Isafe[i], rel_inc[i,ii,:], marker=marker[0], linestyle=linestyle[ii], color=color[i], label=r'T%d $I_{tot}^{mul}$:%d p=1 '%(i+1, Itot_mul[ii]))
        plt.plot(Isafe[i], rel_inc_HP2[i,ii,:], marker=marker[1], linestyle=linestyle[ii], color=color[i], label=r'T%d, $I_{tot}^{mul}$:%d p=2'%(i+1, Itot_mul[ii]))
# plt.legend(fontsize=17, ncols=2, loc="center right")
plt.xlabel(r'$I_{safe}$ (mA)', fontsize=21)
plt.ylabel('% Decrease Activation', fontsize=21)
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
plt.xlim(xmax=19)
plt.tight_layout()
plt.savefig(os.path.join(results_dir,'HPvSP_RelIncDiffTarget.png'))
plt.show()