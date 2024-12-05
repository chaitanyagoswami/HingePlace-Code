import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600

cwd = os.getcwd()
base_dir = os.path.join(cwd,'HingePlace/Results/Sec-5-2-5')
directories = [os.path.join(base_dir, 'BaselineOptDir%d'%(i+1)) for i in range(4)]
rel_inc = np.array([np.load(os.path.join(directory,'HPvsSP_RelInc.npy')) for directory in directories])
I_safe = np.array([[130,140,150,160,170],[170,180,190,200,220],[130,140,150,160,170],[170,180,190,200,220]])
marker = ['^','o']
linestyle = ['dashdot', 'dashed', 'dotted', (0,(3,1,1,1,1,1))]
color = ['darkgreen', 'blue', 'darkorange', 'fuchsia']
Itot_mul = [2,6]
for i in range(rel_inc.shape[0]):
    for ii in range(rel_inc.shape[1]):
        plt.plot(I_safe[i], rel_inc[i,ii,:], marker=marker[ii], linestyle=linestyle[i], color=color[i], label=r'TP-%d, $I_{tot}^{mul}$=%d'%(i+1,Itot_mul[ii]))
plt.legend(ncols=1, fontsize=17, loc='center right')
plt.xlabel(r'$I_{safe} (mA)$', fontsize=21)
plt.ylabel('% Decrease Activation', fontsize=21)
plt.xlim(xmax=np.max(I_safe)*1.4)
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
plt.tight_layout()
plt.savefig(os.path.join(base_dir,'HPvsSP_RelIncDiffTarget_p1.png'))
plt.show()

cwd = os.getcwd()
base_dir = os.path.join(cwd,'HingePlace/Results')
directories = [os.path.join(base_dir, 'BaselineOptDir%d'%(i+1)) for i in range(4)]
rel_inc = np.array([np.load(os.path.join(directory,'HP2vsSP_RelInc.npy')) for directory in directories])
I_safe = np.array([[130,140,150,160,170],[170,180,190,200,220],[130,140,150,160,170],[170,180,190,200,220]])
marker = ['^','o']
linestyle = ['dashdot', 'dashed', 'dotted', (0,(3,1,1,1,1,1))]
color = ['darkgreen', 'blue', 'darkorange', 'fuchsia']
Itot_mul = [2,6]
for i in range(rel_inc.shape[0]):
    for ii in range(rel_inc.shape[1]):
        plt.plot(I_safe[i], rel_inc[i,ii,:], marker=marker[ii], linestyle=linestyle[i], color=color[i], label=r'TP-%d, $I_{tot}^{mul}$=%d'%(i+1,Itot_mul[ii]))
plt.legend(ncols=1, fontsize=17, loc='center right')
plt.xlabel(r'$I_{safe} (mA)$', fontsize=21)
plt.ylabel('% Decrease Activation', fontsize=21)
plt.xlim(xmax=np.max(I_safe)*1.4)
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
plt.tight_layout()
plt.savefig(os.path.join(base_dir,'HPvsSP_RelIncDiffTarget_p2.png'))
plt.show()
