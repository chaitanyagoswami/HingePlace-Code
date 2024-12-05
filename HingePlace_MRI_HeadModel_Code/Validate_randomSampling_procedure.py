from scipy.io import loadmat
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl
from HingePlace_optimizers import HingePlace_p1, SparsePlace
from sklearn.metrics import jaccard_score
mpl.rcParams['figure.dpi'] = 600

cwd = os.getcwd()
forward_mat_dir = os.path.join(cwd,'Forward_Matrix_MRI')
mni2mri = np.array([[1.0226,0.0007,-0.0020,92.2735], [0.0106,1.0132,-0.0193,126.9093], [0.0035,0.0033,0.9907,74.4404], [0,0,0,1.0000]])

ORIGIN_MRI_COORD = mni2mri@np.array([0,0,0,1]).reshape(-1,1)
ORIGIN_MRI_COORD = np.squeeze(np.round(ORIGIN_MRI_COORD)[:3])

#### Loading Forward Matrix
Aall = loadmat(os.path.join(forward_mat_dir,'Aall.mat'))
Aall = Aall['A_brain']
locations = loadmat(os.path.join(forward_mat_dir,'loc_all.mat'))
locations = locations['locs_brain']

## Reducing target matrix to radial-in matrix
directions_all = locations-ORIGIN_MRI_COORD.reshape(1,3)
directions_all = -1*directions_all/np.sqrt(np.sum(np.square(directions_all), axis=1)).reshape(-1,1)
Aall = directions_all[:,0].reshape(-1,1)*Aall[:,0,:]+directions_all[:,1].reshape(-1,1)*Aall[:,1,:]+directions_all[:,2].reshape(-1,1)*Aall[:,2,:]

#### Choosing Motor Cortex as target location
mni_target_loc = np.array([-48,-8,50])
target_loc = mni2mri@np.concatenate([mni_target_loc,[1]]).reshape(-1,1)
target_loc = np.round(target_loc)
target_loc = target_loc[:3]
dist_target = np.sqrt(np.sum(np.square(locations-target_loc.reshape(1,3)), axis=1))

#### Choosing Target Region
target_radius = 2
idx_focus = np.arange(locations.shape[0])[dist_target <= target_radius]
idx_cancel_all = np.arange(locations.shape[0])[dist_target>target_radius]
Af, Ac_all = Aall[idx_focus], Aall[idx_cancel_all]
Af = np.mean(Af, axis=0).reshape(1,-1)

#### Choosing Custom Cancel Matrices
target_cancel = 40
idx_cancel = np.arange(locations.shape[0])[dist_target<=target_cancel]
idx_cancel = np.setdiff1d(idx_cancel,idx_focus)

tmp_all = np.arange(locations.shape[0])[dist_target>target_cancel]

tmp = tmp_all[np.random.permutation(tmp_all.shape[0])[:int(0.1*tmp_all.shape[0])]]
idx_cancel10 = np.array(list(idx_cancel)+list(tmp))
Ac10 = Aall[idx_cancel10]

tmp = tmp_all[np.random.permutation(tmp_all.shape[0])[:int(0.25*tmp_all.shape[0])]]
idx_cancel25 = np.array(list(idx_cancel)+list(tmp))
Ac25 = Aall[idx_cancel25]

tmp = tmp_all[np.random.permutation(tmp_all.shape[0])[:int(0.5*tmp_all.shape[0])]]
idx_cancel50 = np.array(list(idx_cancel)+list(tmp))
Ac50 = Aall[idx_cancel50]

Edes = 1 ## V/m
Isafety = 3.75 ## mA
Itotal = 2*4*Isafety
Etol = [0.5*Edes]

HingePlace_MRI_cwd = os.path.join(cwd,'HingePlaceMRI_Results/AppxD-4')
if not os.path.exists(HingePlace_MRI_cwd):
    os.makedirs(HingePlace_MRI_cwd)

LOAD_FLAG = True
if not LOAD_FLAG:

    I_HP10 = HingePlace_p1(Jdes=Edes, Isafety=Isafety, Jtol=Etol, Itotal=Itotal, Af=Af, Ac=Ac10)
    I_SP10 = SparsePlace(Jdes=Edes, Isafety=Isafety, Itotal=Itotal, Af=Af, Ac=Ac10)
    np.save(os.path.join(HingePlace_MRI_cwd,'HP_I10.npy'), I_HP10)
    np.save(os.path.join(HingePlace_MRI_cwd,'SP_I10.npy'), I_SP10)

    I_HP25 = HingePlace_p1(Jdes=Edes, Isafety=Isafety, Jtol=Etol, Itotal=Itotal, Af=Af, Ac=Ac25)
    I_SP25 = SparsePlace(Jdes=Edes, Isafety=Isafety, Itotal=Itotal, Af=Af, Ac=Ac25)
    np.save(os.path.join(HingePlace_MRI_cwd,'HP_I25.npy'), I_HP25)
    np.save(os.path.join(HingePlace_MRI_cwd,'SP_I25.npy'), I_SP25)

    I_HP50 = HingePlace_p1(Jdes=Edes, Isafety=Isafety, Jtol=Etol, Itotal=Itotal, Af=Af, Ac=Ac50)
    I_SP50 = SparsePlace(Jdes=Edes, Isafety=Isafety, Itotal=Itotal, Af=Af, Ac=Ac50)
    np.save(os.path.join(HingePlace_MRI_cwd,'HP_I50.npy'), I_HP50)
    np.save(os.path.join(HingePlace_MRI_cwd,'SP_I50.npy'), I_SP50)

    I_HP = HingePlace_p1(Jdes=Edes, Isafety=Isafety, Jtol=Etol, Itotal=Itotal, Af=Af, Ac=Ac_all)
    I_SP = SparsePlace(Jdes=Edes, Isafety=Isafety, Itotal=Itotal, Af=Af, Ac=Ac_all)
    np.save(os.path.join(HingePlace_MRI_cwd,'HP_Iwhole.npy'), I_HP)
    np.save(os.path.join(HingePlace_MRI_cwd,'SP_Iwhole.npy'), I_SP)

I_HP = np.load(os.path.join(HingePlace_MRI_cwd, 'HP_Iwhole.npy'))
I_HP50 = np.load(os.path.join(HingePlace_MRI_cwd, 'HP_I50.npy'))
I_HP25 = np.load(os.path.join(HingePlace_MRI_cwd, 'HP_I25.npy'))
I_HP10 = np.load(os.path.join(HingePlace_MRI_cwd, 'HP_I10.npy'))

I_SP = np.load(os.path.join(HingePlace_MRI_cwd, 'SP_Iwhole.npy'))
I_SP50 = np.load(os.path.join(HingePlace_MRI_cwd, 'SP_I50.npy'))
I_SP25 = np.load(os.path.join(HingePlace_MRI_cwd, 'SP_I25.npy'))
I_SP10 = np.load(os.path.join(HingePlace_MRI_cwd, 'SP_I10.npy'))

E_HP = np.squeeze(np.matmul(Aall,I_HP.reshape(-1,1)))
E_HP10 = np.squeeze(np.matmul(Aall,I_HP10.reshape(-1,1)))
E_HP25 = np.squeeze(np.matmul(Aall,I_HP25.reshape(-1,1)))
E_HP50 = np.squeeze(np.matmul(Aall,I_HP50.reshape(-1,1)))

E_SP = np.squeeze(np.matmul(Aall,I_SP.reshape(-1,1)))
E_SP10 = np.squeeze(np.matmul(Aall,I_SP10.reshape(-1,1)))
E_SP25 = np.squeeze(np.matmul(Aall,I_SP25.reshape(-1,1)))
E_SP50 = np.squeeze(np.matmul(Aall,I_SP50.reshape(-1,1)))

##### Jaccard Index of the Activated region
print("Jaccard Score 10%:",jaccard_score(np.abs(E_HP)>0.4, np.abs(E_HP10)>0.4))
print("Jaccard Score 25%:",jaccard_score(np.abs(E_HP)>0.4, np.abs(E_HP25)>0.4))
print("Jaccard Score 50%:",jaccard_score(np.abs(E_HP)>0.4, np.abs(E_HP50)>0.4))
print("Jaccard Score 10% SP:",jaccard_score(np.abs(E_SP)>0.4, np.abs(E_SP10)>0.4))
print("Jaccard Score 25% SP:",jaccard_score(np.abs(E_SP)>0.4, np.abs(E_SP25)>0.4))
print("Jaccard Score 50% SP:",jaccard_score(np.abs(E_SP)>0.4, np.abs(E_SP50)>0.4))


fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
color_map = cm.ScalarMappable(cmap='jet')
alpha_scale = E_HP10.copy()
start_transparency = 0.1
alpha_scale = (alpha_scale - np.min(alpha_scale)) / (np.max(alpha_scale) - np.min(alpha_scale)) * (1 - start_transparency) + start_transparency
rgba = color_map.to_rgba(x=E_HP10, norm=True)
ax.scatter(locations[:, 0], locations[:, 1], locations[:, 2], c=rgba)
ax.set_xlabel('X-axis (mm)', fontsize=20, labelpad=20)
ax.set_ylabel('Y-axis (mm)', fontsize=20, labelpad=4)
ax.set_zlabel('Z-axis (mm)', fontsize=20, labelpad=10)
ax.tick_params(axis='x', labelsize=18, pad=10)
ax.tick_params(axis='y', labelsize=18, pad=-2)
ax.tick_params(axis='z', labelsize=18)
cbar = plt.colorbar(mappable=color_map, ax=ax)
cbar.set_label(label=r'Vm$^{-1}$', size=20)
cbar.ax.tick_params(labelsize=18)
plt.tight_layout()
ax.view_init(40,180)
plt.savefig(os.path.join(HingePlace_MRI_cwd,'HP10.png'))
plt.close()

fig1_activ = plt.figure()
ax = fig1_activ.add_subplot(111, projection='3d')
ax.scatter(locations[np.abs(E_HP10) > 0.4, 0], locations[np.abs(E_HP10) > 0.4, 1], locations[np.abs(E_HP10) > 0.4, 2], alpha=1, linewidths=0, edgecolor=None, c='red', s=30)
ax.set_ylabel('Y-axis (mm)', fontsize=20, labelpad=20)
ax.set_zlabel('Z-axis (mm)', fontsize=20, labelpad=15)
ax.tick_params(axis='x', labelsize=18, pad=10, which='both', bottom=False, top=False, labelbottom=False, labeltop=False, labelright=False, labelleft=False)
ax.tick_params(axis='y', labelsize=18, pad=-2)
ax.tick_params(axis='z', labelsize=18)
plt.tight_layout()
ax.view_init(0, 180)
plt.savefig(os.path.join(HingePlace_MRI_cwd,'HP10_activ.png'))
plt.close()

fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
color_map = cm.ScalarMappable(cmap='jet')
alpha_scale = E_HP25.copy()
start_transparency = 0.1
alpha_scale = (alpha_scale - np.min(alpha_scale)) / (np.max(alpha_scale) - np.min(alpha_scale)) * (1 - start_transparency) + start_transparency
rgba = color_map.to_rgba(x=E_HP25, norm=True)
ax.scatter(locations[:, 0], locations[:, 1], locations[:, 2], c=rgba)
ax.set_xlabel('X-axis (mm)', fontsize=20, labelpad=20)
ax.set_ylabel('Y-axis (mm)', fontsize=20, labelpad=4)
ax.set_zlabel('Z-axis (mm)', fontsize=20, labelpad=10)
ax.tick_params(axis='x', labelsize=18, pad=10)
ax.tick_params(axis='y', labelsize=18, pad=-2)
ax.tick_params(axis='z', labelsize=18)
cbar = plt.colorbar(mappable=color_map, ax=ax)
cbar.set_label(label=r'Vm$^{-1}$', size=20)
cbar.ax.tick_params(labelsize=18)
plt.tight_layout()
ax.view_init(40,180)
plt.savefig(os.path.join(HingePlace_MRI_cwd,'HP25.png'))
plt.close()

fig2_activ = plt.figure()
ax = fig2_activ.add_subplot(111, projection='3d')
ax.scatter(locations[np.abs(E_HP25) > 0.4, 0], locations[np.abs(E_HP25) > 0.4, 1], locations[np.abs(E_HP25) > 0.4, 2], alpha=1, linewidths=0, edgecolor=None, c='red', s=30)
ax.set_ylabel('Y-axis (mm)', fontsize=20, labelpad=20)
ax.set_zlabel('Z-axis (mm)', fontsize=20, labelpad=15)
ax.tick_params(axis='x', labelsize=18, pad=10, which='both', bottom=False, top=False, labelbottom=False, labeltop=False, labelright=False, labelleft=False)
ax.tick_params(axis='y', labelsize=18, pad=-2)
ax.tick_params(axis='z', labelsize=18)
plt.tight_layout()
ax.view_init(0, 180)
plt.savefig(os.path.join(HingePlace_MRI_cwd,'HP25_activ.png'))
plt.close()

fig3 = plt.figure()
ax = fig3.add_subplot(111, projection='3d')
color_map = cm.ScalarMappable(cmap='jet')
alpha_scale = E_HP50.copy()
start_transparency = 0.1
alpha_scale = (alpha_scale - np.min(alpha_scale)) / (np.max(alpha_scale) - np.min(alpha_scale)) * (1 - start_transparency) + start_transparency
rgba = color_map.to_rgba(x=E_HP50, norm=True)
ax.scatter(locations[:, 0], locations[:, 1], locations[:, 2], c=rgba)
ax.set_xlabel('X-axis (mm)', fontsize=20, labelpad=20)
ax.set_ylabel('Y-axis (mm)', fontsize=20, labelpad=4)
ax.set_zlabel('Z-axis (mm)', fontsize=20, labelpad=10)
ax.tick_params(axis='x', labelsize=18, pad=10)
ax.tick_params(axis='y', labelsize=18, pad=-2)
ax.tick_params(axis='z', labelsize=18)
cbar = plt.colorbar(mappable=color_map, ax=ax)
cbar.set_label(label=r'Vm$^{-1}$', size=20)
cbar.ax.tick_params(labelsize=18)
plt.tight_layout()
ax.view_init(40,180)
plt.savefig(os.path.join(HingePlace_MRI_cwd,'HP50.png'))
plt.close()

fig3_activ = plt.figure()
ax = fig3_activ.add_subplot(111, projection='3d')
ax.scatter(locations[np.abs(E_HP50) > 0.4, 0], locations[np.abs(E_HP50) > 0.4, 1], locations[np.abs(E_HP50) > 0.4, 2], alpha=1, linewidths=0, edgecolor=None, c='red', s=30)
ax.set_ylabel('Y-axis (mm)', fontsize=20, labelpad=20)
ax.set_zlabel('Z-axis (mm)', fontsize=20, labelpad=15)
ax.tick_params(axis='x', labelsize=18, pad=10, which='both', bottom=False, top=False, labelbottom=False, labeltop=False, labelright=False, labelleft=False)
ax.tick_params(axis='y', labelsize=18, pad=-2)
ax.tick_params(axis='z', labelsize=18)
plt.tight_layout()
ax.view_init(0, 180)
plt.savefig(os.path.join(HingePlace_MRI_cwd,'HP50_activ.png'))
plt.close()

fig4 = plt.figure()
ax = fig4.add_subplot(111, projection='3d')
color_map = cm.ScalarMappable(cmap='jet')
alpha_scale = E_HP.copy()
start_transparency = 0.1
alpha_scale = (alpha_scale - np.min(alpha_scale)) / (np.max(alpha_scale) - np.min(alpha_scale)) * (1 - start_transparency) + start_transparency
rgba = color_map.to_rgba(x=E_HP, norm=True)
ax.scatter(locations[:, 0], locations[:, 1], locations[:, 2], c=rgba)
ax.set_xlabel('X-axis (mm)', fontsize=20, labelpad=20)
ax.set_ylabel('Y-axis (mm)', fontsize=20, labelpad=4)
ax.set_zlabel('Z-axis (mm)', fontsize=20, labelpad=10)
ax.tick_params(axis='x', labelsize=18, pad=10)
ax.tick_params(axis='y', labelsize=18, pad=-2)
ax.tick_params(axis='z', labelsize=18)
cbar = plt.colorbar(mappable=color_map, ax=ax)
cbar.set_label(label=r'Vm$^{-1}$', size=20)
cbar.ax.tick_params(labelsize=18)
plt.tight_layout()
ax.view_init(40,180)
plt.savefig(os.path.join(HingePlace_MRI_cwd,'HP.png'))
plt.close()

fig4_activ = plt.figure()
ax = fig4_activ.add_subplot(111, projection='3d')
ax.scatter(locations[np.abs(E_HP) > 0.4, 0], locations[np.abs(E_HP) > 0.4, 1], locations[np.abs(E_HP) > 0.4, 2], alpha=1, linewidths=0, edgecolor=None, c='red', s=30)
ax.set_ylabel('Y-axis (mm)', fontsize=20, labelpad=20)
ax.set_zlabel('Z-axis (mm)', fontsize=20, labelpad=15)
ax.tick_params(axis='x', labelsize=18, pad=10, which='both', bottom=False, top=False, labelbottom=False, labeltop=False, labelright=False, labelleft=False)
ax.tick_params(axis='y', labelsize=18, pad=-2)
ax.tick_params(axis='z', labelsize=18)
plt.tight_layout()
ax.view_init(0, 180)
plt.savefig(os.path.join(HingePlace_MRI_cwd,'HP_activ.png'))
plt.close()

fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
color_map = cm.ScalarMappable(cmap='jet')
alpha_scale = E_SP10.copy()
start_transparency = 0.1
alpha_scale = (alpha_scale - np.min(alpha_scale)) / (np.max(alpha_scale) - np.min(alpha_scale)) * (1 - start_transparency) + start_transparency
rgba = color_map.to_rgba(x=E_SP10, norm=True)
ax.scatter(locations[:, 0], locations[:, 1], locations[:, 2], c=rgba)
ax.set_xlabel('X-axis (mm)', fontsize=20, labelpad=20)
ax.set_ylabel('Y-axis (mm)', fontsize=20, labelpad=4)
ax.set_zlabel('Z-axis (mm)', fontsize=20, labelpad=10)
ax.tick_params(axis='x', labelsize=18, pad=10)
ax.tick_params(axis='y', labelsize=18, pad=-2)
ax.tick_params(axis='z', labelsize=18)
cbar = plt.colorbar(mappable=color_map, ax=ax)
cbar.set_label(label=r'Vm$^{-1}$', size=20)
cbar.ax.tick_params(labelsize=18)
plt.tight_layout()
ax.view_init(40,180)
plt.savefig(os.path.join(HingePlace_MRI_cwd,'SP10.png'))
plt.close()

fig1_activ = plt.figure()
ax = fig1_activ.add_subplot(111, projection='3d')
ax.scatter(locations[np.abs(E_SP10) > 0.4, 0], locations[np.abs(E_SP10) > 0.4, 1], locations[np.abs(E_SP10) > 0.4, 2], alpha=1, linewidths=0, edgecolor=None, c='red', s=30)
ax.set_ylabel('Y-axis (mm)', fontsize=20, labelpad=20)
ax.set_zlabel('Z-axis (mm)', fontsize=20, labelpad=15)
ax.tick_params(axis='x', labelsize=18, pad=10, which='both', bottom=False, top=False, labelbottom=False, labeltop=False, labelright=False, labelleft=False)
ax.tick_params(axis='y', labelsize=18, pad=-2)
ax.tick_params(axis='z', labelsize=18)
plt.tight_layout()
ax.view_init(0, 180)
plt.savefig(os.path.join(HingePlace_MRI_cwd,'SP10_activ.png'))
plt.close()

fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
color_map = cm.ScalarMappable(cmap='jet')
alpha_scale = E_SP25.copy()
start_transparency = 0.1
alpha_scale = (alpha_scale - np.min(alpha_scale)) / (np.max(alpha_scale) - np.min(alpha_scale)) * (1 - start_transparency) + start_transparency
rgba = color_map.to_rgba(x=E_SP25, norm=True)
ax.scatter(locations[:, 0], locations[:, 1], locations[:, 2], c=rgba)
ax.set_xlabel('X-axis (mm)', fontsize=20, labelpad=20)
ax.set_ylabel('Y-axis (mm)', fontsize=20, labelpad=4)
ax.set_zlabel('Z-axis (mm)', fontsize=20, labelpad=10)
ax.tick_params(axis='x', labelsize=18, pad=10)
ax.tick_params(axis='y', labelsize=18, pad=-2)
ax.tick_params(axis='z', labelsize=18)
cbar = plt.colorbar(mappable=color_map, ax=ax)
cbar.set_label(label=r'Vm$^{-1}$', size=20)
cbar.ax.tick_params(labelsize=18)
plt.tight_layout()
ax.view_init(40,180)
plt.savefig(os.path.join(HingePlace_MRI_cwd,'SP25.png'))
plt.close()

fig2_activ = plt.figure()
ax = fig2_activ.add_subplot(111, projection='3d')
ax.scatter(locations[np.abs(E_SP25) > 0.4, 0], locations[np.abs(E_SP25) > 0.4, 1], locations[np.abs(E_SP25) > 0.4, 2], alpha=1, linewidths=0, edgecolor=None, c='red', s=30)
ax.set_ylabel('Y-axis (mm)', fontsize=20, labelpad=20)
ax.set_zlabel('Z-axis (mm)', fontsize=20, labelpad=15)
ax.tick_params(axis='x', labelsize=18, pad=10, which='both', bottom=False, top=False, labelbottom=False, labeltop=False, labelright=False, labelleft=False)
ax.tick_params(axis='y', labelsize=18, pad=-2)
ax.tick_params(axis='z', labelsize=18)
plt.tight_layout()
ax.view_init(0, 180)
plt.savefig(os.path.join(HingePlace_MRI_cwd,'SP25_activ.png'))
plt.close()

fig3 = plt.figure()
ax = fig3.add_subplot(111, projection='3d')
color_map = cm.ScalarMappable(cmap='jet')
alpha_scale = E_SP50.copy()
start_transparency = 0.1
alpha_scale = (alpha_scale - np.min(alpha_scale)) / (np.max(alpha_scale) - np.min(alpha_scale)) * (1 - start_transparency) + start_transparency
rgba = color_map.to_rgba(x=E_SP50, norm=True)
ax.scatter(locations[:, 0], locations[:, 1], locations[:, 2], c=rgba)
ax.set_xlabel('X-axis (mm)', fontsize=20, labelpad=20)
ax.set_ylabel('Y-axis (mm)', fontsize=20, labelpad=4)
ax.set_zlabel('Z-axis (mm)', fontsize=20, labelpad=10)
ax.tick_params(axis='x', labelsize=18, pad=10)
ax.tick_params(axis='y', labelsize=18, pad=-2)
ax.tick_params(axis='z', labelsize=18)
cbar = plt.colorbar(mappable=color_map, ax=ax)
cbar.set_label(label=r'Vm$^{-1}$', size=20)
cbar.ax.tick_params(labelsize=18)
plt.tight_layout()
ax.view_init(40,180)
plt.savefig(os.path.join(HingePlace_MRI_cwd,'SP50.png'))
plt.close()

fig3_activ = plt.figure()
ax = fig3_activ.add_subplot(111, projection='3d')
ax.scatter(locations[np.abs(E_SP50) > 0.4, 0], locations[np.abs(E_SP50) > 0.4, 1], locations[np.abs(E_SP50) > 0.4, 2], alpha=1, linewidths=0, edgecolor=None, c='red', s=30)
ax.set_ylabel('Y-axis (mm)', fontsize=20, labelpad=20)
ax.set_zlabel('Z-axis (mm)', fontsize=20, labelpad=15)
ax.tick_params(axis='x', labelsize=18, pad=10, which='both', bottom=False, top=False, labelbottom=False, labeltop=False, labelright=False, labelleft=False)
ax.tick_params(axis='y', labelsize=18, pad=-2)
ax.tick_params(axis='z', labelsize=18)
plt.tight_layout()
ax.view_init(0, 180)
plt.savefig(os.path.join(HingePlace_MRI_cwd,'SP50_activ.png'))
plt.close()

fig4 = plt.figure()
ax = fig4.add_subplot(111, projection='3d')
color_map = cm.ScalarMappable(cmap='jet')
alpha_scale = E_SP.copy()
start_transparency = 0.1
alpha_scale = (alpha_scale - np.min(alpha_scale)) / (np.max(alpha_scale) - np.min(alpha_scale)) * (1 - start_transparency) + start_transparency
rgba = color_map.to_rgba(x=E_SP, norm=True)
ax.scatter(locations[:, 0], locations[:, 1], locations[:, 2], c=rgba)
ax.set_xlabel('X-axis (mm)', fontsize=20, labelpad=20)
ax.set_ylabel('Y-axis (mm)', fontsize=20, labelpad=4)
ax.set_zlabel('Z-axis (mm)', fontsize=20, labelpad=10)
ax.tick_params(axis='x', labelsize=18, pad=10)
ax.tick_params(axis='y', labelsize=18, pad=-2)
ax.tick_params(axis='z', labelsize=18)
cbar = plt.colorbar(mappable=color_map, ax=ax)
cbar.set_label(label=r'Vm$^{-1}$', size=20)
cbar.ax.tick_params(labelsize=18)
plt.tight_layout()
ax.view_init(40,180)
plt.savefig(os.path.join(HingePlace_MRI_cwd,'SP.png'))
plt.close()

fig4_activ = plt.figure()
ax = fig4_activ.add_subplot(111, projection='3d')
ax.scatter(locations[np.abs(E_SP) > 0.4, 0], locations[np.abs(E_SP) > 0.4, 1], locations[np.abs(E_SP) > 0.4, 2], alpha=1, linewidths=0, edgecolor=None, c='red', s=30)
ax.set_ylabel('Y-axis (mm)', fontsize=20, labelpad=20)
ax.set_zlabel('Z-axis (mm)', fontsize=20, labelpad=15)
ax.tick_params(axis='x', labelsize=18, pad=10, which='both', bottom=False, top=False, labelbottom=False, labeltop=False, labelright=False, labelleft=False)
ax.tick_params(axis='y', labelsize=18, pad=-2)
ax.tick_params(axis='z', labelsize=18)
plt.tight_layout()
ax.view_init(0, 180)
plt.savefig(os.path.join(HingePlace_MRI_cwd,'SP_activ.png'))
plt.close()

