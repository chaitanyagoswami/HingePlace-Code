import sys
import time
import os
import numpy as np
from elec_field import ICMS 
from matplotlib import colormaps
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib import cm
import ray
import logging
from SphericalHeadModel import *
from nerve_and_cell_model_helper import cart_to_sph, sph_to_cart
from CancelPoints import CancelPointsSpherical
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 600
np.set_printoptions(suppress=True)
##################################################################################################################################
################################ Directory Structure & Parallelization Params #####3##############################################
os.environ['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE']='1'
ray.init(log_to_driver=False, logging_level=logging.FATAL)
#ray.init()
NUM_CORES = 25

main_dir = os.path.join(os.getcwd(), 'HingePlace/HP-L1L1-equivalence_ver2')
if not os.path.exists(main_dir):
    os.makedirs(main_dir)

matrice_dir = os.path.join('HingePlace/HP_EfieldSim/HP_Field_Matrices')
if not os.path.exists(matrice_dir):
    os.makedirs(matrice_dir)

##################################################################################################################################    

radius_head = 9.2 # cm
depth_nerve = 0.1 ## cm
depth_neuron = 1.3 ## cm

##################################################################################################################################
############################################ E-field Model #######################################################################
## Electric Field Simulator
thickness_skull = 0.5 #cm
thickness_scalp = 0.6 #cm
thickness_CSF = 0.09 #cm
no_of_layers = 4
skull_cond = 0.006 #S/m
brain_cond = 0.33 #S/m
CSF_cond = 1.79 #S/m
scalp_cond = 0.3 #S/m
radius_vec = np.array([radius_head-thickness_skull-thickness_CSF-thickness_scalp, radius_head-thickness_skull-thickness_scalp, radius_head-thickness_scalp, radius_head])
cond_vec = np.array([brain_cond, CSF_cond, skull_cond, scalp_cond])
L = 400

## Patch Specifications
patch_size = 7 #cm
elec_radius = 0.6 #cm
elec_spacing = 2.0 #cm
spacing = 0.05
custom_grid = False
theta_elec, phi_elec = None, None

## Depths for which E-field is calculated 
r_neuron = np.linspace(7.7, 8, 40)
r_lst=np.concatenate([r_neuron,np.array([radius_head-depth_nerve])])

CREATE_NEW_MODEL = False ## Set to True to create a new model
forward_model_path = os.path.join(matrice_dir,'Patch%dmm'%(int(patch_size*10))) ## Directory Path where the model is saved or needs to be saved

## Electric Field Simulator
efield_sim = SphericalHeadModel(r_lst=r_lst,cond_vec=cond_vec,radius_vec=radius_vec,patch_size=patch_size,elec_radius=elec_radius,elec_spacing=elec_spacing,max_l=L,spacing=spacing,custom_grid=custom_grid,theta_elec=theta_elec,phi_elec=phi_elec, save_title=forward_model_path) 
efield_sim._load_forward_model(forward_model_path) 
## Plot Patch Reference for numbering of electrodes
num_elec = efield_sim._return_num_electrodes()
PLOT_PATCH_REF = False
if PLOT_PATCH_REF:
    efield_sim.plot_elec_pttrn(J=np.arange(1,num_elec+1), x_lim=[-patch_size,patch_size], y_lim=[-patch_size,patch_size], fname=os.path.join(patch_dir,'Patch%dmm.png'%(patch_size*10)))

###############################################################################################################################################
############################################ Different HyperParam Cond ########################################################################
epsilon = 10**(-1*np.linspace(0,2,5))
alpha = 10**(-1*np.linspace(1,5,5))
hyperParams_l1l1 = np.meshgrid(epsilon,alpha)
hyperParams_l1l1 = np.concatenate([hyperParams_l1l1[0].reshape(-1,1),hyperParams_l1l1[1].reshape(-1,1)], axis=1)## 0 column-> epsilon, 1 column->alpha
## Cancel Points
offset = 0.5
thickness = 7
spacing = 0.1
cancel_points = CancelPointsSpherical(spacing=spacing) 
focus_pts_lst = []
focus_pts_lst = focus_pts_lst+[np.array([radius_head-depth_neuron, np.pi/2.0, 0]).reshape(1,3)] 
focus_pts_lst = focus_pts_lst+[np.array([radius_head-depth_neuron, np.pi/2.0-2/(radius_head-depth_neuron), 0]).reshape(1,3)]
focus_pts_lst = focus_pts_lst+[np.array([radius_head-depth_neuron, np.pi/2.0-2/(radius_head-depth_neuron), np.pi/2]).reshape(1,3)]
focus_pts_lst = focus_pts_lst+[np.array([radius_head-depth_neuron, np.pi/2.0-2/(radius_head-depth_neuron), np.pi]).reshape(1,3)]
focus_pts_lst = focus_pts_lst+[np.array([radius_head-depth_neuron, np.pi/2.0-2/(radius_head-depth_neuron), 3*np.pi/2]).reshape(1,3)]

J_baseline = np.zeros(num_elec)

## C3-C4
J_baseline[13] = -1
J_baseline[7] = 1
 
pos_norm, pos_X, pos_Y, pos_Z = efield_sim._get_max_locations(r=radius_head-depth_neuron, J=J_baseline)
curr_norm_baseline, curr_X_baseline, curr_Y_baseline, curr_Z_baseline, xy_grid =  efield_sim.calc_all_density(r=radius_head-depth_neuron, J=J_baseline)
pos_norm[2]=np.pi
pos_norm_cart = efield_sim._sph_to_cart(pos_norm.reshape(1,3))
idx_norm = np.argmin(np.sum(np.square(xy_grid - pos_norm_cart[:,:2]), axis=1))
curr_X_value, curr_Y_value, curr_Z_value= curr_X_baseline[idx_norm], curr_Y_baseline[idx_norm], curr_Z_baseline[idx_norm]
curr_scale = 220
Jdes = np.array([curr_scale*curr_X_value, curr_scale*curr_Y_value, curr_scale*curr_Z_value])

savedir_name = ['Center', 'Right', 'Up', 'Left', 'Down']
LOAD_FLAG = False

for focus_pts, baseline_dir in zip(focus_pts_lst, savedir_name):
    cancel_pts = cancel_points.uniform_ring_sample(focus_pts.flatten(), offset, thickness)
    focus_pts = cancel_points.uniform_ring_sample(focus_pts.flatten(), 0, 0.15)
    Jdes = np.concatenate([np.tile(Jdes[0],[focus_pts.shape[0]]),np.tile(Jdes[1],[focus_pts.shape[0]]),np.tile(Jdes[2],[focus_pts.shape[0]])]) ## Check the tiling option
    Af_v, Ac_v, Af_z, Ac_z, Af_x, Ac_x, Af_y, Ac_y = efield_sim._get_Af_and_Ac(focus_pts, cancel_pts)
    hp_pttrn_dir = os.path.join(os.path.join(main_dir,baseline_dir),'HP')
    l1l1_pttrn_dir = os.path.join(os.path.join(main_dir,baseline_dir),'L1L1')
    if not os.path.exists(hp_pttrn_dir):
        os.makedirs(hp_pttrn_dir)
    if not os.path.exists(l1l1_pttrn_dir):
        os.makedirs(l1l1_pttrn_dir)
    Isafety_lst = [200,220,240,280,300]
    Itot_mul = [2,4,6]
    Af_obj = ray.put([Af_x, Af_y, Af_z])
    Ac_obj = ray.put([Ac_x, Ac_y, Ac_z])
    diff_I = []
    for i in range(len(Itot_mul)):
        diff_I_elem = []
        for ii in range(len(Isafety_lst)):
            start_time = time.time()
            Isafety = Isafety_lst[ii]
            Itotal = 2*Itot_mul[i]*Isafety
            J_l1l1, J_HP = [],[]
            if not LOAD_FLAG:
                for iii in range(len(hyperParams_l1l1)):
                    epsilon = hyperParams_l1l1[iii,0]
                    alpha = hyperParams_l1l1[iii,1]
                    J_l1l1 = J_l1l1+[L1L1norm_3d_uppr_bound.remote(Jdes=Jdes, Isafety=Isafety, epsilon=epsilon, alpha=alpha, Itotal=Itotal, Af=Af_obj, Ac=Ac_obj)] 
                J_l1l1 = ray.get(J_l1l1)
                prob_status_l1l1 = [J_l1l1[iii][1] for iii in range(len(hyperParams_l1l1))]
                J_l1l1 = np.array([J_l1l1[iii][0] for iii in range(len(hyperParams_l1l1))])
            
                print("L1L1 Done")
                for iii in range(len(hyperParams_l1l1)):
                    epsilon = hyperParams_l1l1[iii,0]
                    Jdes_prime = np.concatenate([Af_x,Af_y,Af_z], axis=0)@J_l1l1[iii]
                    J_HP = J_HP+[HingePlace_3d_uppr_bound.remote(Jdes=Jdes_prime.flatten(), Isafety=Isafety, Jtol=epsilon*np.max(np.abs(Jdes)), Itotal=np.sum(np.abs(J_l1l1[iii])), Af=Af_obj, Ac=Ac_obj, nu=np.max(np.abs(Jdes)))] 
                J_HP = ray.get(J_HP)
                prob_status_HP = [J_HP[iii][1] for iii in range(len(hyperParams_l1l1))]            
                J_HP = np.array([J_HP[iii][0] for iii in range(len(hyperParams_l1l1))])
                np.save(os.path.join(hp_pttrn_dir,'JHP_Isafety%dmA_Itotal%d.npy'%(Isafety_lst[ii],Itot_mul[i])),J_HP)
                np.save(os.path.join(l1l1_pttrn_dir,'Jl1l1_Isafety%dmA_Itotal%d.npy'%(Isafety_lst[ii],Itot_mul[i])), J_l1l1)
            else:
                J_HP = np.load(os.path.join(hp_pttrn_dir,'JHP_Isafety%dmA_Itotal%d.npy'%(Isafety_lst[ii],Itot_mul[i])))
                J_l1l1 = np.load(os.path.join(l1l1_pttrn_dir,'Jl1l1_Isafety%dmA_Itotal%d.npy'%(Isafety_lst[ii],Itot_mul[i])))
            if not LOAD_FLAG:
                print(np.sum(np.abs(J_l1l1-J_HP), axis=1),np.sum(np.abs(J_l1l1), axis=1), prob_status_HP, prob_status_l1l1)
            else:
                print(np.sum(np.abs(J_l1l1-J_HP), axis=1),np.sum(np.abs(J_l1l1), axis=1))
            diff_I_elem.append(np.sum(np.abs(J_l1l1-J_HP), axis=1)/np.sum(np.abs(J_l1l1), axis=1)*100)
            print("Median Error %.2f+-%.2f"%(np.median(diff_I_elem[-1]),np.sqrt(np.var(diff_I_elem[-1]))))
            print("Time Taken %.2f min"%((time.time()-start_time)/60))
        diff_I.append(np.array(diff_I_elem).copy())

    Isafety_name = [str(Isafety_lst[q]) for q in range(len(Isafety_lst))]
    error_lst = {}
    for i in range(len(Itot_mul)):
        error_lst[str(Itot_mul[i])] = np.median(diff_I[i], axis=1)
    x = np.arange(len(Isafety_name))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots()

    for attribute, measurement in error_lst.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=r"$I_{tot}^{mul}$-"+attribute)
        multiplier += 1
    plt.xlabel(r'$I_{safe}$ (mA)', fontsize=24)
    plt.xticks(x + width, Isafety_name, fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("% Difference", fontsize=24)
    plt.legend(fontsize=20)
    plt.ylim(ymax=0.016)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.join(main_dir,baseline_dir), "Error_Summary"+baseline_dir+".png"))
    plt.show()
