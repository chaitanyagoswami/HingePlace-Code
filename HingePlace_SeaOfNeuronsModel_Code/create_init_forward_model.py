import os
import numpy as np
from SphericalHeadModel import SphericalHeadModel

##################################################################################################################################
################################ Directory Structure & Parallelization Params #####3##############################################

main_dir = os.path.join(os.getcwd(), 'HingePlace')
if not os.path.exists(main_dir):
    os.makedirs(main_dir)

## E-field Directories
efield_dir = os.path.join(main_dir,'HP_EfieldSim')
if not os.path.exists(efield_dir):
    os.makedirs(efield_dir)

patch_dir = os.path.join(efield_dir,'HP_Patch')
if not os.path.exists(patch_dir):
    os.makedirs(patch_dir)

matrice_dir = os.path.join(efield_dir,'HP_Field_Matrices')
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
r_lst = np.linspace(7.7, 8, 40)
forward_model_path = os.path.join(matrice_dir,'Patch%dmm'%(int(patch_size*10))) ## Directory Path where the model is saved or needs to be saved

## Create Forward Model
efield_sim = SphericalHeadModel(r_lst=r_lst,cond_vec=cond_vec,radius_vec=radius_vec,patch_size=patch_size,elec_radius=elec_radius,elec_spacing=elec_spacing,max_l=L,spacing=spacing,custom_grid=custom_grid,theta_elec=theta_elec,phi_elec=phi_elec, save_title=forward_model_path)
efield_sim._calc_forward_model(print_elec_pattern=False, save=True, save_title=forward_model_path)
