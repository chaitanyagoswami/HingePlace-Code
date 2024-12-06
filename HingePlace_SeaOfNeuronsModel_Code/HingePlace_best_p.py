import sys
import time
import os
import numpy as np
from pulse_train import SingePulse_MonoPhasic
from nerve_and_cell_model import nerve_and_cell_model
import ray
import logging
from SphericalHeadModel import SphericalHeadModel
from CancelPoints import CancelPointsSpherical
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600

input_baseline_pttrn = int(sys.argv[1]) 

##################################################################################################################################
################################ Directory Structure & Parallelization Params #####3##############################################
os.environ['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE']='1'
ray.init(log_to_driver=False, logging_level=logging.FATAL)
#ray.init()
NUM_CORES = 20

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

## Neuron Directories
neuron_dir = os.path.join(main_dir,'HP_CorticalNeuron')
if not os.path.exists(neuron_dir):
    os.makedirs(neuron_dir)

## Results
results_dir = os.path.join(main_dir, 'Results')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
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

CREATE_NEW_MODEL = False ## Set to True to create a new model
forward_model_path = os.path.join(matrice_dir,'Patch%dmm'%(int(patch_size*10))) ## Directory Path where the model is saved or needs to be saved

## Electric Field Simulator
efield_sim = SphericalHeadModel(r_lst=r_lst,cond_vec=cond_vec,radius_vec=radius_vec,patch_size=patch_size,elec_radius=elec_radius,elec_spacing=elec_spacing,max_l=L,spacing=spacing,custom_grid=custom_grid,theta_elec=theta_elec,phi_elec=phi_elec, save_title=forward_model_path)

if CREATE_NEW_MODEL:
    efield_sim._calc_forward_model(print_elec_pattern=False, save=True, save_title=forward_model_path)
else:
    efield_sim._load_forward_model(forward_model_path) 


## Plot Patch Reference for numbering of electrodes
num_elec = efield_sim._return_num_electrodes()
PLOT_PATCH_REF = False
if PLOT_PATCH_REF:
    efield_sim.plot_elec_pttrn(J=np.arange(1,num_elec+1), x_lim=[-patch_size,patch_size], y_lim=[-patch_size,patch_size], fname=os.path.join(patch_dir,'Patch%dmm.png'%(patch_size*10)))

#################################################################################################################################
##################################### Nerve and Cell Model #######################################################################

## C-fiber params 
Vinit_nerve = -60 ##mV
temp_nerve = 37 ## celsius
dt_nerve = 0.025 ## ms
periphery_only = True
plot_nerve = False
delay_init_nerve, delay_final_nerve = 20, 2 ## ms
length_parietal = 16 ## cm
length_parietal_xlen = 20 ## cm
length_temporal_ylen = 10 ## cm
spacing = 0.5 ## cm
 
nerve_params = [Vinit_nerve, temp_nerve, dt_nerve, periphery_only, plot_nerve, delay_init_nerve, delay_final_nerve, length_parietal, length_parietal_xlen, length_temporal_ylen, spacing]

## Neuron params 
Vinit_neuron = -65 ## mV
save_state_show = False
plot_neuron = False
human_or_mice = 0 ## 0->human; 1->mice
cell_id_pyr_lst = np.array([16]) ## Different Morphology for L23 Pyr Cells
temp_neuron = 37 ## celsius
dt_neuron = 0.025 ## ms
delay_init_neuron, delay_final_neuron = 2000, 2 ## ms
num_neurons = 400
theta_max = np.pi/2-14/radius_head ## radians
if input_baseline_pttrn == 1:
    x_max = 2.5 ## cm 
    y_max = 2.5 ## cm
elif input_baseline_pttrn==2:
    option = int(sys.argv[2])
    if option == 1:
        x_max = [-5,1]
        y_max = [-2,2.5]
        cell_id_pyr_lst = np.array([6])
    else:
        x_max = [-5,1]
        y_max = [-1.5,2.5]
elif input_baseline_pttrn==3:
    focus_pt_id = int(sys.argv[2])
    if focus_pt_id == 0:
        x_max = [-2.2,3]
        y_max = 2.5
    elif focus_pt_id == 1:
        x_max = [-1.5,4.5]     
        y_max = 2.5 
    elif focus_pt_id == 2:
        x_max = [-4.5,3.8]     
        y_max = [-3.5,5.5] 
    elif focus_pt_id == 3:
        x_max = [-3,1.3]     
        y_max = [-1.3,1.8] 
    elif focus_pt_id == 4:
        x_max = [-3,4.3]     
        y_max = [-4,2.5] 

SEED = 1234
neuron_params = [Vinit_neuron, save_state_show, plot_neuron, human_or_mice, cell_id_pyr_lst, temp_neuron, dt_neuron, delay_init_neuron, delay_final_neuron, num_neurons, theta_max, x_max, y_max]
model =  nerve_and_cell_model(neuron_params=neuron_params, nerve_params=nerve_params, num_cores_nerve=NUM_CORES, num_cores_neuron=NUM_CORES, overall_radius=radius_head, depth_nerve=depth_nerve, depth_neuron=depth_neuron, SEED=SEED)

PLOT_NEURON_ORIENT = False
if PLOT_NEURON_ORIENT:
    if input_baseline_pttrn == 2 and option == 2:
        model.plot_neuron_orient(fname=os.path.join(neuron_dir,"DiffNeuronOrientation_cellId%d"%cell_id_pyr_lst[0]), show=True)
    else:
        model.plot_neuron_orient(fname=os.path.join(neuron_dir,"NeuronOrientation_cellId%d"%cell_id_pyr_lst[0]), show=True)
##################################################################################################################################
########################################### Waveform Model #######################################################################
pulse_train = SingePulse_MonoPhasic()
amp = -1 ## mA
delay = 1 ## ms
total_time = 5 ##ms
sampling_rate = 1e6

##################################################################################################################################
############################################ Baseline Sim ########################################################################

if input_baseline_pttrn == 1:
    baseline_dir = os.path.join(results_dir,os.path.join('BaselineHD-TDCS','Bestp'))
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)
elif input_baseline_pttrn == 2:
    if option == 1: ## Use L23 Pyr cell
        baseline_dir = os.path.join(results_dir,os.path.join('Baseline2ElecL23Pyr','Bestp'))
        if not os.path.exists(baseline_dir):
            os.makedirs(baseline_dir)
    else:
        baseline_dir = os.path.join(results_dir,os.path.join('Baseline2Elec','Bestp'))
        if not os.path.exists(baseline_dir):
            os.makedirs(baseline_dir)
elif input_baseline_pttrn == 3:
    if focus_pt_id == 0:
        baseline_dir = os.path.join(results_dir,os.path.join('BaselineOptDir', 'Bestp'))
    else:
        baseline_dir = os.path.join(results_dir,os.path.join('BaselineOptDir%d'%int(focus_pt_id),'Bestp'))
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)
   
## Baseline Electrode Pattern for HD-TDCS and 2Elec
J_baseline = np.zeros(num_elec)

if input_baseline_pttrn == 1:
    ## HD-TDCS
    J_baseline[0] = -1
    J_baseline[[10,16,7,13]] = 0.25
elif input_baseline_pttrn == 2:
    ## 2-Elec
    J_baseline[13] = -1
    J_baseline[7] = 1
    
## Plot Baseline Pattern
PLOT_BASELINE_PTTRN = False
if PLOT_BASELINE_PTTRN:
    if input_baseline_pttrn == 1:
        efield_sim.plot_elec_pttrn(J=J_baseline, x_lim=[-patch_size,patch_size], y_lim=[-patch_size,patch_size], fname=os.path.join(baseline_dir,'BaselineHD-TDCS.png'))
    elif input_baseline_pttrn == 2:
        efield_sim.plot_elec_pttrn(J=J_baseline, x_lim=[-patch_size,patch_size], y_lim=[-patch_size,patch_size], fname=os.path.join(baseline_dir,'Baseline2Elec.png'))

if input_baseline_pttrn == 1 or input_baseline_pttrn == 2:
    pos_norm, pos_X, pos_Y, pos_Z = efield_sim._get_max_locations(r=radius_head-depth_neuron, J=J_baseline)
    curr_norm_baseline, curr_X_baseline, curr_Y_baseline, curr_Z_baseline, xy_grid =  efield_sim.calc_all_density(r=radius_head-depth_neuron, J=J_baseline)
    pos_norm[2]=np.pi
    pos_norm_cart = efield_sim._sph_to_cart(pos_norm.reshape(1,3))
    idx_norm = np.argmin(np.sum(np.square(xy_grid - pos_norm_cart[:,:2]), axis=1))
    curr_X_value, curr_Y_value, curr_Z_value= curr_X_baseline[idx_norm], curr_Y_baseline[idx_norm], curr_Z_baseline[idx_norm]
    curr_norm_value = curr_norm_baseline[idx_norm]
    
    print("The value of electric field norm at maximum norm pos: %.2f mA/cm^2"%(np.sqrt(curr_X_value**2+curr_Y_value**2+curr_Z_value**2)))
    print("The value of electric field X at maximum norm pos: %.2f mA/cm^2"%(curr_X_value))
    print("The value of electric field Y at maximum norm pos: %.2f mA/cm^2"%(curr_Y_value))
    print("The value of electric field Z at maximum norm pos: %.2f mA/cm^2"%(curr_Z_value))
    
    print("The location of maximum electric field norm: %.2f, %.2f, %.2f"%(pos_norm[0], pos_norm[1], pos_norm[2]))
    print("The location of maximum electric field X: %.2f, %.2f, %.2f"%(pos_X[0], pos_X[1], pos_X[2]))
    print("The location of maximum electric field Y: %.2f, %.2f, %.2f"%(pos_Y[0], pos_Y[1], pos_Y[2]))
    print("The location of maximum electric field Z: %.2f, %.2f, %.2f"%(pos_Z[0], pos_Z[1], pos_Z[2]))

    efield_sim._setJ(J_baseline)

    PLOT_BASELINE_CURR = False
    if PLOT_BASELINE_CURR:
        if input_baseline_pttrn == 1:
            efield_sim.plot_all_curr_density(r=radius_head-depth_neuron, J=J_baseline, fname=os.path.join(baseline_dir,'BaselineHD-TDCS_curr_density'))
        elif input_baseline_pttrn == 2:
            efield_sim.plot_all_curr_density(r=radius_head-depth_neuron, J=J_baseline, fname=os.path.join(baseline_dir,'Baseline2Elec_curr_density'))
    
## Cancel Points
offset = 0.1
thickness = 7
spacing = 0.1
cancel_points = CancelPointsSpherical(spacing=spacing) 
if input_baseline_pttrn == 1:
    focus_pts =np.array([radius_head-depth_neuron, np.pi/2.0, 0]).reshape(1,3) 
elif input_baseline_pttrn == 2:
    focus_pts = pos_norm.reshape(1,3)
elif input_baseline_pttrn == 3:
    focus_pts_lst = [np.array([radius_head-depth_neuron, np.pi/2.0, 0]).reshape(1,3)] 
    focus_pts_lst = focus_pts_lst+[np.array([radius_head-depth_neuron, np.pi/2.0-2/(radius_head-depth_neuron), 0]).reshape(1,3)]
    focus_pts_lst = focus_pts_lst+[np.array([radius_head-depth_neuron, np.pi/2.0-1/(radius_head-depth_neuron), np.pi/2]).reshape(1,3)]
    focus_pts_lst = focus_pts_lst+[np.array([radius_head-depth_neuron, np.pi/2.0-2/(radius_head-depth_neuron), np.pi]).reshape(1,3)]
    focus_pts_lst = focus_pts_lst+[np.array([radius_head-depth_neuron, np.pi/2.0-1/(radius_head-depth_neuron), 3*np.pi/2]).reshape(1,3)]
    focus_pts = focus_pts_lst[focus_pt_id]

cancel_pts = cancel_points.uniform_ring_sample(focus_pts.flatten(), offset, thickness)
Af_v, Ac_v, Af_z, Ac_z, Af_x, Ac_x, Af_y, Ac_y = efield_sim._get_Af_and_Ac(focus_pts, cancel_pts)

## HD-TDCS
if input_baseline_pttrn == 1:
    direction = np.array([0,0,1])
    Itot_mul = [2,4,6]
    Jdes = curr_Z_value*220 ##mA/cm^2
    Isafety_lst = [150,166,183,200,217,234,250] ##mA
elif input_baseline_pttrn == 2:
    direction = np.array([curr_X_value,curr_Y_value,curr_Z_value])
    direction = direction/np.sqrt(np.sum(direction**2))
    Itot_mul = [2,4,6]
    Jdes = curr_norm_value*200 ##mA/cm^2
    Isafety_lst = [130,144,158,172,186,200,214]
elif input_baseline_pttrn == 3:
    direction = np.load(os.path.join(neuron_dir,"best_unit_vec.npy"))
    direction = direction/np.sqrt(np.sum(direction**2))
    Itot_mul = [2,4,6]
    Jdes = np.load(os.path.join(neuron_dir,"activ_thresh.npy")) ##mA/cm^2
    Jdes = np.min(Jdes)*brain_cond*0.1*1.6 ##mA/cm^2
    if focus_pt_id == 0:
        Isafety_lst = [120,130,140,150,160,170]
        Itot_mul = [2,4,6]
    elif focus_pt_id==1:
        Isafety_lst = [130,140,150,160,170]
        Itot_mul = [2,6]
    elif focus_pt_id==2:
        Isafety_lst = [170,180,190,200,220]
        Itot_mul = [2,6]
    elif focus_pt_id==3:
        Isafety_lst = [130,140,150,160,170]
        Itot_mul = [2,6]
    elif focus_pt_id==4:
        Isafety_lst = [170,180,190,200,220]
        Itot_mul = [2,6]



## Pulse Parameters
pw = 0.2 ## ms
amp_array, time_array = pulse_train.amp_train(amp=amp, delay=delay, total_time=total_time, pw=pw, sampling_rate=sampling_rate)
amp_array_lst = [amp_array]+[None]*7
scale_lst = [1] +[None]*7

## Figuring out J-tol
p=np.arange(10)+1

num_points = 20 
np.random.seed(SEED)
rand_tol = np.random.uniform(low=0.1, high=0.7, size=(num_points,3))
best_spikes, best_tol = [], [] 
LOAD_FLAG = False
verbose = False
for m in range(len(p)):
    print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("Solving for %d-th p: %.2f"%(m+1, p[m]))
    print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    tuning_dir = os.path.join(baseline_dir,"p=%dx10e-02"%(int(p[m]*100)))
    if not os.path.exists(tuning_dir):
        os.makedirs(tuning_dir)
    Jtol_HP_spikes_lst = []
    for i in range(rand_tol.shape[0]):
        start_time = time.time()
        print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print("Iteration %d"%(i+1))
        print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        
        if input_baseline_pttrn == 1:     
            Isafety = 200 ## Tunde for the middle injected current
            Itotal = 2*Isafety*4 ## Allow for HingePlace to design effective patterns
        elif input_baseline_pttrn == 2:
            Isafety = 172 ## Tunde for the middle injected current
            Itotal = 2*Isafety*4 ## Allow for HingePlace to design effective patterns
        elif input_baseline_pttrn == 3:
            if focus_pt_id == 0:
                Isafety = 130
                Itotal = 2*Isafety*2
            elif focus_pt_id == 1:
                Isafety = 150
                Itotal = 2*Isafety*4
            elif focus_pt_id == 2:
                Isafety  = 190
                Itotal = 2*Isafety*4
            elif focus_pt_id == 3:
                Isafety  = 130
                Itotal = 2*Isafety*2
            elif focus_pt_id == 4:
                Isafety  = 200
                Itotal = 2*Isafety*4
        if not LOAD_FLAG:
            Jtol = [rand_tol[i,0]*Jdes,rand_tol[i,1]*Jdes,rand_tol[i,2]*Jdes]   
            J_HP = efield_sim.HingePlace_3d(direction=direction, Jdes=Jdes, Isafety=Isafety, Jtol=Jtol, Itotal=Itotal, Af=[Af_x, Af_y, Af_z], Ac=[Ac_x, Ac_y, Ac_z], p=p[m], verbose=verbose)
            print("Time Taken: %.2f s"%(time.time()-start_time))
            efield_sim._setJ(J_HP)
            J_lst = [J_HP]+[np.zeros(num_elec)]*7
            efield_sim._setJ_lst(J_lst)
            elec_field_lst = [efield_sim]+[None]*7
            spikes_nerve_HP, _ = model.stimulate(time_array=time_array, amp_array_lst=amp_array_lst, scale_lst=scale_lst, sampling_rate=sampling_rate, elec_field=efield_sim, scalp_flag=False)
            fname = os.path.join(tuning_dir,'HP_tuning_cortex_activation%d.png'%i)
            model.plot_cortex_activation(savepath=fname, show=False)
            np.save(os.path.join(tuning_dir, "SpikesHP_Jtol%d.npy"%i), spikes_nerve_HP)
            np.save(os.path.join(tuning_dir, "Jtol%d.npy"%i), rand_tol[i])
        else:
            filename = os.path.join(tuning_dir, "SpikesHP_Jtol%d.npy"%i)
            if os.path.exists(filename):
                spikes_nerve_HP = np.load(filename)
            else:
                spikes_nerve_HP = np.ones(num_neurons)
        Jtol_HP_spikes_lst.append(spikes_nerve_HP) 
        print("Time Taken: %.2f s"%(time.time()-start_time))
    Jtol_HP_spikes_lst = np.array(Jtol_HP_spikes_lst)
    best_idx = np.argmin(np.sum(Jtol_HP_spikes_lst, axis=1))
    best_spikes.append(Jtol_HP_spikes_lst[best_idx])
    best_tol.append(rand_tol[best_idx])
    np.save(os.path.join(tuning_dir, "JtolSpikes_lst.npy"), Jtol_HP_spikes_lst)
    np.save(os.path.join(tuning_dir, "Jtol_lst.npy"), rand_tol) 
    xlabels = np.arange(Jtol_HP_spikes_lst.shape[0])+1
    plt.barh(xlabels, list(np.sum(Jtol_HP_spikes_lst, axis=1)))
    plt.xlabel("Diff. Jtol configurations", fontsize=19)
    plt.ylabel("Total Spikes", fontsize=19)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(tuning_dir,"TuningBarPlot.png"))
    plt.close()
    np.save(os.path.join(baseline_dir,'best_spikes.npy'), np.array(best_spikes))
    np.save(os.path.join(baseline_dir,'best_tol.npy'), np.array(best_tol))
best_spikes = np.array(best_spikes)
plt.plot(p,np.sum(best_spikes,axis=1),marker='x')
plt.xlabel(r'$p$', fontsize=24)
plt.ylabel('Total Spikes', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(baseline_dir,'Best-p.png'))
plt.show()

