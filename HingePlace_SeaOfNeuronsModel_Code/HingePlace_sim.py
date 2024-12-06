import sys
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
#ray.init() ### Uncomment to see all the messages from the back-end
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

CREATE_NEW_MODEL = False## Set to True to create a new model
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

## C-fiber params -- Parameters for the scalp c-fiber nerves--not used in this simulation study -- IGNORE!!!
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
num_neurons = 5
theta_max = np.pi/2-14/radius_head ## radians

if input_baseline_pttrn == 1:
    x_max =[-4,4] #[-2.5,2.5] ## cm 
    y_max =[-4,4] #[-2.5,2.5] ## cm
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

PLOT_SEA_OF_NEURON_MODEL = False
if PLOT_SEA_OF_NEURON_MODEL:
    elec_lst = efield_sim.plot_elec_pttrn(J=np.arange(1,num_elec+1), x_lim=[-patch_size,patch_size], y_lim=[-patch_size,patch_size], show=False,)
    coord_elec = efield_sim._sph_to_cart(np.concatenate([np.ones([len(elec_lst),1])*radius_head,elec_lst], axis=1)) 
    model.plot_points_to_sample(coord_elec=coord_elec, J=None, savepath=os.path.join(main_dir,"SysModel"), depth_skull=thickness_scalp, depth_CSF=thickness_scalp+thickness_skull, depth=thickness_scalp+thickness_skull+thickness_CSF, xlim=np.array([-7,7]), ylim=np.array([-7,7]))

##################################################################################################################################
########################################### Waveform Model #######################################################################
pulse_train = SingePulse_MonoPhasic()
amp = -1 ## mA
delay = 1 ## ms
total_time = 5 ##ms
sampling_rate = 1e6

##################################################################################################################################
############################################ Baseline Sim ########################################################################
#### input-baseline-pattern chooses the baseline pattern being used; 1->HD-tDCS, 2->C3 C4, 3->Opt dir at north pole
if input_baseline_pttrn == 1:
    baseline_dir = os.path.join(results_dir,'Sec-5-2-2')
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)
elif input_baseline_pttrn == 2:
    if option == 1: ## Use L23 Pyr cell
        baseline_dir = os.path.join(results_dir,'Sec-5-2-4')
        if not os.path.exists(baseline_dir):
            os.makedirs(baseline_dir)
    else:
        baseline_dir = os.path.join(results_dir,'Sec-5-2-1')
        if not os.path.exists(baseline_dir):
            os.makedirs(baseline_dir)
elif input_baseline_pttrn == 3:
    if focus_pt_id == 0:
        baseline_dir = os.path.join(results_dir,'Sec-5-2-3')
    else:
        baseline_dir = os.path.join(results_dir,'Sec-5-2-5/BaselineOptDir%d'%int(focus_pt_id))
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

## Determing the required input current, direction, and the target of stimulation using the HD-tDCS and C3-C4 baseline patterns
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
    
    ## IP-OP Curve
    CALC_IP_OP_CURVE = False
    if CALC_IP_OP_CURVE:
        J_lst = [J_baseline]+[np.zeros(num_elec)]*7
        efield_sim._setJ_lst(J_lst)
        elec_field_lst = [efield_sim]+[None]*7
        
        pw = 0.2 ## ms
        amp_array_baseline, time_array_baseline = pulse_train.amp_train(amp=amp, delay=delay, total_time=total_time, pw=pw, sampling_rate=sampling_rate)
        amp_array_lst = [amp_array_baseline]+[None]*7
    
        scale = np.linspace(100,300,11)
        baseline_spikes = []
        for i in range(len(scale)):
            scale_lst = [scale[i]]+[None]*7
            spikes_neuron, locations = model.stimulate(time_array=time_array_baseline, amp_array_lst=amp_array_lst, scale_lst=scale_lst, sampling_rate=sampling_rate, elec_field=efield_sim, scalp_flag=False)
            if input_baseline_pttrn == 1:
                np.save(os.path.join(baseline_dir,'BaselineHD-TDCS_cortex_spikes_amp%dmA.npy'%scale[i]), spikes_neuron)
            elif input_baseline_pttrn == 2:
                np.save(os.path.join(baseline_dir,'Baseline2Elec_cortex_spikes_amp%dmA.npy'%scale[i]), spikes_neuron)
            
            if i==0:
                if input_baseline_pttrn == 1:
                    np.save(os.path.join(baseline_dir,'BaselineHD-TDCS_cortex_locations.npy'), locations)
                elif input_baseline_pttrn == 2:
                    np.save(os.path.join(baseline_dir,'Baseline2Elec_cortex_locations.npy'), locations)
    
            baseline_spikes.append(np.sum(spikes_neuron))
            if input_baseline_pttrn == 1:
                fname = os.path.join(baseline_dir,'BaselineHD-TDCS_cortex_activation_amp%dmA.png'%scale[i])
            elif input_baseline_pttrn == 2:
                fname = os.path.join(baseline_dir,'Baseline2Elec_cortex_activation_amp%dmA.png'%scale[i])
            model.plot_cortex_activation(savepath=fname, show=True)
        
        baseline_spikes = np.array(baseline_spikes)
        if input_baseline_pttrn == 1:
            np.save(os.path.join(baseline_dir,'BaselineHD-TDCS_cortex_spikes.npy'), baseline_spikes)
            np.save(os.path.join(baseline_dir,'BaselineHD-TDCS_amp_scale.npy'), scale)
        elif input_baseline_pttrn == 2:
            np.save(os.path.join(baseline_dir,'Baseline2Elec_cortex_spikes.npy'), baseline_spikes)
            np.save(os.path.join(baseline_dir,'Baseline2Elec_amp_scale.npy'), scale)
        plt.plot(scale, baseline_spikes, marker='x', color='tab:blue')
        plt.xlabel('Injected Current (mA)', fontsize=21)
        plt.ylabel('Total Spikes', fontsize=21)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        if input_baseline_pttrn == 1:
            plt.savefig(os.path.join(baseline_dir,'BaselineHD-TDCS_ip-op_curve.png'))
        elif input_baseline_pttrn == 2:
            plt.savefig(os.path.join(baseline_dir,'Baseline2Elec_ip-op_curve.png'))
        plt.show()

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

hp_pttrn_dir = os.path.join(baseline_dir,'HP')
sp_pttrn_dir = os.path.join(baseline_dir,'SP')
if not os.path.exists(hp_pttrn_dir):
    os.makedirs(hp_pttrn_dir)
if not os.path.exists(sp_pttrn_dir):
    os.makedirs(sp_pttrn_dir)

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
    #### Need to run the find_opt_dir_hp before using this option
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
#########
## Figuring out J-tol (E_tol in the manuscriot). All the tuning is done for a specific value of Isafety and Itotal. This value is used for the whole experiment for that technique. For example, for Sec: 5-1-1, the initial Jtol number is found at
## Isafety=150mA, and Itotal=4, but is used for all values of Isafety and Itot. If the HingePlace performs worse than LCMV-E at certain values of Isafety and Itot, then we re-tune it to find a new value of Jtol to be used for those values of Isafety
## and Itot. To see the full potential of HingePlace, we recommend tuning Jtol for each Isafety and Itot but since this can be computationally expensive. We use this ad-hoc way of finding Jtol hyperparameters. Please ensure that you use a different
## filename while re-running the tuning code as mentioned above.
#########
Jtol_tuning_once = True
USE_TUNING_CODE =True
if Jtol_tuning_once and USE_TUNING_CODE:
    tuning_dir = os.path.join(hp_pttrn_dir,'J-tol_Tuning')
    if not os.path.exists(tuning_dir):
        os.makedirs(tuning_dir)

    LOAD = False
    num_points = 30
    np.random.seed(SEED)
    rand_tol = np.random.uniform(low=0.1, high=0.7, size=(num_points,3))
    Jtol_HP_spikes_lst = []
    Jtol_HP2_spikes_lst = []
    Jtol_HP3_spikes_lst = []
    if not LOAD:
        for i in range(rand_tol.shape[0]):
            print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            print("Iteration %d"%(i+1))
            print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

            if input_baseline_pttrn == 1:
                Isafety = 150 ## Tunded for the middle injected current
                Itotal = 2*Isafety*4 ## Allow for HingePlace to design effective patterns
            elif input_baseline_pttrn == 2:
                Isafety = 144 ## Tuned for the middle injected current
                Itotal = 2*Isafety*4 ## Allow for HingePlace to design effective patterns
            elif input_baseline_pttrn == 3:
                if focus_pt_id == 0:
                    Isafety = 150
                    Itotal = 2*Isafety*6
                elif focus_pt_id == 1:
                    Isafety = 140
                    Itotal = 2*Isafety*2
                elif focus_pt_id == 2:
                    Isafety  = 190
                    Itotal = 2*Isafety*6
                elif focus_pt_id == 3:
                    Isafety  = 170
                    Itotal = 2*Isafety*6
                elif focus_pt_id == 4:
                    Isafety  = 200
                    Itotal = 2*Isafety*6

            Jtol = [rand_tol[i,0]*Jdes,rand_tol[i,1]*Jdes,rand_tol[i,2]*Jdes]
            J_HP = efield_sim.HingePlace_3d(direction=direction, Jdes=Jdes, Isafety=Isafety, Jtol=Jtol, Itotal=Itotal, Af=[Af_x, Af_y, Af_z], Ac=[Ac_x, Ac_y, Ac_z], p=1)
            J_HP2 = efield_sim.HingePlace_3d(direction=direction, Jdes=Jdes, Isafety=Isafety, Jtol=Jtol, Itotal=Itotal, Af=[Af_x, Af_y, Af_z], Ac=[Ac_x, Ac_y, Ac_z], p=2)
            J_HP3 = efield_sim.HingePlace_3d(direction=direction, Jdes=Jdes, Isafety=Isafety, Jtol=Jtol, Itotal=Itotal, Af=[Af_x, Af_y, Af_z], Ac=[Ac_x, Ac_y, Ac_z], p=3)

            efield_sim._setJ(J_HP)
            J_lst = [J_HP]+[np.zeros(num_elec)]*7
            efield_sim._setJ_lst(J_lst)
            spikes_nerve_HP, _ = model.stimulate(time_array=time_array, amp_array_lst=amp_array_lst, scale_lst=scale_lst, sampling_rate=sampling_rate, elec_field=efield_sim, scalp_flag=False)
            fname = os.path.join(tuning_dir,'HP_tuning_cortex_activation%d.png'%i)
            model.plot_cortex_activation(savepath=fname, show=False)
            np.save(os.path.join(tuning_dir, "SpikesHP_Jtol%d.npy"%i), spikes_nerve_HP)
            np.save(os.path.join(tuning_dir, "Jtol%d_HP.npy"%i), rand_tol[i])

            efield_sim._setJ(J_HP2)
            J_lst = [J_HP2]+[np.zeros(num_elec)]*7
            efield_sim._setJ_lst(J_lst)
            spikes_nerve_HP, _ = model.stimulate(time_array=time_array, amp_array_lst=amp_array_lst, scale_lst=scale_lst, sampling_rate=sampling_rate, elec_field=efield_sim, scalp_flag=False)
            fname = os.path.join(tuning_dir,'HP2_tuning_cortex_activation%d.png'%i)
            model.plot_cortex_activation(savepath=fname, show=False)
            np.save(os.path.join(tuning_dir, "SpikesHP2_Jtol%d.npy"%i), spikes_nerve_HP)
            np.save(os.path.join(tuning_dir, "Jtol%d_HP2.npy"%i), rand_tol[i])

            efield_sim._setJ(J_HP3)
            J_lst = [J_HP3]+[np.zeros(num_elec)]*7
            efield_sim._setJ_lst(J_lst)
            spikes_nerve_HP3, _ = model.stimulate(time_array=time_array, amp_array_lst=amp_array_lst, scale_lst=scale_lst, sampling_rate=sampling_rate, elec_field=efield_sim, scalp_flag=False)
            fname = os.path.join(tuning_dir,'HP3_tuning_cortex_activation%d.png'%i)
            model.plot_cortex_activation(savepath=fname, show=False)
            np.save(os.path.join(tuning_dir, "SpikesHP3_Jtol%d.npy"%i), spikes_nerve_HP)
            np.save(os.path.join(tuning_dir, "Jtol%d_HP3.npy"%i), rand_tol[i])

    for i in range(rand_tol.shape[0]):
        fname = os.path.join(tuning_dir, "SpikesHP_Jtol%d.npy"%i)
        if os.path.exists(fname):
            spikes_nerve_HP = np.load(fname)
            Jtol_HP_spikes_lst.append(np.sum(spikes_nerve_HP))
        else:
            Jtol_HP_spikes_lst.append(np.inf)

        fname = os.path.join(tuning_dir, "SpikesHP2_Jtol%d.npy"%i)
        if os.path.exists(fname):
            spikes_nerve_HP2 = np.load(fname)
            Jtol_HP2_spikes_lst.append(np.sum(spikes_nerve_HP2))
        else:
            Jtol_HP2_spikes_lst.append(np.inf)

        fname = os.path.join(tuning_dir, "SpikesHP3_Jtol%d.npy"%i)
        if os.path.exists(fname):
            spikes_nerve_HP3 = np.load(fname)
            Jtol_HP3_spikes_lst.append(np.sum(spikes_nerve_HP3))
        else:
            Jtol_HP3_spikes_lst.append(np.inf)

    Jtol_HP_spikes_lst = np.array(Jtol_HP_spikes_lst)
    np.save(os.path.join(tuning_dir, "JtolSpikes_lst_HP.npy"), Jtol_HP_spikes_lst)
    np.save(os.path.join(tuning_dir, "Jtol_lst_HP.npy"), rand_tol)

    Jtol_HP2_spikes_lst = np.array(Jtol_HP2_spikes_lst)
    np.save(os.path.join(tuning_dir, "JtolSpikes_lst_HP2.npy"), Jtol_HP2_spikes_lst)
    np.save(os.path.join(tuning_dir, "Jtol_lst_HP2.npy"), rand_tol)

    Jtol_HP3_spikes_lst = np.array(Jtol_HP3_spikes_lst)
    np.save(os.path.join(tuning_dir, "JtolSpikes_lst_HP3.npy"), Jtol_HP3_spikes_lst)
    np.save(os.path.join(tuning_dir, "Jtol_lst_HP3.npy"), rand_tol)

    xlabels = np.arange(rand_tol.shape[0])+1
    plt.barh(xlabels, list(Jtol_HP_spikes_lst))
    plt.xlabel("Diff. Jtol configurations", fontsize=19)
    plt.ylabel("Total Spikes", fontsize=19)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(tuning_dir,"TuningBarPlot_HP.png"))
    plt.close()

    xlabels = np.arange(rand_tol.shape[0])+1
    plt.barh(xlabels, list(Jtol_HP2_spikes_lst))
    plt.xlabel("Diff. Jtol configurations", fontsize=19)
    plt.ylabel("Total Spikes", fontsize=19)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(tuning_dir,"TuningBarPlot_HP2.png"))
    plt.close()

    xlabels = np.arange(rand_tol.shape[0])+1
    plt.barh(xlabels, list(Jtol_HP3_spikes_lst))
    plt.xlabel("Diff. Jtol configurations", fontsize=19)
    plt.ylabel("Total Spikes", fontsize=19)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(tuning_dir,"TuningBarPlot_HP3.png"))
    plt.close()

    Jtol_HP_spikes_lst = np.load(os.path.join(tuning_dir, "JtolSpikes_lst_HP.npy"))
    rand_tol_HP = np.load(os.path.join(tuning_dir, "Jtol_lst_HP.npy"))
    Jtol_best_HP = rand_tol[np.argmin(Jtol_HP_spikes_lst)]

    Jtol_HP2_spikes_lst = np.load(os.path.join(tuning_dir, "JtolSpikes_lst_HP2.npy"))
    rand_tol_HP2 = np.load(os.path.join(tuning_dir, "Jtol_lst_HP2.npy"))
    Jtol_best_HP2 = rand_tol_HP2[np.argmin(Jtol_HP2_spikes_lst)]

    Jtol_HP3_spikes_lst_HP3 = np.load(os.path.join(tuning_dir, "JtolSpikes_lst_HP3.npy"))
    rand_tol_HP3 = np.load(os.path.join(tuning_dir, "Jtol_lst_HP3.npy"))
    Jtol_best_HP3 = rand_tol_HP3[np.argmin(Jtol_HP3_spikes_lst)]

if not USE_TUNING_CODE:
    #### Manually Supply Jtol values. Default set to zero
    Jtol_best_HP = np.zeros(3)
    Jtol_best_HP2 = np.zeros(3)
    Jtol_best_HP3 = np.zeros(3)

print("Best Jtol configuration:", Jtol_best_HP)
print("Best Jtol configuration HP-2:", Jtol_best_HP2)
print("Best Jtol configuration HP-3:", Jtol_best_HP3)

Jtol_HP = [Jtol_best_HP[0]*Jdes, Jtol_best_HP[1]*Jdes, Jtol_best_HP[2]*Jdes]
Jtol_HP2 = [Jtol_best_HP2[0]*Jdes, Jtol_best_HP2[1]*Jdes, Jtol_best_HP2[2]*Jdes]
Jtol_HP3 = [Jtol_best_HP3[0]*Jdes, Jtol_best_HP3[1]*Jdes, Jtol_best_HP3[2]*Jdes]

rel_inc_HP, rel_inc_HP2, rel_inc_HP3 = [], [], []
start_loop = 0
LOAD_FLAG = False

#### Starting the simulation. Note that in this code we refer to the LCMV-E algorithm as SparsePlace due to legacy naming convention in our group. The loss function of SparsePlace and LCMV-E is exactly the same.

for i in range(start_loop, len(Itot_mul)):
    print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("Running Simulation for Total Current Multiplier %d....."%(Itot_mul[i]))
    print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    
    savedir_hp = os.path.join(hp_pttrn_dir, "TotalCurrent%d"%int(Itot_mul[i]))
    if not os.path.exists(savedir_hp):
        os.makedirs(savedir_hp)
    savedir_sp = os.path.join(sp_pttrn_dir, "TotalCurrent%d"%int(Itot_mul[i]))
    if not os.path.exists(savedir_sp):
        os.makedirs(savedir_sp)
    
    activ_HP, activ_HP2, activ_HP3, activ_SP = [], [], [], []
    if not LOAD_FLAG:
        start_loop_ii = 0
        for ii in range(start_loop_ii, len(Isafety_lst)):
            #########
            ## Figuring out J-tol (E_tol in the manuscriot). We provide the code for tuning Etol at each possible configuration of Isafety and Itot but this is computationally expensive....
            ## Set Jtol_tuning_once to false to use this code
            #########

            if (not Jtol_tuning_once) and USE_TUNING_CODE:
                LOAD = False
                tuning_dir = os.path.join(savedir_hp, 'Isafety%d/J-tol_Tuning' % (int(Isafety_lst[ii])))
                if not os.path.exists(tuning_dir):
                    os.makedirs(tuning_dir)
                num_points = 30
                np.random.seed(SEED)
                rand_tol = np.random.uniform(low=0.1, high=0.7, size=(num_points, 3))
                Jtol_HP_spikes_lst = []
                Jtol_HP2_spikes_lst = []
                Jtol_HP3_spikes_lst = []
                if not LOAD:
                    for i in range(rand_tol.shape[0]):
                        print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                        print("Iteration %d" % (i + 1))
                        print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

                        Isafety = Isafety_lst[ii]  ## Tunded for the middle injected current
                        Itotal = 2 * Isafety * Itot_mul[i]  ## Allow for HingePlace to design effective patterns

                        Jtol = [rand_tol[i, 0] * Jdes, rand_tol[i, 1] * Jdes, rand_tol[i, 2] * Jdes]
                        J_HP = efield_sim.HingePlace_3d(direction=direction, Jdes=Jdes, Isafety=Isafety, Jtol=Jtol, Itotal=Itotal, Af=[Af_x, Af_y, Af_z], Ac=[Ac_x, Ac_y, Ac_z], p=1)
                        J_HP2 = efield_sim.HingePlace_3d(direction=direction, Jdes=Jdes, Isafety=Isafety, Jtol=Jtol, Itotal=Itotal, Af=[Af_x, Af_y, Af_z], Ac=[Ac_x, Ac_y, Ac_z], p=2)
                        J_HP3 = efield_sim.HingePlace_3d(direction=direction, Jdes=Jdes, Isafety=Isafety, Jtol=Jtol, Itotal=Itotal, Af=[Af_x, Af_y, Af_z], Ac=[Ac_x, Ac_y, Ac_z],p=3)

                        efield_sim._setJ(J_HP)
                        J_lst = [J_HP] + [np.zeros(num_elec)] * 7
                        efield_sim._setJ_lst(J_lst)
                        spikes_nerve_HP, _ = model.stimulate(time_array=time_array, amp_array_lst=amp_array_lst, scale_lst=scale_lst, sampling_rate=sampling_rate, elec_field=efield_sim, scalp_flag=False)
                        fname = os.path.join(tuning_dir, 'HP_tuning_cortex_activation%d.png' % i)
                        model.plot_cortex_activation(savepath=fname, show=False)
                        np.save(os.path.join(tuning_dir, "SpikesHP_Jtol%d.npy" % i), spikes_nerve_HP)
                        np.save(os.path.join(tuning_dir, "Jtol%d_HP.npy" % i), rand_tol[i])

                        efield_sim._setJ(J_HP2)
                        J_lst = [J_HP2] + [np.zeros(num_elec)] * 7
                        efield_sim._setJ_lst(J_lst)
                        spikes_nerve_HP, _ = model.stimulate(time_array=time_array, amp_array_lst=amp_array_lst, scale_lst=scale_lst, sampling_rate=sampling_rate, elec_field=efield_sim, scalp_flag=False)
                        fname = os.path.join(tuning_dir, 'HP2_tuning_cortex_activation%d.png' % i)
                        model.plot_cortex_activation(savepath=fname, show=False)
                        np.save(os.path.join(tuning_dir, "SpikesHP2_Jtol%d.npy" % i), spikes_nerve_HP)
                        np.save(os.path.join(tuning_dir, "Jtol%d_HP2.npy" % i), rand_tol[i])

                        efield_sim._setJ(J_HP3)
                        J_lst = [J_HP3] + [np.zeros(num_elec)] * 7
                        efield_sim._setJ_lst(J_lst)
                        spikes_nerve_HP3, _ = model.stimulate(time_array=time_array, amp_array_lst=amp_array_lst, scale_lst=scale_lst, sampling_rate=sampling_rate, elec_field=efield_sim, scalp_flag=False)
                        fname = os.path.join(tuning_dir, 'HP3_tuning_cortex_activation%d.png' % i)
                        model.plot_cortex_activation(savepath=fname, show=False)
                        np.save(os.path.join(tuning_dir, "SpikesHP3_Jtol%d.npy" % i), spikes_nerve_HP)
                        np.save(os.path.join(tuning_dir, "Jtol%d_HP3.npy" % i), rand_tol[i])

                for i in range(rand_tol.shape[0]):
                    fname = os.path.join(tuning_dir, "SpikesHP_Jtol%d.npy" % i)
                    if os.path.exists(fname):
                        spikes_nerve_HP = np.load(fname)
                        Jtol_HP_spikes_lst.append(np.sum(spikes_nerve_HP))
                    else:
                        Jtol_HP_spikes_lst.append(np.inf)

                    fname = os.path.join(tuning_dir, "SpikesHP2_Jtol%d.npy" % i)
                    if os.path.exists(fname):
                        spikes_nerve_HP2 = np.load(fname)
                        Jtol_HP2_spikes_lst.append(np.sum(spikes_nerve_HP2))
                    else:
                        Jtol_HP2_spikes_lst.append(np.inf)

                    fname = os.path.join(tuning_dir, "SpikesHP3_Jtol%d.npy" % i)
                    if os.path.exists(fname):
                        spikes_nerve_HP3 = np.load(fname)
                        Jtol_HP3_spikes_lst.append(np.sum(spikes_nerve_HP3))
                    else:
                        Jtol_HP3_spikes_lst.append(np.inf)

                Jtol_HP_spikes_lst = np.array(Jtol_HP_spikes_lst)
                np.save(os.path.join(tuning_dir, "JtolSpikes_lst_HP.npy"), Jtol_HP_spikes_lst)
                np.save(os.path.join(tuning_dir, "Jtol_lst_HP.npy"), rand_tol)

                Jtol_HP2_spikes_lst = np.array(Jtol_HP2_spikes_lst)
                np.save(os.path.join(tuning_dir, "JtolSpikes_lst_HP2.npy"), Jtol_HP2_spikes_lst)
                np.save(os.path.join(tuning_dir, "Jtol_lst_HP2.npy"), rand_tol)

                Jtol_HP3_spikes_lst = np.array(Jtol_HP3_spikes_lst)
                np.save(os.path.join(tuning_dir, "JtolSpikes_lst_HP3.npy"), Jtol_HP3_spikes_lst)
                np.save(os.path.join(tuning_dir, "Jtol_lst_HP3.npy"), rand_tol)

                xlabels = np.arange(rand_tol.shape[0]) + 1
                plt.barh(xlabels, list(Jtol_HP_spikes_lst))
                plt.xlabel("Diff. Jtol configurations", fontsize=19)
                plt.ylabel("Total Spikes", fontsize=19)
                plt.xticks(fontsize=19)
                plt.yticks(fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(tuning_dir, "TuningBarPlot_HP.png"))
                plt.close()

                xlabels = np.arange(rand_tol.shape[0]) + 1
                plt.barh(xlabels, list(Jtol_HP2_spikes_lst))
                plt.xlabel("Diff. Jtol configurations", fontsize=19)
                plt.ylabel("Total Spikes", fontsize=19)
                plt.xticks(fontsize=19)
                plt.yticks(fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(tuning_dir, "TuningBarPlot_HP2.png"))
                plt.close()

                xlabels = np.arange(rand_tol.shape[0]) + 1
                plt.barh(xlabels, list(Jtol_HP3_spikes_lst))
                plt.xlabel("Diff. Jtol configurations", fontsize=19)
                plt.ylabel("Total Spikes", fontsize=19)
                plt.xticks(fontsize=19)
                plt.yticks(fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(tuning_dir, "TuningBarPlot_HP3.png"))
                plt.close()


                Jtol_HP_spikes_lst = np.load(os.path.join(tuning_dir, "JtolSpikes_lst_HP.npy"))
                rand_tol_HP = np.load(os.path.join(tuning_dir, "Jtol_lst_HP.npy"))
                Jtol_best_HP = rand_tol[np.argmin(Jtol_HP_spikes_lst)]

                Jtol_HP2_spikes_lst = np.load(os.path.join(tuning_dir, "JtolSpikes_lst_HP2.npy"))
                rand_tol_HP2 = np.load(os.path.join(tuning_dir, "Jtol_lst_HP2.npy"))
                Jtol_best_HP2 = rand_tol_HP2[np.argmin(Jtol_HP2_spikes_lst)]

                Jtol_HP3_spikes_lst_HP3 = np.load(os.path.join(tuning_dir, "JtolSpikes_lst_HP3.npy"))
                rand_tol_HP3 = np.load(os.path.join(tuning_dir, "Jtol_lst_HP3.npy"))
                Jtol_best_HP3 = rand_tol_HP3[np.argmin(Jtol_HP3_spikes_lst)]


                print("Best Jtol configuration:", Jtol_best_HP)
                print("Best Jtol configuration HP-2:", Jtol_best_HP2)
                print("Best Jtol configuration HP-3:", Jtol_best_HP3)

                Jtol_HP = [Jtol_best_HP[0] * Jdes, Jtol_best_HP[1] * Jdes, Jtol_best_HP[2] * Jdes]
                Jtol_HP2 = [Jtol_best_HP2[0] * Jdes, Jtol_best_HP2[1] * Jdes, Jtol_best_HP2[2] * Jdes]
                Jtol_HP3 = [Jtol_best_HP3[0] * Jdes, Jtol_best_HP3[1] * Jdes, Jtol_best_HP3[2] * Jdes]
            ###########################################################################

            Isafety = Isafety_lst[ii]
            Itotal = 2*Itot_mul[i]*Isafety
            print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            print("Iteration %d"%(ii+1))
            print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            J_HP = efield_sim.HingePlace_3d(direction=direction, Jdes=Jdes, Isafety=Isafety, Jtol=Jtol_HP, Itotal=Itotal, Af=[Af_x, Af_y, Af_z], Ac=[Ac_x, Ac_y, Ac_z])
            J_HP2 = efield_sim.HingePlace_3d(direction=direction, Jdes=Jdes, Isafety=Isafety, Jtol=Jtol_HP2, Itotal=Itotal, Af=[Af_x, Af_y, Af_z], Ac=[Ac_x, Ac_y, Ac_z], p=2)
            J_HP3 = efield_sim.HingePlace_3d(direction=direction, Jdes=Jdes, Isafety=Isafety, Jtol=Jtol_HP3, Itotal=Itotal, Af=[Af_x, Af_y, Af_z], Ac=[Ac_x, Ac_y, Ac_z], p=3)
            J_SP = efield_sim.SparsePlace_3d(direction=direction, Jdes=Jdes, Isafety=Isafety, Itotal=Itotal, Af=[Af_x, Af_y, Af_z], Ac=[Ac_x, Ac_y, Ac_z])
             
            SHOW = False
            PLOT_HP_PTTRN = True
            if PLOT_HP_PTTRN:
                efield_sim.plot_elec_pttrn(J=J_HP, x_lim=[-patch_size,patch_size], y_lim=[-patch_size,patch_size], fname=os.path.join(savedir_hp, 'HP_ElecPttrn_Isafety%dmA'%int(Isafety)), show=SHOW)
            
            PLOT_HP_CURR = True
            if PLOT_HP_CURR:
                efield_sim.plot_all_curr_density(r=radius_head-depth_neuron, J=J_HP, fname=os.path.join(savedir_hp, 'HP_CurrDensity_Isafety%dmA.png'%int(Isafety)), show=SHOW)
            
            PLOT_HP_PTTRN2 = True
            if PLOT_HP_PTTRN2:
                efield_sim.plot_elec_pttrn(J=J_HP2, x_lim=[-patch_size,patch_size], y_lim=[-patch_size,patch_size], fname=os.path.join(savedir_hp, 'HP2_ElecPttrn_Isafety%dmA'%int(Isafety)), show=SHOW)
            
            PLOT_HP_CURR2 = True
            if PLOT_HP_CURR2:
                efield_sim.plot_all_curr_density(r=radius_head-depth_neuron, J=J_HP2, fname=os.path.join(savedir_hp, 'HP2_CurrDensity_Isafety%dmA.png'%int(Isafety)), show=SHOW)

            PLOT_HP_PTTRN3 = True
            if PLOT_HP_PTTRN3:
                efield_sim.plot_elec_pttrn(J=J_HP3, x_lim=[-patch_size,patch_size], y_lim=[-patch_size,patch_size], fname=os.path.join(savedir_hp, 'HP3_ElecPttrn_Isafety%dmA'%int(Isafety)), show=SHOW)
            
            PLOT_HP_CURR3 = True
            if PLOT_HP_CURR3:
                efield_sim.plot_all_curr_density(r=radius_head-depth_neuron, J=J_HP3, fname=os.path.join(savedir_hp, 'HP3_CurrDensity_Isafety%dmA.png'%int(Isafety)), show=SHOW)
            
            PLOT_SP_PTTRN = True
            if PLOT_HP_PTTRN:
                efield_sim.plot_elec_pttrn(J=J_SP, x_lim=[-patch_size,patch_size], y_lim=[-patch_size,patch_size], fname=os.path.join(savedir_sp, 'SP_ElecPttrn_Isafety%dmA'%int(Isafety)), show=SHOW)
            
            PLOT_SP_CURR = True
            if PLOT_HP_CURR:
                efield_sim.plot_all_curr_density(r=radius_head-depth_neuron, J=J_SP, fname=os.path.join(savedir_sp, 'SP_CurrDensity_Isafety%dmA.png'%int(Isafety)), show=SHOW)
            
            #### Calculate HP-1 spikes
            efield_sim._setJ(J_HP)
            J_lst = [J_HP]+[np.zeros(num_elec)]*7
            efield_sim._setJ_lst(J_lst)
            
            spikes_HP, locations = model.stimulate(time_array=time_array, amp_array_lst=amp_array_lst, scale_lst=scale_lst, sampling_rate=sampling_rate, elec_field=efield_sim, scalp_flag=False)
            fname = os.path.join(savedir_hp,'HP_cortex_activation_Isafety%dmA.png'%int(Isafety))
            model.plot_cortex_activation(savepath=fname, show=False)           
            
            #### Calculate HP-2 spikes
            efield_sim._setJ(J_HP2)
            J_lst = [J_HP2]+[np.zeros(num_elec)]*7
            efield_sim._setJ_lst(J_lst)
            
            spikes_HP2, locations = model.stimulate(time_array=time_array, amp_array_lst=amp_array_lst, scale_lst=scale_lst, sampling_rate=sampling_rate, elec_field=efield_sim, scalp_flag=False)
            fname = os.path.join(savedir_hp,'HP2_cortex_activation_Isafety%dmA.png'%int(Isafety))
            model.plot_cortex_activation(savepath=fname, show=False)
            
            #### Calculate HP-3 spikes
            if input_baseline_pttrn!=3 or focus_pt_id==0:
                efield_sim._setJ(J_HP3)
                J_lst = [J_HP3]+[np.zeros(num_elec)]*7
                efield_sim._setJ_lst(J_lst)
                elec_field_lst = [efield_sim]+[None]*7
                
                spikes_HP3, locations = model.stimulate(time_array=time_array, amp_array_lst=amp_array_lst, scale_lst=scale_lst, sampling_rate=sampling_rate, elec_field=efield_sim, scalp_flag=False)
                fname = os.path.join(savedir_hp,'HP3_cortex_activation_Isafety%dmA.png'%int(Isafety))
                model.plot_cortex_activation(savepath=fname, show=False)
            
            ###### Calculate SP spikes
            efield_sim._setJ(J_SP)
            J_lst = [J_SP]+[np.zeros(num_elec)]*7
            efield_sim._setJ_lst(J_lst)

            scale_lst = [1] +[None]*7
            spikes_SP, _ = model.stimulate(time_array=time_array, amp_array_lst=amp_array_lst, scale_lst=scale_lst, sampling_rate=sampling_rate, elec_field=efield_sim, scalp_flag=False)
            fname = os.path.join(savedir_sp,'SP_cortex_activation_Isafety%dmA.png'%int(Isafety))
            model.plot_cortex_activation(savepath=fname, show=False)
            
            np.save(os.path.join(savedir_hp,"HP_spikes_Isafety%dmA.npy"%int(Isafety)), spikes_HP)
            np.save(os.path.join(savedir_hp,"HP2_spikes_Isafety%dmA.npy"%int(Isafety)), spikes_HP2)
            if input_baseline_pttrn!=3 or focus_pt_id==0:
                np.save(os.path.join(savedir_hp,"HP3_spikes_Isafety%dmA.npy"%int(Isafety)), spikes_HP3)
            np.save(os.path.join(savedir_sp,"SP_spikes_Isafety%dmA.npy"%int(Isafety)), spikes_SP)
            if i==0 and ii==0:
                np.save(os.path.join(baseline_dir,"neuron_locations.npy"), locations)
            print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            print("HP Spikes: %d"%int(np.sum(spikes_HP)))
            print("HP Spikes-2: %d"%int(np.sum(spikes_HP2)))
            if input_baseline_pttrn!=3 or focus_pt_id==0:
                print("HP Spikes-3: %d"%int(np.sum(spikes_HP3)))
            print("SP Spikes: %d"%int(np.sum(spikes_SP)))
            print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            
    for ii in range(len(Isafety_lst)):
        spikes_HP = np.load(os.path.join(savedir_hp,"HP_spikes_Isafety%dmA.npy"%int(Isafety_lst[ii])))
        spikes_HP2 = np.load(os.path.join(savedir_hp,"HP2_spikes_Isafety%dmA.npy"%int(Isafety_lst[ii])))
        if input_baseline_pttrn!=3 or focus_pt_id==0:
            spikes_HP3 = np.load(os.path.join(savedir_hp,"HP3_spikes_Isafety%dmA.npy"%int(Isafety_lst[ii])))
        spikes_SP = np.load(os.path.join(savedir_sp,"SP_spikes_Isafety%dmA.npy"%int(Isafety_lst[ii])))
        activ_HP.append(np.sum(spikes_HP))
        activ_HP2.append(np.sum(spikes_HP2))
        activ_HP3.append(np.sum(spikes_HP3))
        activ_SP.append(np.sum(spikes_SP))

    SHOW=False
    activ_HP, activ_HP2, activ_HP3, activ_SP = np.array(activ_HP), np.array(activ_HP2), np.array(activ_HP3), np.array(activ_SP)
    np.save(os.path.join(savedir_hp,'activation_HP.npy'), activ_HP)
    np.save(os.path.join(savedir_hp,'activation_HP2.npy'), activ_HP2)
    if input_baseline_pttrn!=3 or focus_pt_id==0:
        np.save(os.path.join(savedir_hp,'activation_HP3.npy'), activ_HP3)
    np.save(os.path.join(savedir_sp,'activation_SP.npy'), activ_SP)
    activ_HP, activ_HP2, activ_SP = activ_HP-1, activ_HP2-1, activ_SP-1
    if input_baseline_pttrn!=3 or focus_pt_id==0:
        activ_HP3 = activ_HP3-1
    plt.plot(Isafety_lst, activ_SP, marker='x', label='CDM')
    plt.plot(Isafety_lst, activ_HP, marker='x', label=r'HP:$p=1$')
    plt.plot(Isafety_lst, activ_HP2, marker='x', label=r'HP:$p=2$')
    if input_baseline_pttrn!=3 or focus_pt_id==0:
        plt.plot(Isafety_lst, activ_HP3, marker='x', label=r'HP:$p=3$')
    plt.legend(fontsize=19)
    plt.xlabel(r"$I_{safe}$ (mA)", fontsize=21)
    plt.ylabel("Total Spikes", fontsize=21)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.tight_layout()
    plt.savefig(os.path.join(baseline_dir,"HPvsSP_Itotal%d.png"%Itot_mul[i]))
    if SHOW:
        plt.show()
    else:
        plt.close()
    rel_inc_HP.append((activ_SP.copy()-activ_HP.copy())/activ_SP.copy()*100)
    rel_inc_HP2.append((activ_SP.copy()-activ_HP2.copy())/activ_SP.copy()*100)
    if input_baseline_pttrn!=3 or focus_pt_id==0:
        rel_inc_HP3.append((activ_SP.copy()-activ_HP3.copy())/activ_SP.copy()*100)

    print("#################################################################")
    print("Relative Decrease in Area of HP w.r.t. SP:", rel_inc_HP[i])
    print("Relative Decrease in Area of HP w.r.t. SP:", rel_inc_HP2[i])
    if input_baseline_pttrn!=3 or focus_pt_id==0:
        print("Relative Decrease in Area of HP w.r.t. SP:", rel_inc_HP3[i])
    print("#################################################################")

np.save(os.path.join(baseline_dir,"HPvsSP_RelInc.npy"), np.array(rel_inc_HP))
np.save(os.path.join(baseline_dir,"HP2vsSP_RelInc.npy"), np.array(rel_inc_HP2))
if input_baseline_pttrn!=3 or focus_pt_id==0:
    np.save(os.path.join(baseline_dir,"HP3vsSP_RelInc.npy"), np.array(rel_inc_HP3))

marker = ['^', "o", 'x', 's']
linestyle = ['dashdot', (0,(3,1,1,1,1,1)), 'dotted', 'solid']
color = ['blue', 'green', 'darkmagenta','crimson']
for i in range(len(Itot_mul)):       
    plt.plot(Isafety_lst, rel_inc_HP[i], marker=marker[0], label=r"$p=1$", linestyle=linestyle[0], color=color[0])
    plt.plot(Isafety_lst, rel_inc_HP2[i], marker=marker[1], label=r"$p=2$", linestyle=linestyle[1], color=color[1])
    if input_baseline_pttrn!=3 or focus_pt_id==0:
        plt.plot(Isafety_lst, rel_inc_HP3[i], marker=marker[2], label=r"$p=3$", linestyle=linestyle[2], color=color[2])
    ymax = np.max([np.max(rel_inc_HP[i]),np.max(rel_inc_HP2[i])])
    plt.hlines(y=ymax, xmin=np.min(Isafety_lst), xmax=np.max(Isafety_lst), linestyle='dashed', color='black', alpha=0.4)
    if ymax>60:
        plt.text(x=Isafety_lst[0], y=ymax*0.92, s=str(int(ymax))+r"$\%$", fontsize=25)
    else:
        plt.text(x=Isafety_lst[0], y=ymax*0.86, s=str(int(ymax))+r"$\%$", fontsize=25)
    plt.legend(fontsize=17, ncols=3, loc="upper left")
    plt.xlabel(r"$I_{safe}$ (mA)", fontsize=21)
    plt.ylabel("% Decrease Activation", fontsize=21)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    if ymax>60:
        plt.ylim(ymax=ymax*1.15)
    else:
        plt.ylim(ymax=ymax*1.25)
    plt.tight_layout()
    plt.savefig(os.path.join(baseline_dir,"HPvsSP_RelInc_Itot%d.png"%(Itot_mul[i])))
    plt.close()

marker = ['^', "o", 'x', 's']
linestyle = ['dashdot', (0,(3,1,1,1,1,1)), 'dotted', 'solid']
color = ['blue', 'green', 'darkmagenta']
for i in range(len(Itot_mul)):       
    plt.plot(Isafety_lst, rel_inc_HP[i], marker=marker[0], label=r"$I_{tot}^{mul}=%d, p=1$"%(Itot_mul[i]), linestyle=linestyle[0], color=color[i])
    plt.plot(Isafety_lst, rel_inc_HP2[i], marker=marker[1], label=r"$I_{tot}^{mul}=%d, p=2$"%(Itot_mul[i]), linestyle=linestyle[1], color=color[i])
    if input_baseline_pttrn!=3 or focus_pt_id==0:
        plt.plot(Isafety_lst, rel_inc_HP3[i], marker=marker[2], label=r"$I_{tot}^{mul}=%d, p=3$"%(Itot_mul[i]), linestyle=linestyle[2], color=color[i])
ymax = np.max([np.max(np.array(rel_inc_HP)),np.max(np.array(rel_inc_HP2)),np.max(np.array(rel_inc_HP3))])
plt.xlabel(r"$I_{safe}$ (mA)", fontsize=21)
plt.legend(fontsize=17, ncols=1, loc="center right")
plt.ylabel("% Decrease Activation", fontsize=21)
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
plt.xlim(xmax=np.max(Isafety_lst)*1.4)
plt.tight_layout()
plt.savefig(os.path.join(baseline_dir,"HPvsSP_RelInc.png"))
plt.close()