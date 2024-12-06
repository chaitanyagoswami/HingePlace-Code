import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import ray
import os
from neuron_model_parallel import NeuronSim
from elec_field import UniformField
from pulse_train import SingePulse_MonoPhasic
import sys
from nerve_and_cell_model_helper import fibonacci_sphere
import ray
import logging
ray.init(log_to_driver=False, logging_level=logging.FATAL)
##################################################################################
################## Uniform Experimental Setup ####################################
##################################################################################
SEED = 1234 
np.random.seed(SEED)
print("Setting Random Seed as %s"%(str(round(SEED,3))))
cwd = os.getcwd()
print("Working in the directory: %s. All data will be saved and loaded relative to this directory"%(cwd))
#### Defining Variables for Setting up Simulation

cell_id_lst = [16] ## Different Morphology for L23 Pyr Cells
human_or_mice = ray.put(0) ## 1->mice, 0-> human
temp = ray.put(37.0) ## Celsius, temparature at which neurons are simulated
dt = ray.put(0.025) ## ms, discretization time step
num_cores = 30 ## Number of Cores used for Parallelization
SHOW_PLOTS = False ## Flag used for showing or not showing plots
unit_vec = fibonacci_sphere(samples=300) ## Sampling 20 approximately uniformly spaced unit direction vectors along the electrode locations from ICMS study

angle = np.array([np.pi/2,0]) ## parameter used for specifying rotation of PV morphology
loc = np.array([0,0,0]) ## parameter used for specifying location of Pyr morphology

#### Plotting Directions and Neurons
###################################################################################
###################################################################################
SAVE_PATH = os.path.join(os.getcwd(),'HingePlace/HP_CorticalNeuron')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

PLOT_NEURON_AND_UNITVEC = False
if PLOT_NEURON_AND_UNITVEC:
    def plot_electrode_and_neuron(coord_elec, coord, savepath=None):
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        
        ax.quiver(coord_elec[:,0]*1200, coord_elec[:,1]*1200, coord_elec[:,2]*1200, coord_elec[:,0], coord_elec[:,1], coord_elec[:,2], length=200, linewidth=1.0, color='orange')
        ax.scatter(coord[:,0],coord[:,1],coord[:,2], linewidth=1.0, s=2.0)

        ax.set_xlabel('X-axis (um)', fontsize=14)
        ax.set_ylabel('Y-axis (um)', fontsize=14)
        ax.set_zlabel('Z-axis (um)', fontsize=14)
        ax.set_title('Neuron Orientation w.r.t\n Uniform Field ', fontsize=21)
        #for i in range(coord_elec.shape[0]):
        #    ax.text(coord_elec[i,0],coord_elec[i,1],coord_elec[i,2], 'MonoPolar Electrode')
        ax.tick_params(axis='x',labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='z',labelsize=12)
        ax.view_init(10,120)
        plt.savefig(savepath+'_orientation1.png')
        ax.view_init(10,240)
        plt.savefig(savepath+'_orientation2.png')
        ax.view_init(10,90)
        plt.savefig(savepath+'_orientation3.png')
        ax.view_init(10,0)
        plt.savefig(savepath+'_orientation4.png')
        view_angle = np.linspace(0,360,361)
        def update(frame):
            ax.view_init(10,view_angle[frame])
        ani = animation.FuncAnimation(fig=fig, func=update, frames=361, interval=20)
        ani.save(os.path.join(savepath+'.gif'), writer='pillow')
        ani.save(os.path.join(savepath+'.mp4'), writer='ffmpeg')
        plt.show()

    ################### Plot Neuron Coordinates ##################################################################

    for cell_id in cell_id_lst:
        
        ## Get Neuron Coordinates
        neuron = NeuronSim.remote(human_or_mice=human_or_mice, cell_id=cell_id, temp=temp, dt=dt)
        coord = ray.get(neuron._translate_rotate_neuron.remote(pos_neuron=loc, angle=angle))
        
        ## Plot ICMS Electrode With Neuron 
        savepath_curr = os.path.join(SAVE_PATH,'NeuronOrientation_Uniform_cellid'+str(cell_id)+'.png')
        plot_electrode_and_neuron(coord_elec=unit_vec, coord=coord, savepath=savepath_curr)
        
        del neuron   
    
        
    ############################################################################################################

#### Uniform Stimulation
###################################################################################
###################################################################################

## Generating Waveforms
start_time, time_taken_round = time.time(), 0
print("Generating Waveform...")
pulse_train = SingePulse_MonoPhasic()
amp = -1 ## mA
delay = 1 ## ms
total_time = 5 ##ms
sampling_rate = 1e6
pw = 0.2 ## ms
amp_array, time_array = pulse_train.amp_train(amp=amp, delay=delay, total_time=total_time, pw=pw, sampling_rate=sampling_rate)
amp_array_lst = ray.put([amp_array]+[None]*7)
scale_lst = ray.put([1]+[None]*7)
time_array = ray.put(time_array)
sampling_rate = ray.put(sampling_rate)
save_state_show = ray.put(False)
print("Waveform Generated! Time Taken %s s"%(str(round(time.time()-start_time,3))))

LOAD_DATA_FLAG = False
thresh = np.empty(len(unit_vec))
for l in range(len(unit_vec)):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<")
    print("Starting Simulation for Direction %d"%(l))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<")
    round_start_time = time.time() 
    
    ## Defining Directories for Saving DATA
    #########################################################################################
    SAVE_PATH_rawdata = os.path.join(SAVE_PATH, 'UnitVec'+str(l)+'/RawData')
    SAVE_PATH_plots = os.path.join(SAVE_PATH, 'UnitVec'+str(l)+'/Plots')
    if not os.path.exists(SAVE_PATH_rawdata):
        os.makedirs(SAVE_PATH_rawdata)
    if not os.path.exists(SAVE_PATH_plots):
        os.makedirs(SAVE_PATH_plots)
    
    if not LOAD_DATA_FLAG:
        ## Generate Electric Field Simulator
        start_time = time.time()
        print("Loading Electric Field Simulator...")
        elec_field = ray.put(UniformField(unit_vec=unit_vec[l]))
        print("Electric Field Simulator Loaded! Time Taken %s s"%(str(round(time.time()-start_time,3))))
        
        ### Run Pyr Stimulation
        ######################################################################################
        cell_id = ray.put(cell_id_lst[np.random.randint(len(cell_id_lst))]) ## Randomly choosing a Pyr Morphology out of the 5 available
        neuron = [NeuronSim.remote(human_or_mice=human_or_mice, cell_id=cell_id, temp=temp, dt=dt, elec_field=elec_field) for i in range(num_cores)] ## Initializing neuron model
        print("Pyramidal Cell Id chosen %d."%(int(ray.get(cell_id))))
        ray.get([neuron[i]._set_xtra_param.remote(angle=angle, pos_neuron=loc) for i in range(num_cores)]) ## Setting Extracellular Stim Paramaters
        delay_init, delay_final = ray.put(2000),ray.put(5) ## ms, delay added to the stimulation before and after applying stimulation

  
        ## Deciding the range of amplitude across which to stimulate neurons
        min_amp, max_amp = 0, 200 ## uA
        while True:
            amp_rough = np.linspace(min_amp, max_amp, num_cores)
            results_rough = [neuron[i].stimulate.remote(time_array=time_array, amp_array_lst=amp_array_lst, scale_lst=[amp_rough[i]]+[None]*7, sampling_rate=sampling_rate, delay_init=delay_init, delay_final=delay_final, save_state_show=save_state_show, plot=False) for i in range(num_cores)]
            results_rough = ray.get(results_rough)
            spikes = np.array([results_rough[i][1] for i in range(num_cores)]).flatten()
            min_amp = amp_rough[spikes<=1e-04].copy()
            min_amp = np.max(min_amp)*0.95
            amp_rough = amp_rough[spikes>0.1] 
            spikes = spikes[spikes>0.1] 
            if spikes.size == 0:
                max_amp = max_amp*1.1
                print("Maximum Threshold Too Low!!! Increasing it by 10%")
            else:
                break
        max_amp = np.min(amp_rough[spikes>0.1])*1.05
        print("Minimum and Maximum Threshold adjusted to %s V/m and %s V/m"%(str(round(min_amp,3)),str(round(max_amp,3))))
        amp_lst = np.linspace(min_amp, max_amp, num_cores) ## uA 
        np.save(os.path.join(SAVE_PATH_rawdata,'Amplitude.npy'), amp_lst)
        
        ## Providing pulse stimulation
        ########################################################################################
        start_time = time.time()
        print("Simulation for Pyr Neuron Started...")
        results = [neuron[i].stimulate.remote(time_array=time_array, amp_array_lst=amp_array_lst, scale_lst=[amp_lst[i]]+[None]*7, sampling_rate=sampling_rate, delay_init=delay_init, delay_final=delay_final) for i in range(num_cores)]
        results = ray.get(results)
        spikes = np.array([results[i][1] for i in range(num_cores)]).flatten()
        thresh[l] = np.min(amp_lst[spikes>0.1])
        print("Simulation Finished! Time Taken %s s"%(str(round(time.time()-start_time,3))))
        print("Calculated Threshold: %.2f"%(thresh[l]))
        ## Uncomment to see plots of membrane potential
        #ray.get([neuron[i].plot_sim_result.remote(delay_init=delay_init) for i in range(num_cores)])
        #########################################################################################

        del neuron
       
        ## Uncomment to see plots of membrane potential
        #ray.get([neuron[i].plot_sim_result.remote(delay_init=delay_init) for i in range(num_cores)])
        #########################################################################################


        ## Saving Data
        #######################################################################################
        start_time = time.time()
        print("Saving Raw Data for Direction %d..."%(l))
        np.save(os.path.join(SAVE_PATH_rawdata,'Spikes.npy'), spikes)
        print("Raw Data Saved for Direction %d! Time Taken %s s"%(l,str(round(time.time()-start_time,3))))

        ## Plotting Results    
        #######################################################################################
        
        start_time = time.time()
        print("Plotting Data for Direction %d..."%(l))
        
        ## Cell-Type Comparison for Cell-Type Comparison 
        plt.plot(amp_lst, spikes, marker='x', color='green')
        plt.title('Unit Vec %d: IP/OP'%l, fontsize=22)
        plt.xlabel('Amp of Electric Field (V/m)', fontsize=19)
        plt.ylabel('Spikes', fontsize=19)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_PATH_plots,'IP-OP.png'))
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close()
        print("Plots Saved for Direction %d! Time Taken %s s"%(l,str(round(time.time()-start_time,3))))
    
    else:
        ## Loading Data
        #########################################################################################
        start_time = time.time()
        print("Loading Raw Data for Electrode location %d..."%(l))
        amp_lst = np.load(os.path.join(SAVE_PATH_rawdata,'Amplitude.npy'))
        spikes = np.load(os.path.join(SAVE_PATH_rawdata,'Spikes.npy'))
        print("Raw Data Loaded for Electrode location %d! Time Taken %s s"%(l,str(round(time.time()-start_time,3))))

    time_taken_round = time_taken_round*(l)/(l+1)+(time.time()-round_start_time)/(l+1)
    ETA = ((len(unit_vec)-l-1)*time_taken_round)/3600
    print("Simulation Finished for Direction %d! Time Taken %s hr. ETA for script to finish: %s hr"%(l, str(round((time.time()-round_start_time)/3600,3)),str(round(ETA,3))))

if not LOAD_DATA_FLAG:
    #### Saving Processed DATA
    np.save(os.path.join(SAVE_PATH, 'activ_thresh.npy'), thresh)
else:
    thresh = np.load(os.path.join(SAVE_PATH, 'activ_thresh.npy'))

best_unit_vec = unit_vec[np.argmin(thresh)]
np.save(os.path.join(SAVE_PATH,"best_unit_vec.npy"), best_unit_vec)
print("Optimal Unit Vector: %s"%(np.round(best_unit_vec,2)))

#### Plotting Activation Threshold
###################################################################################
###################################################################################

plt.plot(np.arange(len(unit_vec))+1,thresh, 'x')
plt.title('Threshold For Different Directions', fontsize=22)
plt.xlabel("Idxs of Unit Vectors", fontsize=20)
plt.ylabel("E-field Amp (V/m)", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=18)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH,"thresh.png"))
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()


