from neuron_model_parallel import NeuronSim
from matplotlib import colormaps
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib import cm
import plotly
import pandas as pd
import random
import plotly
from matplotlib import cm
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.signal import find_peaks
import time
import ray
import logging
from cfiber_parallel import CfiberSim
from nerve_and_cell_model_helper import plot_scalp_model, sample_spherical, cart_to_sph, sph_to_cart, fibonacci_sphere, cart_to_sph_v2, sph_to_cart_v2

class nerve_and_cell_model:

    def __init__(self, neuron_params, nerve_params, num_cores_nerve, num_cores_neuron, overall_radius=9.2, depth_nerve=0.1, depth_neuron=1.5, SEED=5241):
        
        self.SEED = SEED
        #### Spherical Model Params
        self.overall_radius = overall_radius ## cm
        self.depth_nerve = depth_nerve ## cm
        self.depth_neuron = depth_neuron ## cm
        self.r_neuron = self.overall_radius-depth_neuron ## cm
        self.r_nerve = ray.put(self.overall_radius-depth_nerve) ## cm

        #### Cores Used for Parallelization
        self.num_cores_nerve = num_cores_nerve
        self.num_cores_neuron = num_cores_neuron

        #### Cfiber Params
        self.Vinit_nerve = ray.put(nerve_params[0])
        self.temp_nerve = ray.put(nerve_params[1]) ## celsius
        self.dt_nerve = ray.put(nerve_params[2]) ## ms
        self.periphery_only = ray.put(nerve_params[3]) 
        self.plot_nerve = ray.put(nerve_params [4])
        self.delay_init_nerve, self.delay_final_nerve = ray.put(nerve_params[5]), ray.put(nerve_params[6])

        self.length_parietal = nerve_params[7]
        self.length_parietal_xlen = nerve_params[8]
        self.length_temporal_ylen = nerve_params[9]
        self.spacing = nerve_params[10]
        
        #### Neuron Params
        self.Vinit_neuron = ray.put(neuron_params[0]) 
        self.save_state_show = ray.put(neuron_params[1])
        self.plot_neuron = ray.put(neuron_params[2])
        self.human_or_mice = ray.put(neuron_params[3])
        self.cell_id_pyr_lst = neuron_params[4] ## Different Morphology for L23 Pyr Cells
        self.temp_neuron = ray.put(neuron_params[5]) ## celsius
        self.dt_neuron = ray.put(neuron_params[6]) ## ms
        self.delay_init_neuron, self.delay_final_neuron = ray.put(neuron_params[7]), ray.put(neuron_params[8])

        self.num_neurons = neuron_params[9]
        self.theta_max = neuron_params[10]
        self.x_max = neuron_params[11] 
        self.y_max = neuron_params[12]
        self._init_neuron_locations(num_neurons=self.num_neurons, theta_max=self.theta_max, cell_id_pyr_lst=self.cell_id_pyr_lst, x_max=self.x_max, y_max=self.y_max, r=self.r_neuron, SEED=self.SEED)

    def _init_neuron_locations(self,num_neurons,theta_max, y_max, x_max, r, cell_id_pyr_lst, SEED=5321):
        np.random.seed(SEED)
        self.points_samples = sample_spherical(num_samples=num_neurons, theta_max=theta_max, y_max=y_max, x_max=x_max, r=r)
        self.angle = cart_to_sph_v2(self.points_samples)
        self.angle[:,1] = np.pi/2-self.angle[:,1]
        self.angle = np.hstack([self.angle[:,1].copy().reshape(-1,1),self.angle[:,2].copy().reshape(-1,1)]) ## parameter used for specifying rotation of Pyr morphology
        self.cell_id_pyr = cell_id_pyr_lst[np.random.randint(len(cell_id_pyr_lst), size=self.points_samples.shape[0])] ## Randomly choosing a Pyr Morphology out of the 5 available
    
    def plot_points_to_sample(self,coord_elec, J, savepath, depth_skull, depth_CSF, depth, xlim, ylim):
        fname=savepath
        skull_samples = np.random.normal(loc=0, scale=1, size=(10**4,3))
        skull_samples = skull_samples/np.sqrt(np.sum(skull_samples**2, axis=1)).reshape(-1,1)*(self.overall_radius-depth_skull)
        skull_samples = cart_to_sph(skull_samples)
        skull_samples = skull_samples[skull_samples[:,1]>(np.pi/2-7/self.overall_radius)]
        skull_samples = sph_to_cart(skull_samples)
        scalp_samples = skull_samples.copy()/(self.overall_radius-depth_skull)*(self.overall_radius)
        csf_samples = skull_samples.copy()/(self.overall_radius-depth_skull)*(9.2-depth_CSF)
        brain_samples = skull_samples.copy()/(self.overall_radius-depth_skull)*(self.overall_radius-depth)
        
        skull_samples =  skull_samples[skull_samples[:,1]<=ylim[1]]
        skull_samples =  skull_samples[skull_samples[:,1]>=ylim[0]]
        skull_samples =  skull_samples[skull_samples[:,1]<=xlim[1]]
        skull_samples =  skull_samples[skull_samples[:,1]>=xlim[0]]       
        
        scalp_samples =  scalp_samples[scalp_samples[:,1]<=ylim[1]]
        scalp_samples =  scalp_samples[scalp_samples[:,1]>=ylim[0]]
        scalp_samples =  scalp_samples[scalp_samples[:,1]<=xlim[1]]
        scalp_samples =  scalp_samples[scalp_samples[:,1]>=xlim[0]]
        
        csf_samples =  csf_samples[csf_samples[:,1]<=ylim[1]]
        csf_samples =  csf_samples[csf_samples[:,1]>=ylim[0]]
        csf_samples =  csf_samples[csf_samples[:,1]<=xlim[1]]
        csf_samples =  csf_samples[csf_samples[:,1]>=xlim[0]]
        
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        if J is not None:
            img = ax.scatter(coord_elec[J>0,0], coord_elec[J>0,1], coord_elec[J>0,2], linewidth=0.3, s=100, color='red')
            img = ax.scatter(coord_elec[J<0,0], coord_elec[J<0,1], coord_elec[J<0,2], linewidth=0.3, s=100, color='black')
        else:
            img = ax.scatter(coord_elec[:,0], coord_elec[:,1], coord_elec[:,2], linewidth=0.3, s=50, color='blue')

        img = ax.scatter(skull_samples[0,0], skull_samples[0,1], skull_samples[0,2], linewidth=0.3, s=10, color='grey', alpha=1, label='Skull')
        img = ax.scatter(skull_samples[:,0], skull_samples[:,1], skull_samples[:,2], linewidth=0.3, s=10, color='grey', alpha=0.1)

        img = ax.scatter(scalp_samples[0,0], scalp_samples[0,1], scalp_samples[0,2], linewidth=0.3, s=10, color='salmon', alpha=1, label='Scalp')
        img = ax.scatter(scalp_samples[:,0], scalp_samples[:,1], scalp_samples[:,2], linewidth=0.3, s=10, color='salmon', alpha=0.1)

        img = ax.scatter(csf_samples[0,0], csf_samples[0,1], csf_samples[0,2], linewidth=0.3, s=10, color='deepskyblue', alpha=1, label='CSF')
        img = ax.scatter(csf_samples[:,0], csf_samples[:,1], csf_samples[:,2], linewidth=0.3, s=10, color='deepskyblue', alpha=0.1)

        img = ax.scatter(brain_samples[0,0], brain_samples[0,1], brain_samples[0,2], linewidth=0.3, s=10, color='crimson',alpha=1, label='Brain')
        img = ax.scatter(brain_samples[:,0], brain_samples[:,1], brain_samples[:,2], linewidth=0.3, s=10, color='crimson',alpha=0.1)

        for i in range(self.points_samples.shape[0]):
            neuron = NeuronSim.remote(human_or_mice=self.human_or_mice, cell_id=self.cell_id_pyr[i], temp=self.temp_neuron, dt=self.dt_neuron, elec_field=None)        
            coord = ray.get(neuron._translate_rotate_neuron.remote(pos_neuron=self.points_samples[i]*10**4, angle=self.angle[i]))
            img = ax.scatter(coord[:,0]*10**(-4),coord[:,1]*10**(-4),coord[:,2]*10**(-4), linewidth=0.1, s=0.1)


        ax.set_xlabel('X-axis (cm)', fontsize=14)
        ax.set_ylabel('Y-axis (cm)', fontsize=14)
        ax.set_zlabel('Z-axis (cm)', fontsize=14)
        
        plt.legend(ncols=4) 
        
        ax.tick_params(axis='x',labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='z',labelsize=12)
        ax.view_init(20,0)
        plt.savefig(fname+"_orient1.png")
        ax.view_init(20,90)
        plt.savefig(fname+"_orient2.png")        
        ax.view_init(20,180)
        plt.savefig(fname+"_orient3.png")
        ax.view_init(20,270)
        plt.savefig(fname+"_orient4.png")        
        ax.view_init(90,0)
        plt.savefig(fname+"_orient5.png")
        ax.view_init(90,180)
        plt.savefig(fname+"_orient6.png")        
        ax.view_init(45,45)
        plt.savefig(fname+"_orient7.png")        
        view_angle = np.linspace(0,360,361)
        def update(frame):
            ax.view_init(25,view_angle[frame])
        ##### Uncomment if you want to see an animation
        #ani = animation.FuncAnimation(fig=fig, func=update, frames=361, interval=20)
        #ani.save(os.path.join(savepath+'.gif'), writer='pillow')
        #ani.save(os.path.join(savepath+'.mp4'), writer='ffmpeg')
        #plt.show()
        plt.close()


    def _cortical_neuron_model(self, num_neurons=250,num_cores=50, theta_max=np.pi/2-8/9.2, y_max=7, x_max=7, r=9.2-1.5, human_or_mice=1, temp=37, cell_id_pyr_lst=[6,7,8,9,10], dt=0.025, elec_field=None, time_array=None, amp_array_lst=None, delay_init=2000, delay_final=5, scale_lst=None, sampling_rate=1e06, plot=False, save_state_show=False, Vinit=-65, SEED=5321):
        
        np.random.seed(SEED)
        split_num = int(np.floor(num_neurons/num_cores)) 
    
        ### Checking Pyr Firing Rates in a 7x7 region
        #######################################################################################
        angle = np.array_split(self.angle, split_num, axis=0)
        loc = np.array_split(self.points_samples*10**4, split_num, axis=0) ## cm->um, parameter used for specifying location of Pyr morphology
        
        ### Run Pyr Stimulation
        ######################################################################################
        cell_id_pyr = np.array_split(self.cell_id_pyr, split_num, axis=0)
        start_time = time.time()
        fr_rate = []
        for num in range(split_num):
            neuron = [NeuronSim.remote(human_or_mice=human_or_mice, cell_id=cell_id_pyr[num][i], temp=temp, dt=dt, elec_field=elec_field) for i in range(loc[num].shape[0])] ## Initializing neuron model
            divide_factor = 1
            length_divide = len(neuron)//divide_factor
            for i in range(divide_factor):
                if i!=divide_factor-1:
                    ray.get([neuron[i]._set_xtra_param.remote(angle=angle[num][i], pos_neuron=loc[num][i]) for i in range(i*length_divide, (i+1)*length_divide)]) ## Setting Extracellular Stim Paramaters
                else:
                    ray.get([neuron[i]._set_xtra_param.remote(angle=angle[num][i], pos_neuron=loc[num][i]) for i in range(i*length_divide, len(neuron))]) ## Setting Extracellular Stim Paramaters

            ########################################################################################
            ## Stimulation
            print("Number of neurons simulated: %d"%((num+1)*num_cores))
            results = [neuron[i].stimulate.remote(time_array=time_array, amp_array_lst=amp_array_lst, scale_lst=scale_lst, sampling_rate=sampling_rate, delay_init=delay_init, delay_final=delay_final, save_state_show=save_state_show, plot=plot, Vinit=Vinit) for i in range(num_cores)]
            results = ray.get(results)
            fr_rate.append(np.array([results[i][1] for i in range(num_cores)]).flatten()) 
            del neuron
        spikes = np.hstack(fr_rate)
        return spikes, self.points_samples
    
    def _cfiber_scalp_model(self, length_parietal=16, length_parietal_xlen=18, length_temporal_ylen=10, spacing=0.5, temp=37, dt=0.025, periphery_only=True, r=9.2, num_cores=50, elec_field=None, delay_init=20, delay_final=2, plot=False, Vinit=-60, save_state_show=False, amp_array_lst=None, scale_lst=None, time_array=None, sampling_rate=1e6, SEED=5321):
        
        np.random.seed(SEED)
        r_nerve = ray.get(self.r_nerve)
        shifts_parietal = np.arange(-length_parietal_xlen/2+1,length_parietal_xlen/2-1, spacing)
        shifts_temporal = np.arange(-length_temporal_ylen/2, length_temporal_ylen/2, spacing)
        temporal_flip = [False]*len(shifts_parietal)+[True]*len(shifts_temporal)+[False]*len(shifts_temporal)
        shifts_temporal = np.concatenate([shifts_temporal.copy(), shifts_temporal.copy()], axis=0)
        total_fibers  = len(shifts_parietal)+len(shifts_temporal)
        shifts = np.concatenate([shifts_parietal, shifts_temporal], axis=0)
        temporal = [False]*len(shifts_parietal)+[True]*len(shifts_temporal)
        length = [length_parietal]*len(shifts_parietal)+[length_parietal_xlen]*len(shifts_temporal)
        num_rounds = total_fibers//num_cores 
        spikes, locations = [], []
        for i in range(num_rounds+1):
            print("Number of nerves simulated: %d"%((i+1)*num_cores))
            if i<num_rounds:
                cfibers = [CfiberSim.remote(length=length[i*num_cores+j], temp=temp, dt=dt, periphery_only=periphery_only, temporal=temporal[i*num_cores+j], temporal_flip=temporal_flip[i*num_cores+j], shift=shifts[i*num_cores+j], r=r, elec_field=elec_field) for j in range(num_cores)]
                ray.get([cfibers[j]._set_xtra_param.remote() for j in range(len(cfibers))])
                results = ray.get([cfibers[j].stimulate.remote(time_array=time_array, amp_array_lst=amp_array_lst, scale_lst=scale_lst, sampling_rate=sampling_rate, delay_init=delay_init, delay_final=delay_final, plot=plot, Vinit=Vinit) for j in range(len(cfibers))])
                spike = np.array([results[j][1] for j in range(len(cfibers))]).flatten()
                location = np.array([results[j][3] for j in range(len(cfibers))]).reshape(-1,3)
                spikes.append(spike)
                locations.append(location)
            else:
                cfibers = [CfiberSim.remote(length=length[i*num_cores+j], temp=temp, dt=dt, periphery_only=periphery_only, temporal=temporal[i*num_cores+j], temporal_flip=temporal_flip[i*num_cores+j], shift=shifts[i*num_cores+j], r=r, elec_field=elec_field) for j in range(total_fibers-i*num_cores)]
                ray.get([cfibers[j]._set_xtra_param.remote() for j in range(len(cfibers))])   
                results = ray.get([cfibers[j].stimulate.remote(time_array=time_array, amp_array_lst=amp_array_lst, scale_lst=scale_lst, sampling_rate=sampling_rate, delay_init=delay_init, delay_final=delay_final, plot=plot, Vinit=Vinit) for j in range(len(cfibers))])
                spike = np.array([results[j][1] for j in range(len(cfibers))]).flatten()
                location = np.array([results[j][3] for j in range(len(cfibers))]).reshape(-1,3)
                spikes.append(spike)
                locations.append(location)
            del cfibers
        spikes = np.concatenate(spikes, axis=0)
        locations = np.concatenate(locations, axis=0)*10**(-4)
        return spikes, locations

    def stimulate(self, time_array, amp_array_lst, scale_lst, sampling_rate, elec_field, scalp_flag=True, cortex_flag=True):
        
        elec_field = ray.put(elec_field)
        scale_lst = ray.put(scale_lst)
        sampling_rate = ray.put(sampling_rate)
        amp_array_lst_scalp, amp_array_lst_cortex = [], []
        scale_scalp, scale_cortex = 1,1#2.5, 1.6
        for i in range(len(amp_array_lst)):
            if amp_array_lst[i] is None:
                amp_array_lst_scalp.append(None)
                amp_array_lst_cortex.append(None)
            else:
                amp_array_lst_scalp.append(scale_scalp*amp_array_lst[i])
                amp_array_lst_cortex.append(scale_cortex*amp_array_lst[i])

        time_array, amp_array_lst = ray.put(time_array), ray.put(amp_array_lst)
        amp_array_lst_scalp, amp_array_lst_cortex = ray.put(amp_array_lst_scalp), ray.put(amp_array_lst_cortex) 
        #### C-fiber Stimulation
        start_time = time.time()
        if scalp_flag:
            print("Stimulating Nerve Fibers")
            
            self.spikes_nerve, self.locations_nerve = self._cfiber_scalp_model(length_parietal=self.length_parietal, length_parietal_xlen=self.length_parietal_xlen, length_temporal_ylen=self.length_temporal_ylen, spacing=self.spacing, temp=self.temp_nerve, dt=self.dt_nerve, periphery_only=self.periphery_only, r=self.r_nerve, num_cores=self.num_cores_nerve, elec_field=elec_field, delay_init=self.delay_init_nerve, delay_final=self.delay_final_nerve, plot=self.plot_nerve, Vinit=self.Vinit_nerve, amp_array_lst=amp_array_lst_scalp, scale_lst=scale_lst, time_array=time_array, sampling_rate=sampling_rate, SEED=self.SEED)
            print("Nerve Fibers Stimulated!!!! Time Takes: %.2f s"%(time.time()-start_time))
        
        #### Neuron Stimulation
        if cortex_flag:
            print("Stimulating Cortical Neurons")
            self.spikes_neuron, self.locations_neuron = self._cortical_neuron_model(num_neurons=self.num_neurons, num_cores=self.num_cores_neuron, theta_max=self.theta_max, y_max=self.y_max, x_max=self.x_max, r=self.r_neuron, human_or_mice=self.human_or_mice, temp=self.temp_neuron, cell_id_pyr_lst=self.cell_id_pyr_lst, SEED=self.SEED, dt=self.dt_neuron, elec_field=elec_field, time_array=time_array, amp_array_lst=amp_array_lst_cortex, delay_init=self.delay_init_neuron, delay_final=self.delay_final_neuron, scale_lst=scale_lst, sampling_rate=sampling_rate, save_state_show=self.save_state_show, plot=self.plot_neuron, Vinit=self.Vinit_neuron)
            print("Cortical Neurons Stimulated!!!! Time Takes: %.2f s"%(time.time()-start_time))
        if scalp_flag and cortex_flag:
            return self.spikes_nerve, self.locations_nerve, self.spikes_neuron, self.locations_neuron
        elif scalp_flag and not cortex_flag:
            return self.spikes_nerve, self.locations_nerve
        elif not scalp_flag and cortex_flag:
            return self.spikes_neuron, self.locations_neuron

    def plot_cortex_activation(self, savepath=None, show=True, spikes_neuron=None):
        color_map = colormaps['jet']
        if spikes_neuron is None:
            img = plt.scatter(self.points_samples[:,0],self.points_samples[:,1],c=self.spikes_neuron, vmin=0, vmax=np.max(self.spikes_neuron), cmap=color_map, linewidth=2.0)
        else:
            img = plt.scatter(self.points_samples[:,0],self.points_samples[:,1],c=spikes_neuron, vmin=0, vmax=np.max(spikes_neuron), cmap=color_map, linewidth=2.0)

        plt.colorbar(img)
        plt.xlabel('X-axis (cm)', fontsize=19)
        plt.ylabel('Y-axis (cm)', fontsize=19)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.title('Cortex Num Spikes', fontsize=22)
        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath)
        if show:
            plt.show()
        else:
            plt.close()

    def plot_neuron_orient(self, fname=None, show=True):
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        
        for i in range(self.points_samples.shape[0]):
            if np.abs(self.points_samples[i,0])<0.5 and np.abs(self.points_samples[i,1])<0.5:
                print(i,self.points_samples.shape[0], self.points_samples[i])
                neuron = NeuronSim.remote(human_or_mice=self.human_or_mice, cell_id=self.cell_id_pyr[i], temp=self.temp_neuron, dt=self.dt_neuron, elec_field=None)        
                coord = ray.get(neuron._translate_rotate_neuron.remote(pos_neuron=self.points_samples[i]*10**4, angle=self.angle[i]))
                img = ax.scatter(coord[:,0]*10**(-4),coord[:,1]*10**(-4),coord[:,2]*10**(-4), linewidth=0.1, s=0.1)
        cortical_layer = sample_spherical(num_samples=10000, theta_max=self.theta_max, y_max=0.5, x_max=0.5, r=self.r_neuron)
        img = ax.scatter(cortical_layer[:,0], cortical_layer[:,1], cortical_layer[:,2], linewidth=0.3, alpha=0.01, s=10, color='black')
        scalp_layer = sample_spherical(num_samples=10000, theta_max=self.theta_max, y_max=0.5, x_max=0.5, r=8)
        img = ax.scatter(scalp_layer[:,0], scalp_layer[:,1], scalp_layer[:,2], linewidth=0.3, s=10, color='crimson',alpha=0.01)

        ax.set_xlabel('X-axis (cm)', fontsize=16)
        ax.set_ylabel('Y-axis (cm)', fontsize=16)
        ax.set_zlabel('Z-axis (cm)', fontsize=16)
        ax.set_title('Neuron Orientation w.r.t\n  Cortical Surface', fontsize=21)
        ax.tick_params(axis='x',labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='z',labelsize=12)
        plt.tight_layout()
        if fname is not None:
            ax.view_init(20,0)
            plt.savefig(fname+"_orient1.png")
            ax.view_init(20,90)
            plt.savefig(fname+"_orient2.png")        
            ax.view_init(20,180)
            plt.savefig(fname+"_orient3.png")
            ax.view_init(20,270)
            plt.savefig(fname+"_orient4.png")        
            ax.view_init(90,0)
            plt.savefig(fname+"_orient5.png")
            ax.view_init(90,180)
            plt.savefig(fname+"_orient6.png")        
            ax.view_init(45,45)
            plt.savefig(fname+"_orient7.png")        
        if show:
            plt.show()
        else:
            plt.clf()
            plt.close()

    def plot_scalp_activation(self, savepath=None, show=True, spikes_nerve=None):
        r_nerve = ray.get(self.r_nerve)
        shifts_parietal = np.arange(-self.length_parietal_xlen/2+1,self.length_parietal_xlen/2-1, self.spacing)
        shifts_temporal = np.arange(-self.length_temporal_ylen/2, self.length_temporal_ylen/2, self.spacing)
        temporal_flip = [False]*len(shifts_parietal)+[True]*len(shifts_temporal)+[False]*len(shifts_temporal)
        shifts_temporal = np.concatenate([shifts_temporal.copy(), shifts_temporal.copy()], axis=0)
        total_fibers  = len(shifts_parietal)+len(shifts_temporal)
        shifts = np.concatenate([shifts_parietal, shifts_temporal], axis=0)
        temporal = [False]*len(shifts_parietal)+[True]*len(shifts_temporal)
        length_parietal_xlen = ray.get(self.r_nerve)-ray.get(self.r_nerve)*np.sin(self.length_parietal_xlen/(2*ray.get(self.r_nerve)))
        length = [self.length_parietal]*len(shifts_parietal)+[length_parietal_xlen]*len(shifts_temporal)
        
        shifts_parietal=[r_nerve*np.sin(shifts_parietal[i]/r_nerve) for i in range(len(shifts_parietal))]
        
        for i in range(len(shifts_parietal)):
            if spikes_nerve is None:
                if self.spikes_nerve[i]!=0:
                    plt.vlines(x=shifts_parietal[i], ymin=-length[i]/2, ymax=length[i]/2, linestyle='-', linewidth=3, color='tab:red')
                    plt.scatter(self.locations_nerve[i,0], self.locations_nerve[i,1], c='tab:red', s=50)
                else:
                    plt.vlines(x=shifts_parietal[i], ymin=-length[i]/2, ymax=length[i]/2, linestyle='-', linewidth=3, color='tab:blue')
            else:
                if spikes_nerve[i]!=0:
                    plt.vlines(x=shifts_parietal[i], ymin=-length[i]/2, ymax=length[i]/2, linestyle='-', linewidth=3, color='tab:red')
                    #plt.scatter(self.locations_nerve[i,0], self.locations_nerve[i,1], c='tab:red', s=50)
                else:
                    plt.vlines(x=shifts_parietal[i], ymin=-length[i]/2, ymax=length[i]/2, linestyle='-', linewidth=3, color='tab:blue')
        

        
        for i in range(len(shifts_temporal)):
            if spikes_nerve is None:
                if self.spikes_nerve[i+len(shifts_parietal)]!=0:
                    if temporal_flip[i+len(shifts_parietal)]:
                        plt.hlines(y=shifts_temporal[i], xmin=ray.get(self.r_nerve)-length[i+len(shifts_parietal)], xmax=ray.get(self.r_nerve), linestyle='-', linewidth=3, color='tab:red')
                    else:
                        plt.hlines(y=shifts_temporal[i], xmin=-ray.get(self.r_nerve), xmax=-ray.get(self.r_nerve)+length[i+len(shifts_parietal)], linestyle='-', linewidth=3, color='tab:red')
                    #plt.scatter(self.locations_nerve[i+len(shifts_parietal),0], self.locations_nerve[i+len(shifts_parietal),1], c='tab:red', s=50)
                else:
                    if temporal_flip[i+len(shifts_parietal)]:
                        plt.hlines(y=shifts_temporal[i], xmin=ray.get(self.r_nerve)-length[i+len(shifts_parietal)], xmax=ray.get(self.r_nerve), linestyle='-', linewidth=3, color='tab:blue')
                    else:
                        plt.hlines(y=shifts_temporal[i], xmin=-ray.get(self.r_nerve), xmax=-ray.get(self.r_nerve)+length[i+len(shifts_parietal)], linestyle='-', linewidth=3, color='tab:blue')
            else:
                if spikes_nerve[i+len(shifts_parietal)]!=0:
                    if temporal_flip[i+len(shifts_parietal)]:
                        plt.hlines(y=shifts_temporal[i], xmin=ray.get(self.r_nerve)-length[i+len(shifts_parietal)], xmax=ray.get(self.r_nerve), linestyle='-', linewidth=3, color='tab:red')
                    else:
                        plt.hlines(y=shifts_temporal[i], xmin=-ray.get(self.r_nerve), xmax=-ray.get(self.r_nerve)+length[i+len(shifts_parietal)], linestyle='-', linewidth=3, color='tab:red')
                    #plt.scatter(self.locations_nerve[i+len(shifts_parietal),0], self.locations_nerve[i+len(shifts_parietal),1], c='tab:red', s=50)
                else:
                    if temporal_flip[i+len(shifts_parietal)]:
                        plt.hlines(y=shifts_temporal[i], xmin=ray.get(self.r_nerve)-length[i+len(shifts_parietal)], xmax=ray.get(self.r_nerve), linestyle='-', linewidth=3, color='tab:blue')
                    else:
                        plt.hlines(y=shifts_temporal[i], xmin=-ray.get(self.r_nerve), xmax=-ray.get(self.r_nerve)+length[i+len(shifts_parietal)], linestyle='-', linewidth=3, color='tab:blue')

        plt.xlabel('X-axis (cm)', fontsize=19)
        plt.ylabel('Y-axis (cm)', fontsize=19)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.title('Scalp Num Spikes', fontsize=22)
        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath)
        if show:
            plt.show()
        else:
            plt.close()



