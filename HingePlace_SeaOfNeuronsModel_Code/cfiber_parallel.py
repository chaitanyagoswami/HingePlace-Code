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

@ray.remote
class CfiberSim:
    
    def __init__(self, length=16, temp=37, dt=0.025, periphery_only=True, temporal=False, temporal_flip=False, shift=8, r=9.2, saxon_q=50, caxon_q=50,  soma_q=25, elec_field=None):
        
        cwd = os.getcwd()##
        from neuron import h, gui, rxd
        from neuron.units import ms, mV, uM
        h.load_file("stdrun.hoc")

        class read_cell:
        
            def __init__(self, length=16, temp=37, dt=0.025, periphery_only=True, temporal=False, temporal_flip=False, shift=8, r=9.2, saxon_q=50, caxon_q=50,  soma_q=25):
                
                self.temporal, self.temporal_flip = temporal, temporal_flip
                if self.temporal:
                    project_len = r-r*np.sin(length/(2*r))
                else:
                    project_len = 2*r*np.sin(length/(2*r))
        
                self.saxon_q, self.caxon_q, self.soma_q = saxon_q, caxon_q, soma_q
                self.paxon_q = int(np.floor(project_len*10))
                self.periphery_only = periphery_only
                self.temporal= temporal
                self.r, self.shift = r, shift
        
                self._define_cfiber()
                h.define_shape()
                if not self.temporal:
                    self._redefine_parietal_frontal_shape(shift=shift, r=r)
                else:
                    self._redefine_temporal_shape(shift=shift, r=r)
        
                self._set_extracellular_stim()
                h.celsius = temp
                h.dt = dt
                #print("Temparature chosen for simulation:", h.celsius)
                #print("Discretization Step for simulation: %s ms"%(str(h.dt)))
            
            def _get_coord(self):
                pts = []
                for sec in h.allsec():
                    for i in range(sec.n3d()):
                        x, y, z = sec.x3d(i),sec.y3d(i),sec.z3d(i)
                        pts.append([x,y,z])
                pts = np.array(pts).reshape(-1,3)
                return pts

            def _define_cfiber(self):
                self._define_paxon()
                if not self.periphery_only:
                    self._define_caxon()
                    self._define_saxon()
                    self._define_soma()
        
            def _redefine_temporal_shape(self,shift,r):
                self._recenter_cfiber()
                self._place_cfiber(r) 
                self._shiftY(shift,r)
        
            def _redefine_parietal_frontal_shape(self, shift, r):       
                self._recenter_cfiber()
                self._place_cfiber(r) 
                self._shiftX(shift,r)
            
            def _rotateY(self,angle,x,y,z):
                xnew = x*np.cos(angle)+z*np.sin(angle)
                znew = z*np.cos(angle)-x*np.sin(angle)
                return xnew, y, znew
            
            def _rotateZ(self,angle,x,y,z):
                xnew = x*np.cos(angle)+y*np.sin(angle)
                ynew = y*np.cos(angle)-x*np.sin(angle)
                return xnew, ynew, z
            
            def _shiftX(self, shift,r):
                angle = shift/r ## cm->um
                for sec in h.allsec():
                    for i in range(sec.n3d()):
                        x, y, z, diam = sec.x3d(i),sec.y3d(i),sec.z3d(i), sec.diam3d(i)
                        xnew, ynew, znew = self._rotateY(angle,x,y,z)
                        h.pt3dchange(i,xnew, ynew, znew, diam, sec=sec)
            
            def _shiftY(self, shift,r):
                angle = shift/r ## cm->um
                for sec in h.allsec():
                    for i in range(sec.n3d()):
                        x, y, z, diam = sec.x3d(i),sec.y3d(i),sec.z3d(i), sec.diam3d(i)
                        xnew, ynew, znew = self._rotateZ(angle,x,y,z)
                        if self.temporal_flip:
                            h.pt3dchange(i,-1*xnew, ynew, znew, diam, sec=sec)
                        else:
                            h.pt3dchange(i, xnew, ynew, znew, diam, sec=sec)

        
            def _place_cfiber(self, r):
                for sec in h.allsec():
                    for i in range(sec.n3d()):
                        x, y, z, diam = sec.x3d(i),sec.y3d(i),sec.z3d(i), sec.diam3d(i)
                        x_new, y_new, z_new = self._project_cart_to_sph(r=r, x=x, y=y, z=z)
                        h.pt3dchange(i,x_new, y_new, z_new, diam, sec=sec)
            
            def _recenter_cfiber(self):
                pt_lst = [] 
                for sec in h.allsec():
                    for i in range(sec.n3d()):
                        x, y, z, diam = sec.x3d(i),sec.y3d(i),sec.z3d(i), sec.diam3d(i)
                        pt_lst.append([x,y,z])
                pt_lst = np.array(pt_lst)
                origin = np.mean(pt_lst, axis=0)
                for sec in h.allsec():
                    for i in range(sec.n3d()):
                        x, y, z, diam = sec.x3d(i),sec.y3d(i),sec.z3d(i), sec.diam3d(i)
                        if self.temporal:
                            h.pt3dchange(i,x-self.r*10**4, y-origin[1], z-origin[2], diam, sec=sec)
                        else:
                            h.pt3dchange(i,y-origin[1], x-origin[0], z-origin[2], diam, sec=sec)  
           
            def _project_cart_to_sph(self, r=9.2, x=0, y=0, z=0):
                r  = r*10**4 ## cm->um
                z = np.sqrt(r**2-x**2-y**2)
                return x, y, z
        
            def _set_extracellular_stim(self):
                cwd = "../neuron_simulator"
                #h.load_file(cwd+"/Backend_Code/fixnseg.hoc")
                #h('geom_nseg()')
                for sec in h.allsec():
                    sec.nseg=51
                    #if sec.nseg == 1:
                    #    sec.nseg = 333
                    sec.insert('extracellular')
                    sec.insert('xtra')
                 
                h.load_file(cwd + "/Backend_Code/Extracellular_Stim/interpxyz.hoc")
                h.load_file(cwd + "/Backend_Code/Extracellular_Stim/setpointers.hoc")
                
            def _define_paxon(self):
                self.paxon = [h.Section(name=f'paxon[{i}]') for i in range(self.paxon_q)]
                for i in range(0, self.paxon_q):
                    self.paxon[i].nseg = 1
                    self.paxon[i].L = 1000
                    self.paxon[i].diam = 0.8
                    self.paxon[i].Ra = 100
                    self.paxon[i].insert('nav17s')
                    self.paxon[i].insert('kdr')
                    self.paxon[i].insert('kta')
                    self.paxon[i].insert('nav18s')
                    self.paxon[i].insert('nav19h')
                    self.paxon[i].insert('leak')
                    self.paxon[i].ena = 63.4
                    self.paxon[i].ek = -68.5
                    self.paxon[i].e_leak = -55
                    self.paxon[i].g_leak = 0.0007
                    self.paxon[i].cm = 1
                for i in range(0, self.paxon_q-1):
                    self.paxon[i].connect(self.paxon[i+1], 1)
        
            def _define_saxon(self):
                self.saxon = [h.Section(name=f'saxon[{i}]') for i in range(self.saxon_q)]        
                for i in range(0, self.saxon_q):
                    self.saxon[i].nseg = 33
                    self.saxon[i].L = 10
                    self.saxon[i].diam = 1.4
                    self.saxon[i].Ra = 100
                    self.saxon[i].insert('nav17s')
                    self.saxon[i].insert('kdr')
                    self.saxon[i].insert('kta')
                    self.saxon[i].insert('nav18s')
                    self.saxon[i].insert('nav19h')
                    self.saxon[i].insert('leak')
                    self.saxon[i].ena = 63.4
                    self.saxon[i].ek = -68.5
                    self.saxon[i].e_leak = -55
                    self.saxon[i].g_leak = 0.0007
                for i in range(0, self.saxon_q-1):
                    self.saxon[i].connect(self.saxon[i+1], 1)
            
            def _define_caxon(self):
                self.caxon = [h.Section(name=f'caxon[{i}]') for i in range(self.caxon_q)]
                for i in range(0, self.caxon_q):
                    self.caxon[i].nseg = 33
                    self.caxon[i].L = 1000
                    self.caxon[i].diam = 0.8
                    self.caxon[i].Ra = 100
                    self.caxon[i].insert('nav17s')
                    self.caxon[i].insert('kdr')
                    self.caxon[i].insert('kta')
                    self.caxon[i].insert('nav18s')
                    self.caxon[i].insert('nav19h')
                    self.caxon[i].insert('leak')
                    self.caxon[i].ena = 63.4
                    self.caxon[i].ek = -68.5
                    self.caxon[i].e_leak = -55
                    self.caxon[i].g_leak = 0.0007
                for i in range(0, self.caxon_q - 1):
                    self.caxon[i].connect(self.caxon[i+1], 1)
                
            def _define_soma(self):        
                soma = h.Section(name='Soma') 
                self.soma = soma 
                self.soma.Ra = 100
                self.soma.cm = 1
                self.soma.L = 1*self.soma_q
                self.soma.diam = 25
                self.soma.nseg = 3
                self.soma.insert('leak') 
                self.soma.e_leak = -55       
                self.soma.g_leak = 0.0001
                self.soma.insert('nav17s')
                self.soma.ena = 63.4  
                self.soma.insert('kdr')
                self.soma.insert('kta')
                self.soma.insert('nav18s')
                self.soma.insert('nav19h')
                self.soma.ek = -68.5
                 
                self.saxon[-1].connect(self.paxon[0], 1)
                if not self.periphery_only:
                    self.caxon[-1].connect(self.paxon[0], 1)
                self.soma.connect(self.saxon[0], 1)
                
            def plot_neuron_better(self):
                pt_lst = [] 
                for sec in h.allsec():
                    for i in range(sec.n3d()):
                        x, y, z, diam = sec.x3d(i),sec.y3d(i),sec.z3d(i), sec.diam3d(i)
                        pt_lst.append([x,y,z])
        
                sec_xyz = np.vstack(pt_lst)
                view_angle = np.linspace(0,360,361)
                fig = plt.figure()
                ax = fig.add_subplot(111,projection='3d')
                 
                img = ax.scatter(sec_xyz[:,0]*10**(-4), sec_xyz[:,1]*10**(-4), sec_xyz[:,2]*10**(-4), linewidth=0.3, alpha=1.0, c='blue')
                ax.set_xlabel('X-axis (cm)', fontsize=16)
                ax.set_ylabel('Y-axis (cm)', fontsize=16)
                ax.set_zlabel('Z-axis (cm)', fontsize=16)
                ax.set_title('Neuron Orientation', fontsize=21)
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                zlim = ax.get_zlim()
                lim = [np.min([xlim[0],ylim[0],zlim[0]]), np.max([xlim[1],ylim[1],zlim[1]])]
                ax.set_zlim(zmin=lim[0], zmax=lim[1])
                ax.set_xlim(xmin=lim[0], xmax=lim[1])
                ax.set_ylim(ymin=lim[0], ymax=lim[1])
                ax.tick_params(axis='x',labelsize=12)
                ax.tick_params(axis='y', labelsize=12)
                ax.tick_params(axis='z',labelsize=12)
                plt.tight_layout()
                
                def update(frame):
                    ax.view_init(10,view_angle[frame])
                ani = animation.FuncAnimation(fig=fig, func=update, frames=361, interval=20)
                #ani.save(os.path.join(neuron_orient_dir,'NeuronOrientation_cellid'+str(cell_id)+'_layer5.gif'), writer='pillow')
                plt.show()
            
            def plot_neuron_default(self):
                ps= h.PlotShape(False)
                ps.plot(plotly).show()
                    
            def _set_xtra_param(self, elec_field, debug=False):
                for sec in h.allsec():
                    if h.ismembrane('xtra', sec=sec):
                        for seg in sec:
                            x, y, z =seg.xtra.x, seg.xtra.y, seg.xtra.z
                            seg.xtra.es1, seg.xtra.es2, seg.xtra.es3, seg.xtra.es4, seg.xtra.es5, seg.xtra.es6, seg.xtra.es7, seg.xtra.es8 = 0,0,0,0,0,0,0,0
                            
                            seg.xtra.es1 = elec_field.eval_voltage(x*10**(-3), y*10**(-3), z*10**(-3), idx=1) ## mV
                            seg.xtra.es2 = elec_field.eval_voltage(x*10**(-3), y*10**(-3), z*10**(-3), idx=2) ## mV
                            seg.xtra.es3 = elec_field.eval_voltage(x*10**(-3), y*10**(-3), z*10**(-3), idx=3) ## mV
                            seg.xtra.es4 = elec_field.eval_voltage(x*10**(-3), y*10**(-3), z*10**(-3), idx=4) ## mV
                            seg.xtra.es5 = elec_field.eval_voltage(x*10**(-3), y*10**(-3), z*10**(-3), idx=5) ## mV
                            seg.xtra.es6 = elec_field.eval_voltage(x*10**(-3), y*10**(-3), z*10**(-3), idx=6) ## mV
                            seg.xtra.es7 = elec_field.eval_voltage(x*10**(-3), y*10**(-3), z*10**(-3), idx=7) ## mV
                            seg.xtra.es8 = elec_field.eval_voltage(x*10**(-3), y*10**(-3), z*10**(-3), idx=8) ## mV
        
                            if debug:
                                if elec_field_lst[0] is not None:
                                    print("Xtra Param 1", seg.xtra.es1, "Unit Voltage", elec_field_lst[0].eval_voltage(x*10**(-3),y*10**(-3),z*10**(-3)), idx=1)   
                                if elec_field_lst[1] is not None:
                                    print("Xtra Param 2", seg.xtra.es2, "Unit Voltage", elec_field_lst[1].eval_voltage(x*10**(-3),y*10**(-3),z*10**(-3)), idx=2)   
                                if elec_field_lst[2] is not None:
                                    print("Xtra Param 3", seg.xtra.es3, "Unit Voltage", elec_field_lst[2].eval_voltage(x*10**(-3),y*10**(-3),z*10**(-3)), idx=3)   
                                if elec_field_lst[3] is not None:
                                    print("Xtra Param 4", seg.xtra.es4, "Unit Voltage", elec_field_lst[3].eval_voltage(x*10**(-3),y*10**(-3),z*10**(-3)), idx=4)   
                                if elec_field_lst[4] is not None:
                                    print("Xtra Param 5", seg.xtra.es5, "Unit Voltage", elec_field_lst[4].eval_voltage(x*10**(-3),y*10**(-3),z*10**(-3)), idx=5)   
                                if elec_field_lst[5] is not None:
                                    print("Xtra Param 6", seg.xtra.es6, "Unit Voltage", elec_field_lst[5].eval_voltage(x*10**(-3),y*10**(-3),z*10**(-3)), idx=6)   
                                if elec_field_lst[6] is not None:
                                    print("Xtra Param 7", seg.xtra.es7, "Unit Voltage", elec_field_lst[6].eval_voltage(x*10**(-3),y*10**(-3),z*10**(-3)), idx=7)   
                                if elec_field_lst[7] is not None:
                                    print("Xtra Param 8", seg.xtra.es8, "Unit Voltage", elec_field_lst[7].eval_voltage(x*10**(-3),y*10**(-3),z*10**(-3)), idx=8)   
        
        
        
            def stimulate(self, time_array, amp_array_lst, sampling_rate=1e5, delay_init=2, delay_final=2, plot=False, Vinit=-60):
                
                delay_final, delay_init = int(delay_final), int(delay_init) 
                ## burn_in_period
                burn_in = delay_init+2  # ms
                burn_out = delay_final  # ms
                burn_in_sample = np.linspace(0, burn_in, int(sampling_rate * burn_in * 1e-3))
                burn_in_amp = np.zeros(len(burn_in_sample))
                burn_out_sample = np.linspace(0, burn_out, int(sampling_rate * burn_out * 1e-3))
                burn_out_amp = np.zeros(len(burn_out_sample))
                if len(amp_array_lst)!=8:
                    raise Exception('Need to provide a list of 8 different Amplitude. Specify None if want to specify less than 8')
                for i in range(8): 
                    if amp_array_lst[i] is not None:
                        amp_array_lst[i] = np.hstack((burn_in_amp.copy(), amp_array_lst[i], burn_out_amp.copy()))
                    else:
                        amp_array_lst[i] = np.hstack((burn_in_amp.copy(), np.zeros(time_array.shape), burn_out_amp.copy()))
        
                time_array_tmp = np.hstack((burn_in_sample, time_array + burn_in, burn_out_sample + time_array[len(time_array) - 1] + burn_in))
                time_array_lst = [time_array_tmp.copy() for i in range(8)] 
                t_vec = [h.Vector(time_array_lst[i]) for i in range(8)]
                stim_waveform = [h.Vector(amp_array_lst[i]) for i in range(8)]
               
                ## Extracellular 
                stim_waveform[0].play(h._ref_stim1_xtra, t_vec[0], 1)
                stim_waveform[1].play(h._ref_stim2_xtra, t_vec[1], 1)
                stim_waveform[2].play(h._ref_stim3_xtra, t_vec[2], 1)
                stim_waveform[3].play(h._ref_stim4_xtra, t_vec[3], 1)
                stim_waveform[4].play(h._ref_stim5_xtra, t_vec[4], 1)
                stim_waveform[5].play(h._ref_stim6_xtra, t_vec[5], 1)
                stim_waveform[6].play(h._ref_stim7_xtra, t_vec[6], 1)
                stim_waveform[7].play(h._ref_stim8_xtra, t_vec[7], 1)
        
                ## Choose Recording Site
                pnode = [h.Vector().record(self.paxon[i](0.5)._ref_v) for i in range(self.paxon_q)]
                
                ## Record Time
                t = h.Vector().record(h._ref_t)  
                h.finitialize(Vinit * mV)
                h.continuerun((np.max(time_array_tmp)) * ms)
                
                self.pnode_recording = [np.array(pnode[i]) for i in range(self.paxon_q)]
                self.t = np.array(t)
                peaks, t_peaks = [], [] 
                for i in range(self.paxon_q):
                    peak, _ = find_peaks(self.pnode_recording[i], prominence=40, width=int(sampling_rate*0.01*10**(-3)))
                    idx = self.pnode_recording[i][peak]>-25 ## mV
                    peak = peak[idx]
                    if len(peak) != 0:
                        t_peaks.append(np.min(self.t[peak]))
                    else:
                        t_peaks.append(np.inf)
                    peaks.append(peak)
                
                init_node = self.paxon[np.argmin(t_peaks)]
                init_coord = []
                for seg in init_node:
                    init_coord.append([seg.xtra.x, seg.xtra.y, seg.xtra.z])
                init_coord = np.mean(np.array(init_coord), axis=0)
                self.peaks = peaks
                self.num_peaks = np.array([len(peaks[i]) for i in range(self.paxon_q)]).flatten()
                if plot:
                    self.plot_sim_result(delay_init=delay_init)
                return self.pnode_recording, np.min([1,np.max(self.num_peaks)]), self.t, init_coord
                
            def plot_sim_result(self, save_path=None, show=True, delay_init=1000):
                skip = 1
                clevel = np.linspace(0,1,self.paxon_q//skip)
        
                for i in range(self.paxon_q//skip):
                    print(np.max(self.t), delay_init)
                    plt.plot(self.t[self.t>delay_init]-delay_init, self.pnode_recording[i*skip][self.t>delay_init], c=np.array(cm.viridis(clevel[i])).reshape(1,-1))
                    plt.plot(self.t[self.peaks[i*skip]]-delay_init, self.pnode_recording[i*skip][self.peaks[i*skip]], 'x', c=np.array(cm.viridis(clevel[i])).reshape(1,-1))
                plt.title("C-Fiber Membrane Potential", fontsize='22')
                plt.xlabel("Time (ms)", fontsize=20)
                plt.ylabel("Membrane Potential (mV)", fontsize=20)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.tight_layout()
                if save_path is not None:
                    plt.savefig(save_path)
                if show:
                    plt.show()
                else:
                    plt.clf()
                    plt.cla()
        self.saxon_q, self.caxon_q, self.soma_q = saxon_q, caxon_q, soma_q
        self.length = length ## cm
        self.temp = temp ## celsius
        self.dt = dt ## ms
        self.periphery_only = periphery_only
        self.temporal = temporal
        self.temporal_flip = temporal_flip
        self.shift = shift ## cm
        self.r = r ## cm
        self.elec_field = elec_field
        self.cell = read_cell(length=self.length, temp=self.temp, dt=self.dt, periphery_only=self.periphery_only, temporal=self.temporal, temporal_flip=temporal_flip, shift=self.shift, r=self.r, saxon_q=self.saxon_q, caxon_q=self.caxon_q, soma_q=self.soma_q)
    
    def stimulate(self, time_array, amp_array_lst, scale_lst=[], sampling_rate=1e5, delay_init=2, delay_final=2, plot=False, Vinit=-60):
        for i in range(8):
            if amp_array_lst[i] is not None:
                amp_array_lst[i] =  scale_lst[i]*amp_array_lst[i]
        return self.cell.stimulate(time_array=time_array, amp_array_lst=amp_array_lst, sampling_rate=sampling_rate, delay_init=delay_init, delay_final=delay_final, plot=plot, Vinit=Vinit)

    def _set_xtra_param(self, debug=False):
        self.cell._set_xtra_param(elec_field=self.elec_field, debug=debug) 
     
    def _reset_elec_field(self, elec_field):
        self.elec_field = elec_field
    
    def _get_Jlst(self):
        return self.elec_field.J_lst 

    def _get_coord(self):
        return self.cell._get_coord()
                
    def plot_sim_result(self, save_path=None, show=True, delay_init=1000):
        self.cell.plot_sim_result(save_path=save_path, show=show, delay_init=delay_init)

 

