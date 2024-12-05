import numpy as np
import matplotlib.pyplot as plt

class MultiPulse_MonoPhasic:
    
    def amp_train(self, amp, pw, delay, total_time, sampling_rate=1e5):
        self.sampling_rate=sampling_rate 
        total_samples = total_time * sampling_rate * 1e-3  ## total time in ms
        time_array = np.linspace(0, total_time, int(total_samples))
        amp_array = np.zeros(time_array.shape)
        pw = pw/len(amp)
        for i in range(len(amp_array)):
            if time_array[i]>=delay:
                for j in range(len(amp)):    
                    if time_array[i]>=delay+j*pw and time_array[i]<=delay+(j+1)*pw:
                        amp_array[i] = amp[j]    
        amp_array = amp_array.flatten()
    
        self.amp_array = amp_array
        self.time_array = time_array
        return amp_array, time_array
 
    def plot_waveform(self, save_path=None, units="V/m", quantity='Electric Field', show=True):
        burn_in = np.max(self.time_array)*0.1  # ms
        burn_out = np.max(self.time_array)*0.1  # ms
    
        burn_in_sample = np.linspace(0, burn_in, int(self.sampling_rate * burn_in * 1e-3))
        burn_in_amp = np.zeros(len(burn_in_sample))
    
        burn_out_sample = np.linspace(0, burn_out, int(self.sampling_rate * burn_out * 1e-3))
        burn_out_amp = np.zeros(len(burn_out_sample))
    
        amp_array = np.hstack((burn_in_amp, self.amp_array, burn_out_amp))
        time_array = np.hstack((burn_in_sample, self.time_array + burn_in, burn_out_sample + self.time_array[len(self.time_array) - 1] + burn_out))
    
        plt.plot(time_array, amp_array)
        plt.title("Temporal Profile of\n Injected "+quantity, fontsize='22')
        plt.xlabel("Time (ms)", fontsize=20)
        plt.ylabel(quantity+"("+units+ ")", fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if show is True:
            plt.show()
        else:
            plt.clf()
            plt.cla()



class SingePulse_MonoPhasic:
    
    def amp_train(self, amp, pw, delay, total_time, sampling_rate=1e5):
        self.sampling_rate=sampling_rate 
        total_samples = total_time * sampling_rate * 1e-3  ## total time in ms
        time_array = np.linspace(0, total_time, int(total_samples))
        amp_array = np.zeros(time_array.shape)
        for i in range(len(amp_array)):
            if time_array[i]>=delay and time_array[i]<=delay+pw  :
                amp_array[i] = amp
            elif time_array[i]>=delay-0.0001*pw and time_array[i]<=delay:
                amp_array[i]= amp/(0.0001*pw)*(time_array[i]-delay+0.0001*pw)
            elif time_array[i]>=delay and time_array[i]<=delay+0.0001*pw:
                amp_array[i] = amp-amp/(0.0001*pw)*(time_array[i]-delay)
     
        amp_array = amp_array.flatten()
    
        self.amp_array = amp_array
        self.time_array = time_array
        return amp_array, time_array
 
    def plot_waveform(self, save_path=None, units="V/m", quantity='Electric Field', show=True):
        burn_in = np.max(self.time_array)*0.1  # ms
        burn_out = np.max(self.time_array)*0.1  # ms
    
        burn_in_sample = np.linspace(0, burn_in, int(self.sampling_rate * burn_in * 1e-3))
        burn_in_amp = np.zeros(len(burn_in_sample))
    
        burn_out_sample = np.linspace(0, burn_out, int(self.sampling_rate * burn_out * 1e-3))
        burn_out_amp = np.zeros(len(burn_out_sample))
    
        amp_array = np.hstack((burn_in_amp, self.amp_array, burn_out_amp))
        time_array = np.hstack((burn_in_sample, self.time_array + burn_in, burn_out_sample + self.time_array[len(self.time_array) - 1] + burn_out))
    
        plt.plot(time_array, amp_array)
        plt.title("Temporal Profile of\n Injected "+quantity, fontsize='22')
        plt.xlabel("Time (ms)", fontsize=20)
        plt.ylabel(quantity+"("+units+ ")", fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if show is True:
            plt.show()
        else:
            plt.clf()
            plt.cla()


class FreqTrain_MonoPhasic:
    
    def amp_train(self, pw_vec, ipw, amp, sampling_rate=1e5):
        self.sampling_rate=sampling_rate 
        pw_vec = np.array(pw_vec).flatten()
        total_time = np.sum(pw_vec)+(len(pw_vec)-1)*ipw
    
        total_samples = total_time * sampling_rate * 1e-3  ## total time in ms
        time_array = np.linspace(0, total_time, int(total_samples))
        
        amp_array = np.zeros(time_array.shape)
        idx = 0 
        prev_pw = 0
        for i in range(len(amp_array)):
            if time_array[i]-prev_pw <= pw_vec[idx] :
                amp_array[i] = 1*amp
            elif (time_array[i]-prev_pw)>=pw_vec[idx] and (time_array[i]-prev_pw)<pw_vec[idx]+ipw:
                amp_array[i]==0
            else:
                idx = idx+1
                prev_pw = time_array[i]
    
     
        amp_array = amp_array.flatten()
    
        self.amp_array = amp_array
        self.time_array = time_array
        return amp_array, time_array
 
    def plot_waveform(self, save_path=None, units="V/m", quantity='Electric Field', show=True):
        burn_in = np.max(self.time_array)*0.1  # ms
        burn_out = np.max(self.time_array)*0.1  # ms
    
        burn_in_sample = np.linspace(0, burn_in, int(self.sampling_rate * burn_in * 1e-3))
        burn_in_amp = np.zeros(len(burn_in_sample))
    
        burn_out_sample = np.linspace(0, burn_out, int(self.sampling_rate * burn_out * 1e-3))
        burn_out_amp = np.zeros(len(burn_out_sample))
    
        amp_array = np.hstack((burn_in_amp, self.amp_array, burn_out_amp))
        time_array = np.hstack((burn_in_sample, self.time_array + burn_in, burn_out_sample + self.time_array[len(self.time_array) - 1] + burn_out))
    
        plt.plot(time_array, amp_array)
        plt.title("Temporal Profile of\n Injected "+quantity, fontsize='22')
        plt.xlabel("Time (ms)", fontsize=20)
        plt.ylabel(quantity+"("+units+ ")", fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if show is True:
            plt.show()
        else:
            plt.clf()
            plt.cla()


class FreqTrain_BiPhasic:
    
    def amp_train(self, pw_vec, ipw, amp, sampling_rate=1e5):
        self.sampling_rate=sampling_rate 
        pw_vec = np.array(pw_vec).flatten()
        total_time = 2*np.sum(pw_vec)+(len(pw_vec)-1)*ipw
    
        total_samples = total_time * sampling_rate * 1e-3  ## total time in ms
        time_array = np.linspace(0, total_time, int(total_samples))
        
        amp_array = np.zeros(time_array.shape)
        idx = 0 
        prev_pw = 0
        for i in range(len(amp_array)):
            if time_array[i]-prev_pw <= pw_vec[idx] :
                amp_array[i] = 1*amp
            elif (time_array[i]-prev_pw)<=2*pw_vec[idx] and (time_array[i]-prev_pw)>pw_vec[idx]:
                amp_array[i] = -1*amp
            elif (time_array[i]-prev_pw)>=2*pw_vec[idx] and (time_array[i]-prev_pw)<pw_vec[idx]*2+ipw:
                amp_array[i]==0
            else:
                idx = idx+1
                prev_pw = time_array[i]

        amp_array = amp_array.flatten()
    
        self.amp_array = amp_array
        self.time_array = time_array
        return amp_array, time_array
 
    def plot_waveform(self, save_path=None, units="V/m", quantity='Electric Field', show=True):
        burn_in = np.max(self.time_array)*0.1  # ms
        burn_out = np.max(self.time_array)*0.1  # ms
    
        burn_in_sample = np.linspace(0, burn_in, int(self.sampling_rate * burn_in * 1e-3))
        burn_in_amp = np.zeros(len(burn_in_sample))
    
        burn_out_sample = np.linspace(0, burn_out, int(self.sampling_rate * burn_out * 1e-3))
        burn_out_amp = np.zeros(len(burn_out_sample))
    
        amp_array = np.hstack((burn_in_amp, self.amp_array, burn_out_amp))
        time_array = np.hstack((burn_in_sample, self.time_array + burn_in, burn_out_sample + self.time_array[len(self.time_array) - 1] + burn_out))
    
        plt.plot(time_array, amp_array)
        plt.title("Temporal Profile of\n Injected "+quantity, fontsize='22')
        plt.xlabel("Time (ms)", fontsize=20)
        plt.ylabel(quantity+"("+units+ ")", fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if show is True:
            plt.show()
        else:
            plt.clf()
            plt.cla()


class PulseTrain_BiPhasic:
    
    def amp_train(self, amp_vec, freq, pulse_width, sampling_rate=1e5):
        self.sampling_rate=sampling_rate 
        amp_vec = np.array(amp_vec).flatten()
        total_time = 1000/freq*len(amp_vec)
    
        total_samples = total_time * sampling_rate * 1e-3  ## total time in ms
        time_array = np.linspace(0, total_time, int(total_samples))
        
        amp_array = np.zeros(time_array.shape)
        for i in range(len(amp_array)):
            idx = int(np.floor(time_array[i]/1000*freq))
            if idx == len(amp_vec):
                idx = len(amp_vec)-1
            if time_array[i] - idx*1000/freq<=pulse_width:
                amp_array[i] = 1*amp_vec[idx]
            elif time_array[i] - idx*1000/freq<=2*pulse_width:
                amp_array[i] = -1*amp_vec[idx]
     
        amp_array = amp_array.flatten()
    
        self.amp_array = amp_array
        self.time_array = time_array
        return amp_array, time_array
 
    def plot_waveform(self, save_path=None, units="V/m", quantity='Electric Field', show=True):
        burn_in = np.max(self.time_array)*0.1  # ms
        burn_out = np.max(self.time_array)*0.1  # ms
    
        burn_in_sample = np.linspace(0, burn_in, int(self.sampling_rate * burn_in * 1e-3))
        burn_in_amp = np.zeros(len(burn_in_sample))
    
        burn_out_sample = np.linspace(0, burn_out, int(self.sampling_rate * burn_out * 1e-3))
        burn_out_amp = np.zeros(len(burn_out_sample))
    
        amp_array = np.hstack((burn_in_amp, self.amp_array, burn_out_amp))
        time_array = np.hstack((burn_in_sample, self.time_array + burn_in, burn_out_sample + self.time_array[len(self.time_array) - 1] + burn_out))
    
        plt.plot(time_array, amp_array)
        plt.title("Temporal Profile of\n Injected "+quantity, fontsize='22')
        plt.xlabel("Time (ms)", fontsize=20)
        plt.ylabel(quantity+"("+units+ ")", fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if show is True:
            plt.show()
        else:
            plt.clf()
            plt.cla()

class PulseTrain_MonoPhasic:
    
    def amp_train(self, amp_vec, freq, pulse_width, sampling_rate=1e5):
        
        amp_vec = np.array(amp_vec).flatten()
        total_time = 1000/freq*len(amp_vec)
        self.sampling_rate = sampling_rate 
        total_samples = total_time * sampling_rate * 1e-3  ## total time in ms
        time_array = np.linspace(0, total_time, int(total_samples))
        
        amp_array = np.zeros(time_array.shape)
       
        for i in range(len(amp_array)):
            idx = int(np.floor(time_array[i]/1000*freq))
            if idx == len(amp_vec):
                idx = len(amp_vec)-1
            if time_array[i] - idx*1000/freq<=pulse_width:
                amp_array[i] = amp_vec[idx]
            
    
        amp_array = amp_array.flatten()
    
        self.amp_array = amp_array
        self.time_array = time_array
        return amp_array, time_array
    
    def plot_waveform(self, save_path=None, units="V/m", quantity='Electric Field', show=True):
        burn_in = np.max(self.time_array)*0.1  # ms
        burn_out = np.max(self.time_array)*0.1  # ms
    
        burn_in_sample = np.linspace(0, burn_in, int(self.sampling_rate * burn_in * 1e-3))
        burn_in_amp = np.zeros(len(burn_in_sample))
    
        burn_out_sample = np.linspace(0, burn_out, int(self.sampling_rate * burn_out * 1e-3))
        burn_out_amp = np.zeros(len(burn_out_sample))
    
        amp_array = np.hstack((burn_in_amp, self.amp_array, burn_out_amp))
        time_array = np.hstack((burn_in_sample, self.time_array + burn_in, burn_out_sample + self.time_array[len(self.time_array) - 1] + burn_out))
    
        plt.plot(time_array, amp_array)
        plt.title("Temporal Profile of\n Injected "+quantity, fontsize='22')
        plt.xlabel("Time (ms)", fontsize=20)
        plt.ylabel(quantity+"("+units+ ")", fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if show is True:
            plt.show()
        else:
            plt.clf()
            plt.cla()


class PulseTrain_Sinusoid:
    
    def amp_train(self, amp, freq, total_time, sampling_rate=1e6):
        
        self.sampling_rate = sampling_rate 
        total_samples = total_time * sampling_rate * 1e-3  ## total time in ms
        time_array = np.linspace(0, total_time, int(total_samples))
        
        amp_array = amp*np.sin(2*np.pi*freq*time_array*1e-3)
        
        #ramp_samples = np.sum(time_array<200)
        #ramp_inc = np.linspace(0,amp,ramp_samples)
        #ramp_dec = np.flip(ramp_inc.copy())
        #amp_array[:ramp_samples] = amp_array[:ramp_samples]*ramp_inc
        #amp_array[-ramp_samples:] = amp_array[-ramp_samples:]*ramp_dec
        amp_array = amp_array.flatten()
    
        self.amp_array = amp_array
        self.time_array = time_array
        return amp_array, time_array
    
    def plot_waveform(self, save_path=None, units="V/m", quantity='Electric Field', show=True):
        burn_in = np.max(self.time_array)*0.1  # ms
        burn_out = np.max(self.time_array)*0.1  # ms
    
        burn_in_sample = np.linspace(0, burn_in, int(self.sampling_rate * burn_in * 1e-3))
        burn_in_amp = np.zeros(len(burn_in_sample))
    
        burn_out_sample = np.linspace(0, burn_out, int(self.sampling_rate * burn_out * 1e-3))
        burn_out_amp = np.zeros(len(burn_out_sample))
    
        amp_array = np.hstack((burn_in_amp, self.amp_array, burn_out_amp))
        time_array = np.hstack((burn_in_sample, self.time_array + burn_in, burn_out_sample + self.time_array[len(self.time_array) - 1] + burn_out))
    
        plt.plot(time_array, amp_array)
        plt.title("Temporal Profile of\n Injected "+quantity, fontsize='22')
        plt.xlabel("Time (ms)", fontsize=20)
        plt.ylabel(quantity+"("+units+ ")", fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if show is True:
            plt.show()
        else:
            plt.clf()
            plt.cla()
 
class PulseTrain_TI:
    
    def amp_train(self, amp1, amp2, freq1, freq2, total_time, sampling_rate=1e6):
        
        self.sampling_rate = sampling_rate 
        total_samples = total_time * sampling_rate * 1e-3  ## total time in ms
        time_array = np.linspace(0, total_time, int(total_samples))
        
        amp_array = amp1*np.sin(2*np.pi*freq1*time_array*1e-3)+amp2*np.sin(2*np.pi*freq2*time_array*1e-3)
        amp_array = amp_array.flatten()
    
        self.amp_array = amp_array
        self.time_array = time_array
        return amp_array, time_array
    
    def plot_waveform(self, save_path=None, units="V/m", quantity='Electric Field', show=True):
        burn_in = np.max(self.time_array)*0.1  # ms
        burn_out = np.max(self.time_array)*0.1  # ms
    
        burn_in_sample = np.linspace(0, burn_in, int(self.sampling_rate * burn_in * 1e-3))
        burn_in_amp = np.zeros(len(burn_in_sample))
    
        burn_out_sample = np.linspace(0, burn_out, int(self.sampling_rate * burn_out * 1e-3))
        burn_out_amp = np.zeros(len(burn_out_sample))
    
        amp_array = np.hstack((burn_in_amp, self.amp_array, burn_out_amp))
        time_array = np.hstack((burn_in_sample, self.time_array + burn_in, burn_out_sample + self.time_array[len(self.time_array) - 1] + burn_out))
    
        plt.plot(time_array, amp_array)
        plt.title("Temporal Profile of\n Injected "+quantity, fontsize='22')
        plt.xlabel("Time (ms)", fontsize=20)
        plt.ylabel(quantity+"("+units+ ")", fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if show is True:
            plt.show()
        else:
            plt.clf()
            plt.cla()
 
