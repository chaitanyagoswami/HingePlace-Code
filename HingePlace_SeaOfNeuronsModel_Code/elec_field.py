import numpy as np
from datetime import date

class UniformField:
    def __init__(self, unit_vec=np.array([0,0,1])):
        self.unit_vec = unit_vec
    def eval_voltage(self, x, y, z, idx=1):
        voltage = (self.unit_vec[0]*x+self.unit_vec[1]*y+self.unit_vec[2]*z)##  a field of 1 mV/mm
        return voltage ## mV

class ICMS:
    def __init__(self, x,y,z,conductivity):
        self.x, self.y, self.z, self.cond = x,y,z,conductivity
    def eval_voltage(self, x, y, z):
        r = np.sqrt((x-self.x)**2+(y-self.y)**2+(z-self.z)**2)*1e-03 ## converting mm to m
        voltage = (1e-06/(4*np.pi*self.cond*r)) ## converting microamp to amp, and then converting volts to millivolts         
        return voltage*1000 ## mV
    def eval_efield_mag(self, x,y,z):
        r = np.sqrt((x-self.x)**2+(y-self.y)**2+(z-self.z)**2)*1e-03 ## converting mm to m
        efield = -1*(1e-06/(4*np.pi*self.cond*r**2)) ## converting microamp to amp, and then converting volts to millivolts         
        return np.abs(efield)


