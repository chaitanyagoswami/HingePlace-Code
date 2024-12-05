import numpy as np
import time
from ElectricGrid import PreDefElecGrid
from TransferFunction import TransferFunction
from CancelPoints import CancelPointsSpherical
import pyshtools as shp_harm
import cvxpy as cvx
import matplotlib.pyplot as plt
import seaborn as sns
import ray
from matplotlib import colormaps
from cvxpy import INFEASIBLE

class SphericalHeadModel(PreDefElecGrid, TransferFunction, CancelPointsSpherical):
    
    def __init__(self, r_lst, cond_vec, radius_vec, patch_size, elec_radius, elec_spacing, max_l=300, spacing=0.1, custom_grid=False, theta_elec=None, phi_elec=None, save_title=None):
        
        self.r_lst = r_lst
        self.eps = 0.0001*10**(-2)
        self.cond_vec = cond_vec
        self.radius_vec = radius_vec*10**(-2)  ## cm to m
        self.max_l = max_l
        self.spacing =spacing*10**(-2)
        self.r_max = np.max(self.radius_vec)

        self.N_lat = 2 * self.max_l + 2
        self.N_long = self.N_lat
            
        self.theta_min = -0.2
        self.long_phi = np.arange(0, 2*np.pi+self.eps, 2*np.pi/self.N_long)
        self.lat_theta = np.arange(-np.pi/2.0, np.pi/2.0+self.eps, np.pi/self.N_lat)
        self.phi_lst = self.long_phi[:-1]
        self.theta_lst = self.lat_theta[self.lat_theta>=self.theta_min]

        lat_theta_fl, long_phi_fl = np.meshgrid(self.lat_theta, self.long_phi)
        lat_theta_fl, long_phi_fl = lat_theta_fl.flatten(), long_phi_fl.flatten()
        self.electrode_spherical_map_flatten = np.transpose(np.vstack((lat_theta_fl, long_phi_fl)))
        
        self.electrode_spherical_map = np.zeros((len(self.lat_theta), len(self.long_phi)))
        
        lat_theta_fl, long_phi_fl = np.meshgrid(self.theta_lst, self.phi_lst)
        lat_theta_fl, long_phi_fl = lat_theta_fl.flatten(), long_phi_fl.flatten()
        self.spherical_map_flatten = np.vstack([lat_theta_fl, long_phi_fl]).T
        self.custom_grid = custom_grid

        if self.custom_grid is False:
            self.patch_size = patch_size * 10**(-2)
            self.elec_spacing = elec_spacing * 10**(-2)
        else:
            self.theta_elec = theta_elec
            self.phi_elec = phi_elec
        self.elec_radius = elec_radius * 10**(-2)
        self.curr_density_calculated = False
        self.forward_model_calc_flag = False
        self.J=None
        self.voltage_J=None
        self.save_title = save_title
        

    def _sph_to_cart(self,pos):
        if len(pos.shape) == 1:
            pos = pos.reshape(1,-1)
        x = pos[:,0]*np.cos(pos[:,1])*np.cos(pos[:,2])
        y = pos[:,0]*np.cos(pos[:,1])*np.sin(pos[:,2])
        z = pos[:,0]*np.sin(pos[:,1])
        cart_pos = np.hstack([x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)])
        return cart_pos
    
    def _cart_to_sph(self, pos):
        if len(pos.shape) == 1:
            pos = pos.reshape(1,-1)
        r = np.sqrt(np.sum(pos**2, axis=1)).reshape(-1,1)
        theta = np.arcsin(pos[:,2]/r.flatten()).reshape(-1,1)
        phi = np.arctan2(pos[:,1],pos[:,0]).reshape(-1,1)
        sph_pos = np.hstack([r,theta,phi]) 
        sph_pos[sph_pos[:,2]<0,2] = sph_pos[sph_pos[:,2]<0,2]+2*np.pi
        return sph_pos
    
    def _return_lat_and_long(self):
        lat_theta_fl, long_phi_fl = np.meshgrid(self.lat_theta, self.long_phi)
        lat_theta_fl, long_phi_fl = lat_theta_fl.flatten(), long_phi_fl.flatten()
        return [lat_theta_fl, long_phi_fl]
    
    def _load_forward_model(self,save_title=None): 
        
        self.points = np.load(save_title+"_locations.npy")
        self.forward_model_calc_flag = True
    
    def _calc_forward_model(self, print_elec_pattern=True, save=True, save_title=None):
        
        ## Generate a pseudo-uniform sampling of electrode points
        if self.custom_grid is False:
            elec_lst, ground_elec, elec_radius = self.uniform_sampling_north_pole(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**(2), patch_size=self.patch_size*10**(2))
        else:
            elec_lst, ground_elec, elec_radius = np.hstack((self.theta_elec.reshape(-1,1), self.phi_elec.reshape(-1,1))), np.array((-np.pi/2.0,0),), self.elec_radius*np.ones((len(self.theta_elec)))*10**2

        if print_elec_pattern:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=np.ones(len(elec_lst)))
            grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
            grid_electrode.plot()
        
        ## Calculating the Transfer Function
        self.A_voltage = []
        self.A_Jr = []
        self.A_Jtheta = []        
        self.A_Jphi =[] 
        previous_len = 0
        self.points = []
        r = self.r_lst
        for j in range(len(r)):
            print("Radii Remaining: %d"%(len(r)-j-1))
            tau_V, tau_Jr, _ = self.calc_tauL(r[j])
            tau_V = np.hstack((np.array((0,)), tau_V))
            tau_Jr = np.hstack((np.array((0,)), tau_Jr))
            layer_idx = int(np.where(np.abs(self.radius_vec-np.min(self.radius_vec[self.radius_vec-r[j]*10**(-2)>0]))<self.eps)[0])
            cond = self.cond_vec[layer_idx]
            all_points_subsampled = np.concatenate([r[j]*np.ones([self.spherical_map_flatten.shape[0],1]),self.spherical_map_flatten.copy()], axis=1)            
            ## Calculate the Forward Model
            A_voltage, A_Jr, A_Jtheta, A_Jphi = np.empty([len(all_points_subsampled),len(elec_lst)]), np.empty([len(all_points_subsampled),len(elec_lst)]), np.empty([len(all_points_subsampled),len(elec_lst)]), np.empty([len(all_points_subsampled),len(elec_lst)])
            for i in range(len(elec_lst)):

                start_time = time.time()
                elec_pos = np.vstack((elec_lst[i], ground_elec))
                elec_radii = self.elec_radius*np.ones(len(elec_pos))*10**(2)
                inj_curr = np.array((1/(np.pi*(self.elec_radius*10**2)**2), -1/(np.pi*(self.elec_radius*10**2)**2))) ## mA/cm^2

                self.electrode_sampling(elec_pos, elec_radii, inj_curr, None)
                elec_grid_array = np.flip(self.electrode_spherical_map, axis=0)
                ## Taking the forward spherical harmonic transform
                elec_grid = shp_harm.SHGrid.from_array(elec_grid_array)
                coeff = elec_grid.expand()
                coeff_array = coeff.coeffs

                ## Calculating Radial Current Density
                Jr_coeff = np.zeros(coeff_array.shape)
                Jr_coeff[0,:,:] = np.transpose(np.transpose(coeff_array[0])*tau_Jr)
                Jr_coeff[1,:,:] = np.transpose(np.transpose(coeff_array[1])*tau_Jr)
                Jr_coeff = shp_harm.SHCoeffs.from_array(Jr_coeff)
                Jr = Jr_coeff.expand(grid='DH')
                Jr_data = Jr.data
                Jr_data = Jr_data[:,:-1]
                Jr_data = np.flip(Jr_data, axis=0)
                Jr_data = Jr_data[self.lat_theta>self.theta_min]
                
                ## Calculating Voltage
                voltage_coeff = np.zeros(coeff_array.shape)
                voltage_coeff[0,:,:] = np.transpose(np.transpose(coeff_array[0])*tau_V)
                voltage_coeff[1,:,:] = np.transpose(np.transpose(coeff_array[1])*tau_V)
                voltage_coeff = shp_harm.SHCoeffs.from_array(voltage_coeff)
                voltage = voltage_coeff.expand(grid='DH')
                voltage_data = voltage.data*10 ## V
                voltage_data = voltage_data[:,:-1]
                voltage_data = np.flip(voltage_data, axis=0)
                voltage_data =voltage_data[self.lat_theta>self.theta_min]
                
                ## Calculating tangential theta `current density
                Jtheta_data = np.empty(voltage_data.shape)
                del_theta = (self.theta_lst[1]-self.theta_lst[0]) ## -1 sign to correct for the opposite order of self.lat_theta and how data is stored in voltage_data
                for k in range(voltage_data.shape[0]):
                    if k==0:
                        Jtheta_data[k,:] = -1*cond*(voltage_data[k+1,:]-voltage_data[k,:])/(r[j]*del_theta*10**(-2)*10) ## cm->m, A/m^2->mA/cm^2
                    elif k==voltage_data.shape[0]-1:
                        Jtheta_data[k,:] = -1*cond*(voltage_data[k,:]-voltage_data[k-1,:])/(r[j]*del_theta*10**(-2)*10)
                    else:
                        Jtheta_data[k,:] = -1*cond*(voltage_data[k+1,:]-voltage_data[k-1,:])/(2*r[j]*del_theta*10**(-2)*10) 
                
                ## Calculating tangential phi current density
                Jphi_data = np.empty(voltage_data.shape)
                del_phi = (self.phi_lst[1]-self.phi_lst[0])
                cos_array = np.cos(np.flip(self.theta_lst))
                denom = del_phi*r[j]*cos_array*10**(-1) ## mA/cm^2
                idx_theta_zero = np.abs(denom)<self.eps
                Jphi_data[idx_theta_zero,:] = 0
                idx_theta = np.logical_not(idx_theta_zero)
                for k in range(voltage_data.shape[1]):
                    if k==0:
                        Jphi_data[idx_theta,k] = -1*cond*(voltage_data[idx_theta,k+1]-voltage_data[idx_theta,k])/denom[idx_theta]
                    elif k==voltage_data.shape[1]-1:
                        Jphi_data[idx_theta,k] = -1*cond*(voltage_data[idx_theta,k]-voltage_data[idx_theta,k-1])/denom[idx_theta]
                    else:
                        Jphi_data[idx_theta,k] = -1*cond*(voltage_data[idx_theta,k+1]-voltage_data[idx_theta,k-1])/(2*denom[idx_theta])
                
                voltage_flatten = voltage_data.T.flatten()
                Jr_flatten = Jr_data.T.flatten()
                Jtheta_flatten = Jtheta_data.T.flatten()
                Jphi_flatten = Jphi_data.T.flatten()
                A_voltage[:,i] = voltage_flatten
                A_Jr[:,i] = Jr_flatten
                A_Jtheta[:,i] = Jtheta_flatten
                A_Jphi[:,i] = Jphi_flatten
                
                print("Electrode Remaining:", len(elec_lst)-i-1, "Time Taken: %.2f s"%(time.time()-start_time))
            
            self.A_voltage.append(A_voltage.copy())
            self.A_Jr.append(A_Jr.copy())
            self.A_Jtheta.append(A_Jtheta.copy())
            self.A_Jphi.append(A_Jphi.copy()) 
            self.points.append(all_points_subsampled.copy())
        
        self.points = np.concatenate(self.points, axis=0)
        self.A_voltage = np.concatenate(self.A_voltage, axis=0)
        self.A_Jr = np.concatenate(self.A_Jr, axis=0)
        self.A_Jtheta = np.concatenate(self.A_Jtheta, axis=0)
        self.A_Jphi = np.concatenate(self.A_Jphi, axis=0)

        cos_array_theta, sin_array_theta = np.cos(self.points[:,1]), np.sin(self.points[:,1])
        cos_array_phi, sin_array_phi = np.cos(self.points[:,2]), np.sin(self.points[:,2])

        self.A_Jz = self.A_Jr*sin_array_theta.reshape(-1,1)+self.A_Jtheta*cos_array_theta.reshape(-1,1)
        self.A_Jx = (self.A_Jr*cos_array_theta.reshape(-1,1))*cos_array_phi.reshape(-1,1)-(self.A_Jtheta*sin_array_theta.reshape(-1,1))*cos_array_phi.reshape(-1,1)-self.A_Jphi*sin_array_phi.reshape(-1,1)
        self.A_Jy = (self.A_Jr*cos_array_theta.reshape(-1,1))*sin_array_phi.reshape(-1,1)-(self.A_Jtheta*sin_array_theta.reshape(-1,1))*sin_array_phi.reshape(-1,1)+self.A_Jphi*cos_array_phi.reshape(-1,1)
        if save:
            np.save(save_title+"_voltage.npy", self.A_voltage)
            np.save(save_title+"_Jr.npy", self.A_Jr)
            np.save(save_title+"_Jtheta.npy", self.A_Jtheta)
            np.save(save_title+"_Jphi.npy", self.A_Jphi)
            np.save(save_title+"_Jz.npy", self.A_Jz)
            np.save(save_title+"_Jx.npy", self.A_Jx)
            np.save(save_title+"_Jy.npy", self.A_Jy)
            np.save(save_title+"_locations.npy", self.points)
        self.forward_model_calc_flag = True
        return [self.A_Jx, self.A_Jy, self.A_Jz, self.A_voltage], self.points
    
    
    def _get_Af_and_Ac(self, focus_points, cancel_points):    
        self.A_voltage = np.load(self.save_title+"_voltage.npy")
        self.A_Jz = np.load(self.save_title+"_Jz.npy")
        self.A_Jx = np.load(self.save_title+"_Jx.npy")
        self.A_Jy = np.load(self.save_title+"_Jy.npy")
        
        if not self.forward_model_calc_flag:
            raise Exception('Either load a valid forward model or calculate one using _calc_forward_model()!!!!!')
        
        Af_v, Ac_v = np.zeros([len(focus_points),self.A_voltage.shape[1]]), np.zeros([len(cancel_points),self.A_voltage.shape[1]])
        Af_z, Ac_z = np.zeros([len(focus_points),self.A_voltage.shape[1]]), np.zeros([len(cancel_points),self.A_voltage.shape[1]])
        Af_x, Ac_x = np.zeros([len(focus_points),self.A_voltage.shape[1]]), np.zeros([len(cancel_points),self.A_voltage.shape[1]])
        Af_y, Ac_y = np.zeros([len(focus_points),self.A_voltage.shape[1]]), np.zeros([len(cancel_points),self.A_voltage.shape[1]])
        for i in range(len(focus_points)):
            r_idx = np.argmin(np.abs(focus_points[i,0]-self.r_lst))
            np.searchsorted(self.r_lst,focus_points[i,0]) 
            theta_idx = np.searchsorted(self.theta_lst, focus_points[i,1])
            phi_idx = np.searchsorted(self.phi_lst, focus_points[i,2])
            
            
            if theta_idx == 0:
                theta_idx_prev = 0
            elif theta_idx == len(self.lat_theta):
                theta_idx_prev=theta_idx-1
                theta_idx = theta_idx-1
            else:
                theta_idx_prev = theta_idx-1
            
            if phi_idx == 0:
                phi_idx_prev = 0
            elif theta_idx == len(self.lat_theta):
                phi_idx_prev = phi_idx-1
                phi_idx = phi_idx-1
            else:
                phi_idx_prev = phi_idx-1
            
            idx1 = r_idx*len(self.phi_lst)*len(self.theta_lst)+phi_idx*len(self.theta_lst)+theta_idx
            idx2 = r_idx*len(self.phi_lst)*len(self.theta_lst)+phi_idx_prev*len(self.theta_lst)+theta_idx_prev
            
            dist_forward = (self.points[idx1,0]*np.sin(self.points[idx1,1])-focus_points[i,0]*np.sin(focus_points[i,1]))**2
            dist_forward = dist_forward+(self.points[idx1,0]*np.cos(self.points[idx1,1])*np.cos(self.points[idx1,2])-focus_points[i,0]*np.cos(focus_points[i,1])*np.cos(focus_points[i,2]))**2            
            dist_forward = dist_forward+(self.points[idx1,0]*np.cos(self.points[idx1,1])*np.sin(self.points[idx1,2])-focus_points[i,0]*np.cos(focus_points[i,1])*np.sin(focus_points[i,2]))**2            
            dist_forward = np.sqrt(dist_forward)
            
            dist_backward = (self.points[idx2,0]*np.sin(self.points[idx2,1])-focus_points[i,0]*np.sin(focus_points[i,1]))**2
            dist_backward = dist_backward+(self.points[idx2,0]*np.cos(self.points[idx2,1])*np.cos(self.points[idx2,2])-focus_points[i,0]*np.cos(focus_points[i,1])*np.cos(focus_points[i,2]))**2            
            dist_backward = dist_backward+(self.points[idx2,0]*np.cos(self.points[idx2,1])*np.sin(self.points[idx2,2])-focus_points[i,0]*np.cos(focus_points[i,1])*np.sin(focus_points[i,2]))**2            
            dist_backward = np.sqrt(dist_backward)

            if dist_forward==0 and dist_backward==0:
                Af_v[i] = (self.A_voltage[idx1]+self.A_voltage[idx2])/2
                Af_x[i] = (self.A_Jx[idx1]+self.A_Jx[idx2])/2
                Af_y[i] = (self.A_Jy[idx1]+self.A_Jy[idx2])/2                
                Af_z[i] = (self.A_Jz[idx1]+self.A_Jz[idx2])/2
            else:
                Af_v[i] = dist_backward/(dist_forward+dist_backward)*self.A_voltage[idx1]+dist_forward/(dist_forward+dist_backward)*self.A_voltage[idx2]
                Af_x[i] = dist_backward/(dist_forward+dist_backward)*self.A_Jx[idx1]+dist_forward/(dist_forward+dist_backward)*self.A_Jx[idx2]
                Af_y[i] = dist_backward/(dist_forward+dist_backward)*self.A_Jy[idx1]+dist_forward/(dist_forward+dist_backward)*self.A_Jy[idx2]
                Af_z[i] = dist_backward/(dist_forward+dist_backward)*self.A_Jz[idx1]+dist_forward/(dist_forward+dist_backward)*self.A_Jz[idx2] 
                            
        for i in range(len(cancel_points)):
            r_idx = np.argmin(np.abs(self.r_lst-cancel_points[i,0])) 
            theta_idx = np.searchsorted(self.theta_lst, cancel_points[i,1])
            phi_idx = np.searchsorted(self.phi_lst, cancel_points[i,2])
            
            
            if theta_idx == 0:
                theta_idx_prev = 0
            elif theta_idx == len(self.theta_lst):
                theta_idx_prev=theta_idx-1
                theta_idx = theta_idx-1
            else:
                theta_idx_prev = theta_idx-1
            
            if phi_idx == 0:
                phi_idx_prev = 0
            elif phi_idx == len(self.phi_lst):
                phi_idx_prev = phi_idx-1
                phi_idx = phi_idx-1
            else:
                phi_idx_prev = phi_idx-1

            idx1 = r_idx*len(self.phi_lst)*len(self.theta_lst)+phi_idx*len(self.theta_lst)+theta_idx
            idx2 = r_idx*len(self.phi_lst)*len(self.theta_lst)+phi_idx_prev*len(self.theta_lst)+theta_idx_prev
            dist_forward = (self.points[idx1,0]*np.sin(self.points[idx1,1])-cancel_points[i,0]*np.sin(cancel_points[i,1]))**2
            dist_forward = dist_forward+(self.points[idx1,0]*np.cos(self.points[idx1,1])*np.cos(self.points[idx1,2])-cancel_points[i,0]*np.cos(cancel_points[i,1])*np.cos(cancel_points[i,2]))**2            
            dist_forward = dist_forward+(self.points[idx1,0]*np.cos(self.points[idx1,1])*np.sin(self.points[idx1,2])-cancel_points[i,0]*np.cos(cancel_points[i,1])*np.sin(cancel_points[i,2]))**2            
            dist_forward = np.sqrt(dist_forward)
            
            dist_backward = (self.points[idx2,0]*np.sin(self.points[idx2,1])-cancel_points[i,0]*np.sin(cancel_points[i,1]))**2
            dist_backward = dist_backward+(self.points[idx2,0]*np.cos(self.points[idx2,1])*np.cos(self.points[idx2,2])-cancel_points[i,0]*np.cos(cancel_points[i,1])*np.cos(cancel_points[i,2]))**2            
            dist_backward = dist_backward+(self.points[idx2,0]*np.cos(self.points[idx2,1])*np.sin(self.points[idx2,2])-cancel_points[i,0]*np.cos(cancel_points[i,1])*np.sin(cancel_points[i,2]))**2            
            dist_backward = np.sqrt(dist_backward)

            if dist_forward==0 and dist_backward==0:
                Ac_v[i] = (self.A_voltage[idx1]+self.A_voltage[idx2])/2
                Ac_x[i] = (self.A_Jx[idx1]+self.A_Jx[idx2])/2
                Ac_y[i] = (self.A_Jy[idx1]+self.A_Jy[idx2])/2                
                Ac_z[i] = (self.A_Jz[idx1]+self.A_Jz[idx2])/2
            else:
                Ac_v[i] = dist_backward/(dist_forward+dist_backward)*self.A_voltage[idx1]+dist_forward/(dist_forward+dist_backward)*self.A_voltage[idx2]
                Ac_x[i] = dist_backward/(dist_forward+dist_backward)*self.A_Jx[idx1]+dist_forward/(dist_forward+dist_backward)*self.A_Jx[idx2]
                Ac_y[i] = dist_backward/(dist_forward+dist_backward)*self.A_Jy[idx1]+dist_forward/(dist_forward+dist_backward)*self.A_Jy[idx2]
                Ac_z[i] = dist_backward/(dist_forward+dist_backward)*self.A_Jz[idx1]+dist_forward/(dist_forward+dist_backward)*self.A_Jz[idx2] 
        
        del self.A_voltage, self.A_Jx, self.A_Jy, self.A_Jz
        return Af_v, Ac_v, Af_z, Ac_z, Af_x, Ac_x, Af_y, Ac_y 

    def calc_all_density(self, r, J=None):

        self.A_voltage = np.load(self.save_title+"_voltage.npy")
        self.A_Jr = np.load(self.save_title+"_Jr.npy")
        self.A_Jtheta = np.load(self.save_title+"_Jtheta.npy")
        self.A_Jphi = np.load(self.save_title+"_Jphi.npy")
        
        self.A_Jz = np.load(self.save_title+"_Jz.npy")
        self.A_Jx = np.load(self.save_title+"_Jx.npy")
        self.A_Jy = np.load(self.save_title+"_Jy.npy")
        
        ## Electrode Density
        A_voltage = self.A_voltage[self.points[:,1]>=0]
        A_Jr, A_Jtheta, A_Jphi = self.A_Jr[self.points[:,1]>=0], self.A_Jtheta[self.points[:,1]>=0], self.A_Jphi[self.points[:,1]>=0]
        A_Jx, A_Jy, A_Jz = self.A_Jx[self.points[:,1]>=0], self.A_Jy[self.points[:,1]>=0], self.A_Jz[self.points[:,1]>=0]
        points = self.points[self.points[:,1]>=0]
        r = self.r_lst[np.argmin(np.abs(r-self.r_lst))] 
        curr_density_X = np.matmul(A_Jx[np.abs(points[:,0]-r)<self.eps], J.reshape(-1,1)).flatten()
        curr_density_Y = np.matmul(A_Jy[np.abs(points[:,0]-r)<self.eps], J.reshape(-1,1)).flatten()
        curr_density_Z = np.matmul(A_Jz[np.abs(points[:,0]-r)<self.eps], J.reshape(-1,1)).flatten()
        
        curr_density_r = np.matmul(A_Jr[np.abs(points[:,0]-r)<self.eps], J.reshape(-1,1)).flatten()
        curr_density_theta = np.matmul(A_Jtheta[np.abs(points[:,0]-r)<self.eps], J.reshape(-1,1)).flatten()
        curr_density_phi = np.matmul(A_Jphi[np.abs(points[:,0]-r)<self.eps], J.reshape(-1,1)).flatten()

        curr_density_norm = np.sqrt(curr_density_X**2+curr_density_Y**2+curr_density_Z**2)
        points = points[np.abs(points[:,0]-r)<self.eps]
        xy_grid = [points[:,0]*np.cos(points[:,1])*np.cos(points[:,2]), points[:,0]*np.cos(points[:,1])*np.sin(points[:,2])] 
        xy_grid = np.concatenate([xy_grid[0].reshape(-1,1), xy_grid[1].reshape(-1,1)], axis=1)
        
        del self.A_voltage 
        del self.A_Jr
        del self.A_Jtheta 
        del self.A_Jphi
        del self.A_Jz 
        del self.A_Jx 
        del self.A_Jy 
        return curr_density_norm, curr_density_X, curr_density_Y, curr_density_Z, xy_grid
    
    def _get_max_locations(self, r, J):
        curr_density_norm, curr_density_X, curr_density_Y, curr_density_Z, xy_grid= self.calc_all_density(r=r, J=J)
        pos_norm = xy_grid[np.argmax(curr_density_norm)]
        pos_norm = self._cart_to_sph(np.array([pos_norm[0], pos_norm[1], np.sqrt(r**2-pos_norm[0]**2-pos_norm[1]**2)]).reshape(1,3))
        pos_X = xy_grid[np.argmax(curr_density_X)]
        pos_X = self._cart_to_sph(np.array([pos_X[0], pos_X[1], np.sqrt(r**2-pos_X[0]**2-pos_X[1]**2)]).reshape(1,3))    
        pos_Y = xy_grid[np.argmax(curr_density_Y)]
        pos_Y = self._cart_to_sph(np.array([pos_Y[0], pos_Y[1], np.sqrt(r**2-pos_Y[0]**2-pos_Y[1]**2)]).reshape(1,3))
        pos_Z = xy_grid[np.argmax(-1*curr_density_Z)]
        pos_Z = self._cart_to_sph(np.array([pos_Z[0], pos_Z[1], np.sqrt(r**2-pos_Z[0]**2-pos_Z[1]**2)]).reshape(1,3))
        return pos_norm.flatten(), pos_X.flatten(), pos_Y.flatten(), pos_Z.flatten()

    def plot_given_all_curr_density(self, xy_grid, curr_density, x_limit=None, y_limit=None, abs_flag=True, fname=None, show=True):
        title_lst = ['Current Density-Norm', 'Current Density-X', 'Current Density-Y', 'Current Density-Z']
        suffix_lst = ['_norm.png', '_x.png', '_y.png', '_z.png']
        for i in range(len(curr_density)):
            color_map = colormaps['jet']
            img = plt.scatter(xy_grid[:,0], xy_grid[:,1], c=curr_density[i], vmin=np.min(curr_density[i]), vmax=np.max(curr_density[i]), cmap=color_map, linewidth=0.0001, s=1)
            cbar = plt.colorbar(img)
            cbar.set_label(r'mA/cm$^2$', fontsize=19)
            cbar.ax.tick_params(labelsize=16)
            plt.xlabel('X-axis (cm)', fontsize=19)
            plt.ylabel('Y-axis (cm)', fontsize=19)
            plt.xticks(fontsize=18, rotation=45)
            plt.yticks(fontsize=18)
            plt.title(title_lst[i], fontsize=22)
            plt.tight_layout()
            if fname is not None:
                plt.savefig(fname+suffix_lst[i])
            if show:
                plt.show()
            else:
                plt.close()

    def plot_all_curr_density(self, r, J=None, fname=None, x_limit=None, y_limit=None, abs_flag=False, show=True):
        if J is None:
            raise Exception('Supply a valid current configuration')
        else:
            curr_density_norm, curr_density_X, curr_density_Y, curr_density_Z, xy_grid = self.calc_all_density(r,J)

        curr_density_norm_max = round(float(np.mean(np.sort(curr_density_norm.flatten())[-10:])),3)
        curr_density_X_max = round(float(np.mean(np.sort(curr_density_X.flatten())[-10:])),3)
        curr_density_Y_max = round(float(np.mean(np.sort(curr_density_Y.flatten())[-10:])),3)
        curr_density_Z_max = round(float(np.mean(np.sort(curr_density_Z.flatten())[-10:])),3)
        #print(">>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        #print("Current Density Norm at height %s cm: %s mA/cm^2"%(str(round(self.r_max*10**2-r,3)),str(curr_density_norm_max)))
        #print("Current Density X at height %s cm: %s mA/cm^2"%(str(round(self.r_max*10**2-r,3)),str(curr_density_X_max)))
        #print("Current Density Y at height %s cm: %s mA/cm^2"%(str(round(self.r_max*10**2-r,3)),str(curr_density_Y_max)))
        #print("Current Density Z at height %s cm: %s mA/cm^2"%(str(round(self.r_max*10**2-r,3)),str(curr_density_Z_max)))
        #print(">>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        return self.plot_given_all_curr_density(xy_grid=xy_grid, curr_density=[curr_density_norm, curr_density_X, curr_density_Y, curr_density_Z], fname=fname, x_limit=x_limit, y_limit=y_limit, abs_flag=abs_flag, show=show)
     
    def _return_num_electrodes(self,):
        
        if self.custom_grid is False:
            elec_lst, ground_elec, elec_radius = self.uniform_sampling_north_pole(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**(2), patch_size=self.patch_size*10**(2))
        else:
            elec_lst, ground_elec, elec_radius = np.hstack((self.theta_elec.reshape(-1,1), self.phi_elec.reshape(-1,1))),  np.array((-np.pi/2.0,0),), self.elec_radius*np.ones((len(self.theta_elec)))*10**2
        return len(elec_lst)

    def plot_elec_pttrn(self, J, x_lim=None, y_lim=None, fname=None, show=True):
        if self.custom_grid is False:
            elec_lst, ground_elec, elec_radius = self.uniform_sampling_north_pole(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**(2), patch_size=self.patch_size*10**(2))
        else:
            elec_lst, ground_elec, elec_radius = np.hstack((self.theta_elec.reshape(-1,1), self.phi_elec.reshape(-1,1))),  np.array((-np.pi/2.0,0),), self.elec_radius*np.ones((len(self.theta_elec)))*10**2

        self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=J)
        grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')

        data = self.electrode_spherical_map
        elec_loc = np.concatenate([self.r_max*10**2*np.ones([len(elec_lst),1]),elec_lst], axis=1)
        elec_loc = self._sph_to_cart(elec_loc)
        
        long_fl, lat_fl = np.meshgrid(self.long_phi, self.lat_theta)
        long_fl, lat_fl = long_fl.flatten(), lat_fl.flatten()
        spherical_map_flatten = np.transpose(np.vstack((long_fl, lat_fl)))

        ### Plotting the first half
        idx = spherical_map_flatten[:, 1] >= 0
        x, y = self.r_max * 10**2*np.cos(spherical_map_flatten[idx, 1]) * np.cos(spherical_map_flatten[idx, 0]), self.r_max * 10**2*np.cos(spherical_map_flatten[idx, 1]) * np.sin(spherical_map_flatten[idx, 0])

        data_flatten = data.flatten()
        data_flatten = data_flatten[idx]
        if x_lim is None:
            x_lim = np.array((np.min(x), np.max(x)))
        if y_lim is None:
            y_lim = np.array((np.min(y), np.max(y)))
        x_discretize = np.arange(x_lim[0], x_lim[1] + self.eps, (x_lim[1] - x_lim[0]) / 100)
        y_discretize = np.arange(y_lim[0], y_lim[1] + self.eps, (y_lim[1] - y_lim[0]) / 100)
        elec_loc_id = np.empty([len(elec_loc),2])
        for i in range(elec_loc.shape[0]):
            elec_loc_id[i,0] = np.argmin(np.abs((x_discretize)-elec_loc[i,0]))
            elec_loc_id[i,1] = np.argmin(np.abs(np.flip(y_discretize)-elec_loc[i,1]))
        spacing_x = (x_lim[1] - x_lim[0]) / 100
        spacing_y = (y_lim[1] - y_lim[0]) / 100
        data_projected = np.zeros((len(x_discretize), len(y_discretize)))
        for i in range(len(x_discretize)):
            for j in range(len(y_discretize)):
                data_projected[i, j] = np.mean(data_flatten[np.where(np.square(x - x_discretize[i]) + np.square(y - y_discretize[j]) <= spacing_x ** 2 + spacing_y ** 2)])
        data_projected = np.transpose(data_projected)
        data_projected = np.flip(data_projected, axis=0)
        hp = sns.heatmap(data_projected, cmap="seismic")
        cbar = hp.figure.axes[-1]
        cbar.set_ylabel(r'mA', fontsize=21)
        cbar.tick_params(labelsize=16)

        for i in range(len(J)):
            if np.abs(J[i])>self.eps:
                plt.text(elec_loc_id[i,0]-1, elec_loc_id[i,1]-1, str(round(J[i],2)), color='white')
        labels_x = np.linspace(x_lim[0], x_lim[1], 11)
        labels_x = [str(round(labels_x[i],1))[0:4] for i in range(len(labels_x))]
        labels_y = np.linspace(y_lim[0], y_lim[1], 11)
        labels_y = [str(round(labels_y[i],1))[0:4] for i in range(len(labels_y))]
        labels_y = np.flip(np.array(labels_y))
        x_ticks = np.linspace(0,len(x_discretize),len(labels_x))
        y_ticks = np.linspace(0, len(y_discretize), len(labels_y))
        plt.xticks(x_ticks, labels_x, fontsize='15')
        plt.yticks(y_ticks, labels_y, fontsize='15')
        plt.xlabel('X-axis (cm)', fontsize='19')
        plt.ylabel('Y-axis (cm)', fontsize='19')
        plt.title('Electrode Pattern', fontsize='21')
        plt.tight_layout()
        if fname is not None:
            plt.savefig(fname)
        if show:
            plt.show()
        else:
            plt.close()
        return elec_lst
    
    
  
    def _setJ(self,J):
        
        self.A_voltage = np.load(self.save_title+"_voltage.npy")
        self.points = np.load(self.save_title+"_locations.npy")
        
        self.J = J
        self.voltage_J = np.matmul(self.A_voltage,J.reshape(-1,1)).flatten()
        
        del self.A_voltage

    def _setJ_lst(self,J_lst):
        self.A_voltage = np.load(self.save_title+"_voltage.npy")
        self.points = np.load(self.save_title+"_locations.npy") 

        self.J_lst = J_lst
        self.voltage_J_lst = [np.matmul(self.A_voltage,J_lst[i].reshape(-1,1)) for i in range(len(J_lst))]
        del self.A_voltage

    def eval_voltage(self, x,y,z, idx=1):
        if np.ndim(x)==0:
            x = np.array([x,])
            y = np.array([y,])
            z = np.array([z,])
        x, y, z = x.reshape(-1,1)*10**(-1), y.reshape(-1,1)*10**(-1), z.reshape(-1,1)*10**(-1) ## mm->cm
        sph = self._cart_to_sph(np.hstack([x,y,z])) 
        r, theta, phi = sph[:,0].reshape(-1,1),sph[:,1].reshape(-1,1), sph[:,2].reshape(-1,1)
         
        if self.J_lst is None:
            raise Exception('Internal injected current pattern is not defined. Try to define the parameter J before using this function!!!!')
       
        r, theta, phi =  r.flatten(), theta.flatten(), phi.flatten()
        x,y,z = x.flatten(), y.flatten(), z.flatten()
        voltage = np.empty(len(theta))
        for i in range(len(theta)):
            r_idx = np.searchsorted(self.r_lst, r[i])#np.argmin(np.abs(self.r_lst-r[i])) 
            theta_idx = np.searchsorted(self.theta_lst, theta[i])
            phi_idx = np.searchsorted(self.phi_lst, phi[i])
            
            if r_idx == 0:
                r_idx_prev = 0
            elif r_idx == len(self.r_lst):
                r_idx_prev = r_idx-1
                r_idx = r_idx-1
            else:
                r_idx_prev = r_idx-1            
            
            if theta_idx == 0:
                theta_idx_prev = 0
            elif theta_idx == len(self.theta_lst):
                theta_idx_prev=theta_idx-1
                theta_idx = theta_idx-1
            else:
                theta_idx_prev = theta_idx-1
            
            if phi_idx == 0:
                phi_idx_prev = 0
            elif phi_idx == len(self.phi_lst):
                phi_idx_prev = phi_idx-1
                phi_idx = phi_idx-1
            else:
                phi_idx_prev = phi_idx-1

            idx1 = r_idx*len(self.phi_lst)*len(self.theta_lst)+phi_idx*len(self.theta_lst)+theta_idx 
            idx2 = r_idx*len(self.phi_lst)*len(self.theta_lst)+phi_idx*len(self.theta_lst)+theta_idx_prev           
            idx3 = r_idx*len(self.phi_lst)*len(self.theta_lst)+phi_idx_prev*len(self.theta_lst)+theta_idx       
            idx4 = r_idx*len(self.phi_lst)*len(self.theta_lst)+phi_idx_prev*len(self.theta_lst)+theta_idx_prev           
            
            idx5 = r_idx_prev*len(self.phi_lst)*len(self.theta_lst)+phi_idx*len(self.theta_lst)+theta_idx 
            idx6 = r_idx_prev*len(self.phi_lst)*len(self.theta_lst)+phi_idx*len(self.theta_lst)+theta_idx_prev           
            idx7 = r_idx_prev*len(self.phi_lst)*len(self.theta_lst)+phi_idx_prev*len(self.theta_lst)+theta_idx       
            idx8 = r_idx_prev*len(self.phi_lst)*len(self.theta_lst)+phi_idx_prev*len(self.theta_lst)+theta_idx_prev           

            idx_lst = [idx1,idx2,idx3,idx4,idx5,idx6,idx7,idx8]
            voltage_avg_lst = np.array([self.voltage_J_lst[idx-1][idx_lst[j]] for j in range(len(idx_lst))])
            points_avg_lst = self.points[idx_lst] 
            points_avg_lst = self._sph_to_cart(points_avg_lst)
            dist = [np.sqrt((points_avg_lst[j,0]-x[i])**2+(points_avg_lst[j,1]-y[i])**2+(points_avg_lst[j,2]-z[i])**2) for j in range(len(idx_lst))]
            dist = np.array(dist)
            RBF_var = 0.015 
            RBF_dist = np.exp(-dist/RBF_var)
            RBF_dist = RBF_dist/np.sum(RBF_dist)
            if np.abs(np.sum(dist))<self.eps:
                voltage[i] = np.mean(voltage)
            else:
                voltage[i] = np.sum(np.multiply(RBF_dist,voltage_avg_lst.flatten()))
            #print("Voltage Interpolated",np.sum(np.multiply(RBF_dist,voltage_avg_lst.flatten())))
            #print("Voltage ", voltage_avg_lst*10**3)
            #print("Distance", dist)
            #print("RBF", RBF_dist)
            #continue_flag = input('Continue??')
            return voltage*10**3 #mV 

    def calc_distance(self, points):
        points = np.concatenate([self.r_max*10**2*np.ones([points.shape[0],1]), points.copy()], axis=1)
        points_cart = self._sph_to_cart(points)
        
        self.dist_mat = np.empty((len(points), len(points)))
        for i in range(len(points)):
            for j in range(len(points)):
                #self.dist_mat[i,j] = np.abs(points_cart[i,1]-points_cart[j,1])
                self.dist_mat[i, j] = np.sqrt(np.sum((points_cart[i]-points_cart[j])**2))                
                if np.isnan(self.dist_mat[i,j]):
                    print(points[i],points[j])
                    print(np.cos(points[i, 0])*np.cos(points[j, 0])*np.cos(points[i, 1] - points[j, 1])+np.sin(points[i, 0]) * np.sin(points[j, 0]))


    def HingePlace_3d(self, direction, Jdes, Isafety, Jtol, Itotal, Af=None, Ac=None,p=1, verbose=True):
        Af_x, Af_y, Af_z = Af[0], Af[1], Af[2]
        Ac_x, Ac_y, Ac_z = Ac[0], Ac[1], Ac[2]
        I = cvx.Variable(Af_x.shape[1])
        constraints = [direction[0]*(Af_x@I)+direction[1]*(Af_y@I)+direction[2]*(Af_z@I)== Jdes, cvx.sum(I) == 0, cvx.atoms.norm_inf(I) <= Isafety, cvx.atoms.norm1(I) <= Itotal]
        
        J_x, J_y, J_z = Ac_x@I, Ac_y@I, Ac_z@I
        J = np.concatenate([Ac_x,Ac_y,Ac_z], axis=0)@I
        Jtol_concat = np.concatenate([Jtol[0]*np.ones((Ac_x.shape[0])), Jtol[1]*np.ones((Ac_y.shape[0])), Jtol[2]*np.ones((Ac_z.shape[0]))])
        if p>1:
            term1 = cvx.atoms.sum(cvx.atoms.power(cvx.atoms.maximum(0,J-Jtol_concat),p))
            term2 = cvx.atoms.sum(cvx.atoms.power(cvx.atoms.maximum(0,-J-Jtol_concat),p))
            #term1 = cvx.atoms.pnorm(cvx.atoms.maximum(0,J-Jtol_concat),p)
            #term2 = cvx.atoms.pnorm(cvx.atoms.maximum(0,-J-Jtol_concat),p)         
        else:
            term1 = cvx.sum(cvx.atoms.maximum(0,J_x-Jtol[0]*np.ones((Ac_x.shape[0]))))+cvx.sum(cvx.atoms.maximum(0,J_y- Jtol[1]*np.ones((Ac_y.shape[0]))))+cvx.sum(cvx.atoms.maximum(0,J_z-Jtol[2]*np.ones((Ac_z.shape[0]))))
            term2 = cvx.sum(cvx.atoms.maximum(0,-J_x-Jtol[0]*np.ones((Ac_x.shape[0]))))+cvx.sum(cvx.atoms.maximum(0,-J_y-Jtol[1]*np.ones((Ac_y.shape[0]))))+cvx.sum(cvx.atoms.maximum(0,-J_z-Jtol[2]*np.ones((Ac_z.shape[0]))))
        
        obj = cvx.Minimize(term1+term2)
        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=verbose, max_iter=700, solver=cvx.CLARABEL)
        
        print("status:", prob.status)
        print("optimal value", prob.value)
        if prob.status == 'infeasible':
            return None
        else:
            return I.value
    
    def SparsePlace_3d(self, direction, Jdes, Isafety, Itotal, Af=None, Ac=None):
        Af_x, Af_y, Af_z = Af[0], Af[1], Af[2]
        Ac_x, Ac_y, Ac_z = Ac[0], Ac[1], Ac[2]
        I = cvx.Variable(Af_x.shape[1])
        constraints = [direction[0]*(Af_x@I)+direction[1]*(Af_y@I)+direction[2]*(Af_z@I)== Jdes, cvx.sum(I) == 0, cvx.atoms.norm_inf(I) <= Isafety, cvx.atoms.norm1(I) <= Itotal]
        J_x, J_y, J_z = Ac_x@I, Ac_y@I, Ac_z@I
        term1 = cvx.atoms.square(cvx.atoms.norm(J_x))+cvx.atoms.square(cvx.atoms.norm(J_y))+cvx.atoms.square(cvx.atoms.norm(J_z))        
        obj = cvx.Minimize(term1)
        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=True, max_iter=500)#solver=cvx.MOSEK #max_iters  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        if prob.status == 'infeasible':
            return None
        else:
            return I.value
        
@ray.remote 
def HingePlace_3d_uppr_bound(Jdes, Isafety, Jtol, Itotal, Af=None, Ac=None, nu=0):
    Af = np.concatenate(Af, axis=0)
    print(np.linalg.cond(Af))
    Ac_x, Ac_y, Ac_z = Ac[0], Ac[1], Ac[2]
    I = cvx.Variable(Af.shape[1])
    constraints = [Af@I == Jdes, cvx.sum(I) == 0, I <= Isafety*np.ones(Af.shape[1]), cvx.atoms.norm1(I) <= Itotal]
    
    J_x, J_y, J_z = Ac_x@I, Ac_y@I, Ac_z@I
    term1 = cvx.sum(cvx.atoms.maximum(0,J_x-Jtol*np.ones((Ac_x.shape[0]))))+cvx.sum(cvx.atoms.maximum(0,J_y-Jtol*np.ones((Ac_y.shape[0]))))+cvx.sum(cvx.atoms.maximum(0,J_z-Jtol*np.ones((Ac_z.shape[0]))))
    term2 = cvx.sum(cvx.atoms.maximum(0,-J_x-Jtol*np.ones((Ac_x.shape[0]))))+cvx.sum(cvx.atoms.maximum(0,-J_y-Jtol*np.ones((Ac_y.shape[0]))))+cvx.sum(cvx.atoms.maximum(0,-J_z-Jtol*np.ones((Ac_z.shape[0]))))
    
    obj = cvx.Minimize(term1+term2+nu*1e-09*cvx.atoms.norm(I))
    prob = cvx.Problem(obj, constraints)
    prob.solve(verbose=True, max_iter=1000)#solver=cvx.MOSEK #max_iters  # Returns the optimal value.
    print("status:", prob.status)
    print("optimal value", prob.value)
    if prob.status == 'infeasible':
        return None
    else:
        return I.value, prob.status    


@ray.remote
def L1L1norm_3d_uppr_bound(Jdes, Isafety, epsilon, alpha, Itotal, Af=None, Ac=None):
    Af_x, Af_y, Af_z = Af[0], Af[1], Af[2]
    Ac_x, Ac_y, Ac_z = Ac[0], Ac[1], Ac[2]
    I = cvx.Variable(Af_x.shape[1])
    constraints = [cvx.sum(I) == 0, I <= Isafety*np.ones(Af_x.shape[1]), cvx.atoms.norm1(I) <= Itotal]
    J_x, J_y, J_z = Ac_x@I, Ac_y@I, Ac_z@I
    zeta = np.linalg.norm(np.concatenate([Af_x,Af_y,Af_z], axis=0), ord=1)
    nu = np.max(np.abs(Jdes))
    psi_ep = cvx.atoms.maximum(epsilon,cvx.atoms.abs(J_x)/nu)+cvx.atoms.maximum(epsilon,cvx.atoms.abs(J_y)/nu)+cvx.atoms.maximum(epsilon,cvx.atoms.abs(J_z)/nu)
    obj = cvx.atoms.norm1(Af_x@I-Jdes[:Af_x.shape[0]])+cvx.atoms.norm1(Af_y@I-Jdes[Af_x.shape[0]:Af_x.shape[0]+Af_y.shape[0]])+cvx.atoms.norm1(Af_z@I-Jdes[Af_x.shape[0]+Af_y.shape[0]:Af_x.shape[0]+Af_y.shape[0]+Af_z.shape[0]])
    obj = cvx.Minimize(obj+cvx.atoms.norm1(psi_ep)+alpha*zeta*cvx.atoms.norm1(I)+1e-09*cvx.atoms.norm(I))
    prob = cvx.Problem(obj, constraints)
    prob.solve(verbose=True, max_iter=500)#solver=cvx.MOSEK #max_iters  # Returns the optimal value.
    print("status:", prob.status)
    print("optimal value", prob.value)
    if prob.status == 'infeasible':
        return None
    else:
        return I.value, prob.status

@ray.remote
def constrained_Maximization(direction, alpha, Isafety, Itotal, Af=None, Ac=None):
    Af_x, Af_y, Af_z = Af[0], Af[1], Af[2]
    Ac_x, Ac_y, Ac_z = Ac[0], Ac[1], Ac[2]
    I = cvx.Variable(Af_x.shape[1])
    J_x, J_y, J_z = Ac_x@I, Ac_y@I, Ac_z@I
    J_norm = cvx.atoms.norm(J_x)**2+cvx.atoms.norm(J_y)**2+cvx.atoms.norm(J_z)**2
    constraints = [J_norm<= alpha, cvx.sum(I) == 0, cvx.atoms.norm_inf(I)<=Isafety, cvx.atoms.norm1(I)<=Itotal]
    obj = cvx.Maximize(cvx.sum(direction[0]*(Af_x@I)+direction[1]*(Af_y@I)+direction[2]*(Af_z@I)))
    prob = cvx.Problem(obj, constraints)
    prob.solve(verbose=True, max_iter=700)  # Returns the optimal value.
    print("status:", prob.status)
    print("optimal value", prob.value)
    if prob.status == 'infeasible':
        return None
    else:
        return I.value, prob.status

@ray.remote
def SparsePlace_3d(direction, Jdes, Isafety, Itotal, Af=None, Ac=None): 
    Af_x, Af_y, Af_z = Af[0], Af[1], Af[2]
    Ac_x, Ac_y, Ac_z = Ac[0], Ac[1], Ac[2]
    I = cvx.Variable(Af_x.shape[1])
    constraints = [direction[0]*(Af_x@I)+direction[1]*(Af_y@I)+direction[2]*(Af_z@I)== Jdes, cvx.sum(I) == 0, cvx.atoms.norm_inf(I) <= Isafety, cvx.atoms.norm1(I) <= Itotal]
    J_x, J_y, J_z = Ac_x@I, Ac_y@I, Ac_z@I
    term1 = cvx.atoms.square(cvx.atoms.norm(J_x))+cvx.atoms.square(cvx.atoms.norm(J_y))+cvx.atoms.square(cvx.atoms.norm(J_z))        
    obj = cvx.Minimize(term1)
    prob = cvx.Problem(obj, constraints)
    prob.solve(verbose=True, max_iter=700)#solver=cvx.MOSEK #max_iters  # Returns the optimal value.
    print("status:", prob.status)
    print("optimal value", prob.value)
    if prob.status == 'infeasible':
        return None
    else:
        return I.value, prob.status


