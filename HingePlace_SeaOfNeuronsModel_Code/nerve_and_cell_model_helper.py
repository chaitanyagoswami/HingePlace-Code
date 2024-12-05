import numpy as np
import math

def cart_to_sph(pos):
    if len(pos.shape) == 1:
        pos = pos.reshape(1,-1)
    r = np.sqrt(np.sum(pos**2, axis=1)).reshape(-1,1)
    theta = np.arcsin(pos[:,2].reshape(-1,1)/r).reshape(-1,1)
    phi = np.arctan2(pos[:,1],pos[:,0]).reshape(-1,1)
    sph_pos = np.hstack([r,theta,phi])
    sph_pos[sph_pos[:,2]<0,2] = sph_pos[sph_pos[:,2]<0,2]+2*np.pi
    return sph_pos
    
def sph_to_cart(pos):
    if len(pos.shape) == 1:
        pos = pos.reshape(1,-1)
    x = pos[:,0]*np.cos(pos[:,1])*np.cos(pos[:,2])
    y = pos[:,0]*np.cos(pos[:,1])*np.sin(pos[:,2])
    z = pos[:,0]*np.sin(pos[:,1])
    cart_pos = np.hstack([x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)])
    return cart_pos

def cart_to_sph_v2(pos):
    if len(pos.shape) == 1:
        pos = pos.reshape(1,-1)
    r = np.sqrt(np.sum(pos**2, axis=1)).reshape(-1,1)
    theta = np.arccos(pos[:,2].reshape(-1,1)/r).reshape(-1,1)
    phi = np.arctan2(pos[:,1],pos[:,0]).reshape(-1,1)
    sph_pos = np.hstack([r,theta,phi])
    sph_pos[sph_pos[:,2]<0,2] = sph_pos[sph_pos[:,2]<0,2]+2*np.pi
    return sph_pos
    
def sph_to_cart_v2(pos):
    if len(pos.shape) == 1:
        pos = pos.reshape(1,-1)
    x = pos[:,0]*np.sin(pos[:,1])*np.cos(pos[:,2])
    y = pos[:,0]*np.sin(pos[:,1])*np.sin(pos[:,2])
    z = pos[:,0]*np.cos(pos[:,1])
    cart_pos = np.hstack([x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)])
    return cart_pos


def fibonacci_sphere(samples=1000):
    points = []
    phi = math.pi*(math.sqrt(5.)-1.)  # golden angle in radians
    for i in range(samples):
        y = 1-(i/float(samples-1))*2  # y goes from 1 to -1
        radius = math.sqrt(1-y*y)  # radius at y
        theta = phi*i  # golden angle increment
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))
    return np.array(points)

def sample_spherical(num_samples, theta_max, y_max, x_max, r=1):
    tot_samples = 0
    samples = []
    while tot_samples<num_samples:
        samples_cand = np.random.normal(loc=0, scale=1, size=(10000,3))
        samples_cand = samples_cand/np.sqrt(np.sum(samples_cand**2, axis=1)).reshape(-1,1)*r
        samples_cand = cart_to_sph(samples_cand)
        samples_cand = samples_cand[samples_cand[:,1]>theta_max]
        samples_cand = sph_to_cart(samples_cand)
        if np.size(y_max) == 1 and np.size(x_max)==1:
            samples_cand = samples_cand[np.abs(samples_cand[:,1])<y_max]
            samples_cand = samples_cand[np.abs(samples_cand[:,0])<x_max]
        elif np.size(y_max)==1 and np.size(x_max)==2:
            samples_cand = samples_cand[np.abs(samples_cand[:,1])<y_max]
            samples_cand = samples_cand[samples_cand[:,0]<x_max[1]]
            samples_cand = samples_cand[samples_cand[:,0]>x_max[0]]
        elif np.size(y_max)==2 and np.size(x_max)==1:
            samples_cand = samples_cand[np.abs(samples_cand[:,0])<x_max]
            samples_cand = samples_cand[samples_cand[:,1]<y_max[1]]
            samples_cand = samples_cand[samples_cand[:,1]>y_max[0]]
        elif np.size(y_max)==2 and np.size(x_max)==2:
            samples_cand = samples_cand[samples_cand[:,0]<x_max[1]]
            samples_cand = samples_cand[samples_cand[:,0]>x_max[0]]
            samples_cand = samples_cand[samples_cand[:,1]<y_max[1]]
            samples_cand = samples_cand[samples_cand[:,1]>y_max[0]]
        else:
            raise Exception('Check the size of x_max and y_max! Should be a 2-element list or a float')
        samples.append(samples_cand)
        tot_samples = tot_samples+samples_cand.shape[0]
    samples = np.vstack(samples)
    samples = samples[:num_samples]
    return samples

##### Plotting Functions #########
def plot_scalp_model(length_parietal=16, length_parietal_xlen=18, length_temporal_ylen=10, spacing=0.5, temp=37, dt=0.025, periphery_only=True, r=9.2, nfibers=50, elec_field_lst=None):
    
    shifts_parietal = np.arange(-length_parietal_xlen/2+1,length_parietal_xlen/2-1, spacing)
    shifts_temporal = np.arange(-length_temporal_ylen/2, length_temporal_ylen/2, spacing)
    temporal_flip = [False]*len(shifts_parietal)+[True]*len(shifts_temporal)+[False]*len(shifts_temporal)
    shifts_temporal = np.concatenate([shifts_temporal.copy(), shifts_temporal.copy()], axis=0)
    total_fibers  = len(shifts_parietal)+len(shifts_temporal)
    shifts = np.concatenate([shifts_parietal, shifts_temporal], axis=0)
    temporal = [False]*len(shifts_parietal)+[True]*len(shifts_temporal)
    length = [length_parietal]*len(shifts_parietal)+[length_parietal_xlen]*len(shifts_temporal)

    cfibers = [CfiberSim.remote(length=length[i], temp=temp, dt=dt, periphery_only=periphery_only, temporal=temporal[i], temporal_flip=temporal_flip[i], shift=shifts[i], r=r, elec_field_lst=elec_field_lst) for i in range(total_fibers)]
    coord = ray.get([cfibers[i]._get_coord.remote() for i in range(total_fibers)])
    view_angle = np.linspace(0,360,361)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    for i in range(len(coord)):
        ax.scatter(coord[i][:,0]*10**(-4), coord[i][:,1]*10**(-4), coord[i][:,2]*10**(-4), linewidth=0.3, alpha=1.0, c='blue', s=2)
    ax.scatter(x*10**(-1),y*10**(-1),z*10**(-1),c='red', s=10)
    ax.set_xlabel('X-axis (cm)', fontsize=16)
    ax.set_ylabel('Y-axis (cm)', fontsize=16)
    ax.set_zlabel('Z-axis (cm)', fontsize=16)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    lim = [np.min([xlim[0],ylim[0],zlim[0]]), np.max([xlim[1],ylim[1],zlim[1]])]
    #ax.set_zlim(zmin=lim[0], zmax=lim[1])
    #ax.set_xlim(xmin=lim[0], xmax=lim[1])
    #ax.set_ylim(ymin=lim[0], ymax=lim[1])
    ax.tick_params(axis='x',labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='z',labelsize=12)
    plt.tight_layout()
    
    #def update(frame):
    #    ax.view_init(10,view_angle[frame])
    #ani = animation.FuncAnimation(fig=fig, func=update, frames=361, interval=20)
    ##ani.save(os.path.join(neuron_orient_dir,'NeuronOrientation_cellid'+str(cell_id)+'_layer5.gif'), writer='pillow')
    plt.show()
 
