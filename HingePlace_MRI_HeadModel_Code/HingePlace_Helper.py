import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600

def find_elec_id(elec_name, electrode_names):
    elec_id = 0
    for name in electrode_names:
        if name == elec_name:
            break
        else:
            elec_id = elec_id + 1
    return elec_id

def plot_electric_field(Efield, savepath, locations):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    color_map = cm.ScalarMappable(cmap='jet')
    rgba = color_map.to_rgba(x=Efield, norm=True)
    ax.scatter(locations[:, 0], locations[:, 1], locations[:, 2], c=rgba)
    ax.set_xlabel('X-axis (mm)', fontsize=20, labelpad=20)
    ax.set_ylabel('Y-axis (mm)', fontsize=20, labelpad=4)
    ax.set_zlabel('Z-axis (mm)', fontsize=20, labelpad=10)
    ax.tick_params(axis='x', labelsize=18, pad=10)
    ax.tick_params(axis='y', labelsize=18, pad=-2)
    ax.tick_params(axis='z', labelsize=18)
    cbar = plt.colorbar(mappable=color_map, ax=ax)
    cbar.set_label(label=r"Vm$^{-1}$", size=20)
    cbar.ax.tick_params(labelsize=18)
    cbar.ax.set_label(r"Vm$^{-1}$")
    ax.view_init(40,180)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

def plot_activation(Efield, locations, savepath, locations_cancel, view_init=[0,180]):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(locations_cancel[np.abs(Efield) > 0.8, 0], locations_cancel[np.abs(Efield) > 0.8, 1], locations_cancel[np.abs(Efield) > 0.8, 2], alpha=1, linewidths=0, edgecolor=None, c='red', s=30)
    ax.scatter(locations[:, 0], locations[:, 1], locations[:, 2], alpha=0.05, s=0.1, c='salmon')
    ax.set_ylabel('Y-axis (mm)', fontsize=20, labelpad=20)
    ax.set_zlabel('Z-axis (mm)', fontsize=20, labelpad=15)
    ax.tick_params(axis='x', labelsize=18, pad=10, which='both', bottom=False, top=False, labelbottom=False, labeltop=False, labelright=False, labelleft=False)
    ax.tick_params(axis='y', labelsize=18, pad=-2)
    ax.tick_params(axis='z', labelsize=18)
    # ax.set_ylim([80, 150])
    plt.tight_layout()
    ax.view_init(view_init[0], view_init[180])
    plt.savefig(savepath)
    plt.close()