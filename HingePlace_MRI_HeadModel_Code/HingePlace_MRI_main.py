from scipy.io import loadmat
import os
import numpy as np
from HingePlace_optimizers import HingePlace_p1, HingePlace_p2, HingePlace_p3, SparsePlace
import matplotlib.pyplot as plt
import matplotlib as mpl
from HingePlace_Helper import find_elec_id, plot_electric_field, plot_activation
mpl.rcParams['figure.dpi'] = 600

cwd = os.getcwd()
forward_mat_dir = os.path.join(cwd,'Forward_Matrix_MRI')
results_dir = os.path.join(cwd,'HingePlaceMRI_Results')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
base_dir = os.path.join(results_dir,'Sec-5-1-1')
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

SEED = 12344
mni2mri = np.array([[1.0226,0.0007,-0.0020,92.2735], [0.0106,1.0132,-0.0193,126.9093], [0.0035,0.0033,0.9907,74.4404], [0,0,0,1.0000]])

ORIGIN_MRI_COORD = mni2mri@np.array([0,0,0,1]).reshape(-1,1)
ORIGIN_MRI_COORD = np.squeeze(np.round(ORIGIN_MRI_COORD)[:3])

#### Electrode Locations Names
electrode_names = []
with open(os.path.join(forward_mat_dir,'ElectrodeNames.txt'), 'r') as f:
    for lines in f.readlines():
        electrode_names.append(lines[:-1])

#### Loading Forward Matrices and locations
Aall = loadmat(os.path.join(forward_mat_dir,'Aall.mat'))
Aall = Aall['A_brain']
locations = loadmat(os.path.join(forward_mat_dir,'loc_all.mat'))
locations = locations['locs_brain']
elec_locations = loadmat(os.path.join(forward_mat_dir,'elec_locs.mat'))
elec_locations = elec_locations['elec_locs']

#### Plot Brain and Electrode Locations
###############################################
PLOT_ELEC_PTTRN = False
if PLOT_ELEC_PTTRN:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    start_transparency = 0.1
    ax.scatter(locations[:, 0], locations[:, 1], locations[:, 2], c='salmon', s=0.1)
    ax.scatter(elec_locations[:, 0], elec_locations[:, 1], elec_locations[:, 2], c='blue', s=1)
    ax.set_ylabel('Y-axis (mm)', fontsize=20, labelpad=10)
    ax.set_zlabel('Z-axis (mm)', fontsize=20, labelpad=10)
    ax.tick_params(axis='x', labelsize=18, pad=10,which='both',bottom=False,top=False,labelbottom=False)
    ax.tick_params(axis='y', labelsize=18, pad=-2)
    ax.tick_params(axis='z', labelsize=18)
    plt.tight_layout()
    ax.view_init(0,180)
    plt.savefig(os.path.join(results_dir,'Electrode_Plot_SagittalView.png'))
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    start_transparency = 0.1
    ax.scatter(locations[:, 0], locations[:, 1], locations[:, 2], c='salmon', s=0.1)
    ax.scatter(elec_locations[:, 0], elec_locations[:, 1], elec_locations[:, 2], c='blue', s=1)
    ax.set_xlabel('X-axis (mm)', fontsize=20, labelpad=20)
    ax.set_ylabel('Y-axis (mm)', fontsize=20, labelpad=30)
    ax.tick_params(axis='x', labelsize=18, pad=5)
    ax.tick_params(axis='y', labelsize=18, pad=15)
    ax.tick_params(axis='z', labelsize=18, which='both',bottom=False,top=False,labelbottom=False, labeltop=False, labelright=False, labelleft=False)
    plt.tight_layout()
    ax.view_init(90,270)
    plt.savefig(os.path.join(results_dir,'Electrode_Plot_TopView.png'))
    plt.close()

#### Plot Target Locations in the Brain
###############################################
PLOT_TARGET_LOCS = False
if PLOT_TARGET_LOCS:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    start_transparency = 0.1
    mni_target_locs = np.array([[-48, -8, 50],[-48, -21, 50], [-22,-20,77], [5,8,61], [48, -8, 50]])
    target_locs_lst = np.empty([mni_target_locs.shape[0],3])
    target_radius_lst = [2,2,3,2,2]
    for i in range(len(mni_target_locs)):
        temp = np.squeeze(np.matmul(mni2mri,np.array(list(mni_target_locs[i])+[1]).reshape(-1,1)))
        target_loc = np.round(temp[0:3])
        dist_target = np.sqrt(np.sum(np.square(locations - target_loc.reshape(1, 3)), axis=1))
        if i==0:
            ax.scatter(locations[dist_target<target_radius_lst[i], 0], locations[dist_target<target_radius_lst[i], 1], locations[dist_target<target_radius_lst[i], 2], s=50, label='MC')
        else:
            ax.scatter(locations[dist_target < target_radius_lst[i], 0], locations[dist_target < target_radius_lst[i], 1],locations[dist_target < target_radius_lst[i], 2], s=50, label='%d'%i)
    ax.scatter(locations[:, 0], locations[:, 1], locations[:, 2], c='salmon', s=0.1,alpha=0.05)
    ax.set_xlabel('X-axis (mm)', fontsize=20, labelpad=20)
    ax.set_ylabel('Y-axis (mm)', fontsize=20, labelpad=30)
    ax.tick_params(axis='x', labelsize=18, pad=5)
    ax.tick_params(axis='y', labelsize=18, pad=15)
    ax.tick_params(axis='z', labelsize=18, which='both',bottom=False,top=False,labelbottom=False, labeltop=False, labelright=False, labelleft=False)
    ax.view_init(90,270)
    plt.legend(fontsize=20, bbox_to_anchor=(0.84,0.77))
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir,'TargetLocs.png'))
    plt.close()

###############################################
#### Radial-in Direction
directions_all = locations-ORIGIN_MRI_COORD.reshape(1,3)
directions_all = -1*directions_all/np.sqrt(np.sum(np.square(directions_all), axis=1)).reshape(-1,1)
Aall = directions_all[:,0].reshape(-1,1)*Aall[:,0,:]+directions_all[:,1].reshape(-1,1)*Aall[:,1,:]+directions_all[:,2].reshape(-1,1)*Aall[:,2,:]
A_plot = Aall.copy()

#### Target Location
mni_target_loc = np.array([-48,-8,50])
target_loc = mni2mri@np.concatenate([mni_target_loc,[1]]).reshape(-1,1)
target_loc = np.round(target_loc)
target_loc = target_loc[:3]

#### Creating Focus and Cancel Matrix
dist_target = np.sqrt(np.sum(np.square(locations-target_loc.reshape(1,3)), axis=1))
target_radius = 2 ## distance in voxel space
idx_focus = np.arange(locations.shape[0])[dist_target<=target_radius]
idx_cancel_all = np.setdiff1d(np.arange(locations.shape[0]), idx_focus)
Af, A_check = np.mean(A_plot[idx_focus], axis=0).reshape(1,-1), A_plot[idx_cancel_all]
locations_cancel = locations[idx_cancel_all]
target_cancel = 40 ## distance in voxel space
idx_cancel = np.arange(locations.shape[0])[dist_target<=target_cancel]
idx_cancel = np.setdiff1d(idx_cancel,idx_focus)
tmp = np.arange(locations.shape[0])[dist_target>target_cancel]
tmp = tmp[np.random.permutation(tmp.shape[0])[:int(0.25*tmp.shape[0])]]
idx_cancel = np.array(list(idx_cancel)+list(tmp))
Ac = A_plot[idx_cancel]

#### Different Losses Illustration
PLOT_LOSSES = False
if PLOT_LOSSES:
    #### Baseline Electrode Pattern
    C3_id, C4_id = find_elec_id('C3', electrode_names=electrode_names), find_elec_id('C4', electrode_names=electrode_names)
    I = np.zeros(len(electrode_names))
    I[C3_id] = 1
    I[C4_id] = -1
    ##### Cancel Electric Field
    Ec = A_check@I
    Ec = np.array([np.max(Ec),])
    scale = np.linspace(-10,10,100)
    square_loss = scale**2*(np.sum(np.square(Ec)))
    hinge_loss1, hinge_loss2, hinge_loss3, hinge_loss5, hinge_loss10, hinge_loss_inf = [], [], [], [], [], []
    for i in range(len(scale)):
        hinge_loss_vector = np.max(np.concatenate([np.zeros([Ec.shape[0],1]), Ec.reshape(-1,1)*scale[i]-0.7], axis=1), axis=1)
        hinge_loss_vector = hinge_loss_vector+np.max(np.concatenate([np.zeros([Ec.shape[0], 1]), -Ec.reshape(-1,1)*scale[i]-0.7], axis=1),axis=1)
        hinge_loss1.append(np.sum(hinge_loss_vector))
        hinge_loss2.append(np.sum(np.square(hinge_loss_vector)))
        hinge_loss3.append(np.sum(hinge_loss_vector**3))
        hinge_loss5.append(np.sum(hinge_loss_vector**5))
        hinge_loss10.append(np.sum(hinge_loss_vector**10))
        hinge_loss_inf.append(np.max(hinge_loss_vector))

    hinge_loss1 = np.array(hinge_loss1)
    hinge_loss2 = np.array(hinge_loss2)
    hinge_loss3 = np.array(hinge_loss3)
    hinge_loss5 = np.array(hinge_loss5)
    hinge_loss10 = np.array(hinge_loss10)
    hinge_loss_inf = np.array(hinge_loss_inf)
    linestyle='solid'
    plt.plot(scale*Ec,1/np.max(square_loss)*square_loss, label='Square', linestyle=linestyle, linewidth=2)
    plt.plot(scale*Ec,1/np.max(hinge_loss1)*hinge_loss1, label='Hinge p=1', linestyle=linestyle, linewidth=2)
    plt.plot(scale*Ec,1/np.max(hinge_loss2)*hinge_loss2, label='Hinge p=2', linestyle=linestyle, linewidth=2)
    plt.plot(scale*Ec,1/np.max(hinge_loss3)*hinge_loss3, label='Hinge p=3', linestyle=linestyle, linewidth=2)
    plt.plot(scale*Ec,1/np.max(hinge_loss10)*hinge_loss10, label='Hinge p=10',linestyle=linestyle, linewidth=2)
    plt.vlines(x=0.7,ymin=0, ymax=1, linestyles='--', color='black')
    plt.vlines(x=-0.7,ymin=0, ymax=1, linestyles='--', color='black')
    plt.text(x=0.15,y=0.3, s=r'$E_{tol}^+$', fontsize=20)
    plt.text(x=-0.65,y=0.3, s=r'$E_{tol}^-$', fontsize=20)
    plt.legend(fontsize=16)
    plt.xlabel(r"E-field at a single voxel (Vm$^{-1}$)", fontsize=22)
    plt.ylabel("Normalized Loss", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir,'HingeLoss_Visualisation.png'))
    plt.show()

    linestyle='solid'
    plt.plot(scale*Ec,1/np.max(square_loss)*square_loss, label='Square', linestyle=linestyle, linewidth=5, color='C0')
    plt.vlines(x=0.7,ymin=0, ymax=1, linestyles='--', color='black')
    plt.vlines(x=-0.7,ymin=0, ymax=1, linestyles='--', color='black')
    plt.text(x=0.1,y=0.3, s=r'$E_{tol}^+$', fontsize=24)
    plt.text(x=-0.65,y=0.3, s=r'$E_{tol}^-$', fontsize=24)
    plt.hlines(y=0,xmin=np.min(scale*Ec), xmax=np.max(scale*Ec), color='black', linestyles='dashed', linewidth=2.0)
    plt.xlabel(r"E-field at a single voxel (Vm$^{-1}$)", fontsize=24)
    plt.ylabel("Normalized Loss", fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.tight_layout()
    plt.savefig(os.path.join('HingeLoss_SquareLoss.png'))
    plt.show()

    linestyle='solid'
    plt.plot(scale*Ec,1/np.max(hinge_loss2)*hinge_loss2, label='Hinge p=2', linestyle=linestyle, linewidth=5, color='C2')
    plt.vlines(x=0.7,ymin=0, ymax=1, linestyles='--', color='black')
    plt.vlines(x=-0.7,ymin=0, ymax=1, linestyles='--', color='black')
    plt.text(x=0.1,y=0.3, s=r'$E_{tol}^+$', fontsize=24)
    plt.text(x=-0.65,y=0.3, s=r'$E_{tol}^-$', fontsize=24)
    plt.hlines(y=0,xmin=np.min(scale*Ec), xmax=np.max(scale*Ec), color='black', linestyles='dashed', linewidth=2.0)
    plt.xlabel(r"E-field at a single voxel (Vm$^{-1}$)", fontsize=24)
    plt.ylabel("Normalized Loss", fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir,'HingeLoss_HingeLoss.png'))
    plt.show()

#### Hyperparamaters
Edes = 1 ## V/m
Isafety_lst = np.linspace(3,4.5,5) ## mA
Itot_mul = [2,4,8]
Etol = [0.1*Edes, 0.5*Edes, 0.6*Edes, 0.7*Edes] ## p=1
Etol_HP2 = [0.65*Edes, 0.55*Edes, 0.35*Edes, 0.01*Edes] ## p=2
Etol_HP3 = [0.65*Edes, 0.55*Edes, 0.35*Edes, 0.01*Edes] ## p=3
rel_inc_HP, rel_inc_HP2, rel_inc_HP3,  Isafe = [], [], [], []

for i in range(len(Itot_mul)):

    print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("Running Simulation for Total Current Multiplier %d....." % (Itot_mul[i]))
    print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    savedir_hp = os.path.join(base_dir, "HP_TotalCurrent%d" % int(Itot_mul[i]))
    if not os.path.exists(savedir_hp):
        os.makedirs(savedir_hp)

    savedir_sp = os.path.join(base_dir, "SPTotalCurrent%d" % int(Itot_mul[i]))
    if not os.path.exists(savedir_sp):
        os.makedirs(savedir_sp)

    activ_HP, activ_HP2, activ_HP3, activ_SP = [], [], [], []
    for ii in range(len(Isafety_lst)):
        Isafety = Isafety_lst[ii]
        Itotal = 2 * Itot_mul[i] * Isafety
        print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print("Iteration %d" % (ii + 1))
        print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        #### Running the HingePlace algorithm for p=2 #######
        ## Finding the optimal tolerance HingePlace p=2
        area_HP2, I_HP2 = np.inf, None
        LOAD_FLAG_HP2 = False ## Load already saved patterns
        for kk in range(len(Etol_HP2)): ### Brute-force search to find the best possible Etol value
            if not LOAD_FLAG_HP2:
                I_HP2_tmp = HingePlace_p2(Jdes=Edes, Isafety=Isafety, Jtol=Etol_HP2[kk], Itotal=Itotal, Af=Af, Ac=Ac)
                if I_HP2_tmp is not None:
                    np.save(os.path.join(savedir_hp, "I_HP2_Isafety%dmA_Etol%d.npy" % (int(Isafety * 10), kk + 1)), I_HP2_tmp)
            else:
                I_HP2_tmp = np.load(os.path.join(savedir_hp, "I_HP2_Isafety%dmA_Etol%d.npy" % (int(Isafety*10), kk+1)))

            if I_HP2_tmp is not None and np.sum(np.abs(np.matmul(A_check, I_HP2_tmp)) > 0.8 * Edes) < area_HP2:
                I_HP2 = I_HP2_tmp.copy()
                area_HP2 = np.sum(np.abs(np.matmul(A_check, I_HP2)) > 0.8 * Edes)

        if np.isinf(area_HP2): ### Assign -1 to indicate that the optimization was infeasible for all tested Etol
            area_HP2 = -1

        PLOT_EFIELD_AND_ACTIVATION_HP2 = False
        if PLOT_EFIELD_AND_ACTIVATION_HP2 and area_HP2 != -1: #### Plot the electric field and activated region of the best performing Etol value
            efield_HP2 = np.matmul(A_plot, I_HP2)
            efield_off_target_HP2 = np.matmul(A_check, I_HP2)
            plot_electric_field(Efield=efield_HP2, locations=locations, savepath=os.path.join(savedir_hp,"Efield_I_HP2_Isafety%dmA.png" % (int(Isafety * 10))))
            plot_activation(Efield=efield_off_target_HP2, locations_cancel=locations_cancel, locations=locations,savepath=os.path.join(savedir_hp, "Activation_I_HP2_Isafety%dmA.png" % (int(Isafety * 10))))

        #######################################################
        #### Running the HingePlace algorithm for p=1 #########
        ## Finding the optimal tolerance HingePlace p=1
        area_HP, I_HP = np.inf, None
        LOAD_FLAG_HP = False ## Load already saved patterns
        for kk in range(len(Etol)): ### Brute-force search to find the best possible Etol value
            if not LOAD_FLAG_HP2:
                I_HP_tmp = HingePlace_p1(Jdes=Edes, Isafety=Isafety, Jtol=Etol, Itotal=Itotal, Af=Af, Ac=Ac)
                if I_HP_tmp is not None:
                    np.save(os.path.join(savedir_hp, "I_HP_Isafety%dmA_Etol%d.npy" % (int(Isafety * 10), kk + 1)), I_HP_tmp)
            else:
                I_HP_tmp = np.load(os.path.join(savedir_hp, "I_HP_Isafety%dmA_Etol%d.npy" % (int(Isafety*10), kk+1)))

            if I_HP_tmp is not None and np.sum(np.abs(np.matmul(A_check, I_HP_tmp)) > 0.8 * Edes) < area_HP:
                I_HP = I_HP_tmp.copy()
                area_HP = np.sum(np.abs(np.matmul(A_check, I_HP)) > 0.8 * Edes)

        if np.isinf(area_HP): ### Assign -1 to indicate that the optimization was infeasible for all tested Etol
            area_HP = -1

        PLOT_EFIELD_AND_ACTIVATION_HP = False
        if PLOT_EFIELD_AND_ACTIVATION_HP and area_HP != -1: #### Plot the electric field and activated region of the best performing Etol value
            efield_HP = np.matmul(A_plot, I_HP)
            efield_off_target_HP = np.matmul(A_check, I_HP)
            plot_electric_field(Efield=efield_HP, locations=locations, savepath=os.path.join(savedir_hp,"Efield_I_HP_Isafety%dmA.png" % (int(Isafety * 10))))
            plot_activation(Efield=efield_off_target_HP, locations_cancel=locations_cancel, locations=locations,savepath=os.path.join(savedir_hp, "Activation_I_HP_Isafety%dmA.png" % (int(Isafety * 10))))

        #######################################################
        #### Running the HingePlace algorithm for p=3 #######
        ## Finding the optimal tolerance HingePlace p=3
        area_HP3, I_HP3 = np.inf, None
        LOAD_FLAG_HP3 = False
        for kk in range(len(Etol_HP3)):
            if not LOAD_FLAG_HP3:
                I_HP3_tmp = HingePlace_p3(Jdes=Edes, Isafety=Isafety, Jtol=Etol_HP3[kk], Itotal=Itotal, Af=Af, Ac=Ac)
                if I_HP3_tmp is not None:
                    np.save(os.path.join(savedir_hp, "I_HP3_Isafety%dmA_Etol%d.npy" % (int(Isafety * 10), kk + 1)), I_HP3_tmp)
            else:
                I_HP3_tmp = np.load(os.path.join(savedir_hp, "I_HP3_Isafety%dmA_Etol%d.npy" % (int(Isafety*10), kk + 1)))

            if I_HP3_tmp is not None and np.sum(np.abs(np.matmul(A_check, I_HP3_tmp)) > 0.8 * Edes) < area_HP3:
                area_HP3 = np.sum(np.abs(np.matmul(A_check, I_HP3_tmp)) > 0.8 * Edes)
                I_HP3 = I_HP3_tmp.copy()

        if np.isinf(area_HP3):
            area_HP3 = -1

        PLOT_EFIELD_AND_ACTIVATION_HP3 = False
        if PLOT_EFIELD_AND_ACTIVATION_HP3 and area_HP3 != -1: #### Plot the electric field and activated region of the best performing Etol value
            efield_HP3 = np.matmul(A_plot, I_HP3)
            efield_off_target_HP3 = np.matmul(A_check, I_HP3)
            plot_electric_field(Efield=efield_HP3, locations=locations, savepath=os.path.join(savedir_hp,"Efield_I_HP3_Isafety%dmA.png" % (int(Isafety * 10))))
            plot_activation(Efield=efield_off_target_HP3, locations_cancel=locations_cancel, locations=locations,savepath=os.path.join(savedir_hp, "Activation_I_HP3_Isafety%dmA.png" % (int(Isafety * 10))))

        #######################################################
        ########### Running the LCMV-E algorithm ##############
        ##### In this code we call the LCMV-E as the SparsePlace algorithm due to some legacy naming convention our group had.
        ##### Note that the SparsePlace algorithm utilizes the same optimization objective as the LCMV-E optimization
        LOAD_FLAG_SP = False
        if not LOAD_FLAG_SP:
            I_SP = SparsePlace(Jdes=Edes, Isafety=Isafety, Itotal=Itotal, Af=Af, Ac=Ac)
            if I_SP is not None:
                area_SP = np.sum(np.abs(np.matmul(A_check, I_SP)) > 0.8 * Edes)
                np.save(os.path.join(savedir_sp, "I_SP_Isafety%dmA.npy" % int(Isafety * 10)), I_SP)
            else:
                area_SP = -1
        else:
            I_SP = np.load(os.path.join(savedir_sp, "I_SP_Isafety%dmA.npy" % int(Isafety * 10)))

        PLOT_EFIELD_AND_ACTIVATION_SP = False
        if PLOT_EFIELD_AND_ACTIVATION_SP and area_SP != -1: #### Plot the electric field and activated region of the best performing Etol value
            efield_SP = np.matmul(A_plot, I_SP)
            efield_off_target_HP = np.matmul(A_check, I_HP)
            plot_electric_field(Efield=efield_HP, locations=locations, savepath=os.path.join(savedir_hp,"Efield_I_HP_Isafety%dmA.png" % (int(Isafety * 10))))
            plot_activation(Efield=efield_off_target_HP, locations_cancel=locations_cancel, locations=locations,savepath=os.path.join(savedir_hp, "Activation_I_HP_Isafety%dmA.png" % (int(Isafety * 10))))

        print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print("HP Spikes: %d" % int(area_HP))
        print("HP Spikes2: %d" % int(area_HP2))
        print("HP Spikes3: %d" % int(area_HP3))
        print("SP Spikes: %d" % int(area_SP))
        print("Relative Increase-HP: %.2f" % ((area_SP - area_HP) / area_SP * 100))
        print("Relative Increase-HP2: %.2f" % ((area_SP - area_HP2) / area_SP * 100))
        print("Relative Increase-HP3: %.2f" % ((area_SP - area_HP3) / area_SP * 100))
        activ_HP.append(area_HP)
        activ_HP2.append(area_HP2)
        activ_HP3.append(area_HP3)
        activ_SP.append(area_SP)
        np.save(os.path.join(savedir_hp, "AreaHP_Isafety%dmA.npy" % int(Isafety*10)), area_HP)
        np.save(os.path.join(savedir_hp, "AreaHP2_Isafety%dmA.npy" % int(Isafety*10)), area_HP2)
        np.save(os.path.join(savedir_hp, "AreaHP3_Isafety%dmA.npy" % int(Isafety*10)), area_HP3)
        np.save(os.path.join(savedir_sp, "AreaSP_Isafety%dmA.npy" % int(Isafety*10)), area_SP)

    activ_HP, activ_HP2, activ_HP3, activ_SP = np.array(activ_HP), np.array(activ_HP2), np.array(activ_HP3), np.array(activ_SP)
    np.save(os.path.join(savedir_hp, 'activation_HP.npy'), activ_HP)
    np.save(os.path.join(savedir_hp, 'activation_HP2.npy'), activ_HP2)
    np.save(os.path.join(savedir_hp, 'activation_HP3.npy'), activ_HP3)
    np.save(os.path.join(savedir_sp, 'activation_SP.npy'), activ_SP)

    plt.plot(np.array(Isafety_lst)[activ_SP!=-1], activ_SP[activ_SP!=-1], marker='x', label='DCM')
    plt.plot(np.array(Isafety_lst)[activ_HP!=-1], activ_HP[activ_HP!=-1], marker='x', label='HP1')
    plt.plot(np.array(Isafety_lst)[activ_HP2!=-1], activ_HP2[activ_HP != -1], marker='x', label='HP2')
    plt.plot(np.array(Isafety_lst)[activ_HP3!=-1], activ_HP3[activ_HP != -1], marker='x', label='HP3')
    plt.legend(fontsize=19)
    plt.xlabel(r"$I_{safe}$ (mA)", fontsize=21)
    plt.ylabel("Total Points", fontsize=21)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "HPvsSP_Itotal%d.png" % Itot_mul[i]))
    plt.close()

    rel_inc_HP.append((activ_SP[activ_HP!=-1].copy() - activ_HP[activ_HP!=-1].copy()) / activ_SP[activ_HP!=-1].copy() * 100)
    rel_inc_HP2.append((activ_SP[activ_HP != -1].copy() - activ_HP2[activ_HP != -1].copy()) / activ_SP[activ_HP != -1].copy() * 100)
    rel_inc_HP3.append((activ_SP[activ_HP != -1].copy() - activ_HP3[activ_HP != -1].copy()) / activ_SP[activ_HP != -1].copy() * 100)
    Isafe.append(np.array(Isafety_lst)[activ_HP!=-1])
    print("#################################################################")
    print("Relative Decrease in Area of HP w.r.t. SP:", rel_inc_HP[i])
    print("Relative Decrease in Area of HP2 w.r.t. SP:", rel_inc_HP2[i])
    print("Relative Decrease in Area of HP3 w.r.t. SP:", rel_inc_HP3[i])
    print("#################################################################")

np.save(os.path.join(base_dir, "HPvsSP_RelInc.npy"), np.array(rel_inc_HP))
np.save(os.path.join(base_dir, "HPvsSP2_RelInc.npy"), np.array(rel_inc_HP2))
np.save(os.path.join(base_dir, "HPvsSP3_RelInc.npy"), np.array(rel_inc_HP3))
np.save(os.path.join(base_dir, "Isafe.npy"), np.array(Isafe))

marker = ['^', "o", 'x']
linestyle = ['dashdot', (0,(3,1,1,1,1,1)), 'dotted']
color = ['blue', 'green', 'darkmagenta']
for i in range(len(Itot_mul)):
    plt.plot(Isafety_lst, rel_inc_HP[i], marker=marker[0], label=r"$p=1$", linestyle=linestyle[0], color=color[0])
    plt.plot(Isafety_lst, rel_inc_HP2[i], marker=marker[1], label=r"$p=2$", linestyle=linestyle[1], color=color[1])
    plt.plot(Isafety_lst, rel_inc_HP3[i], marker=marker[2], label=r"$p=3$", linestyle=linestyle[2], color=color[2])
    ymax = np.max([np.max(rel_inc_HP[i]), np.max(rel_inc_HP2[i]), np.max(rel_inc_HP3[i])])
    plt.hlines(y=ymax, xmin=np.min(Isafety_lst), xmax=np.max(Isafety_lst), linestyle='dashed', color='black', alpha=0.4)
    if ymax > 60:
        plt.text(x=Isafety_lst[0], y=ymax * 0.92, s=str(int(ymax)) + r"$\%$", fontsize=25)
    else:
        plt.text(x=Isafety_lst[0], y=ymax * 0.9, s=str(int(ymax)) + r"$\%$", fontsize=25)
    plt.legend(fontsize=17, ncols=3, loc="upper left")
    plt.xlabel(r"$I_{safe}$ (mA)", fontsize=21)
    plt.ylabel("% Decrease Activation", fontsize=21)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    if ymax > 60:
        plt.ylim(ymax=ymax * 1.15)
    else:
        plt.ylim(ymax=ymax * 1.25)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "HPvsSP_RelInc_Itot%d.png" % (Itot_mul[i])))
    plt.close()

