Code for running computational studies in sea of neurons model.

For running the code: 

1. Unzip the cells.zip and mechansims.zip files.
2. Run the following commmand in the terminal: sh init_setup_run_once.sh
3. Note that the code requires significant RAM memory to generate. You can test it by reducing the size of the forward model. Just reduce the number of sampling in r_lst variable across the scripts create_init_forward_model.py, HingePlace_sim.py, HingePlace_L1L1_equivalence.py, SparsePlace_DCM_equivalence.py, and HingePlace_best_p.py from 40 to a smaller number until which your RAM can store the model. NOte that this might reduce the accuracy of your simulations. Reducing num_cores would also help in reducing parallelizations.
5. The main script is the HingePlace_sim.py which contains the simulation code for running the results of Sec 5.2 in the manuscript.
6. You need to run the init_opt_dir_HP.py once before you run the HingePlace_best_p.py or HingePlace_sim.py to calculate the optimal direction and the optimal threshold for neural stimulation of the neuron model.
7. run_hp_sim.sh provides the required arguments to run the different simulation studies of Sec. 5.2. Note that the tuning of Etol is computationally expensive, and we use an ad-hoc method to tune the Etol hyper-parameter as described in the manuscript. Consequently, tuning of Etol requires manual attention and the scripts in run_hp_sim.sh would have to be run again accordingly.
8. HingePlace_L1L1_equivalence.py and SparsePlace_DCM_equivalence.py provide the equivalence results b/w HingePlace and L1L1, and DCM and LCMV-E, respectively, provided in the Appendix. HingePlace_best_p.py can be used to see the performance of the HingePlace algorithm across different values of p. Rest of the scripts are the back-end used to simulate the spherical head model and the neuron models.

Common Issues: 
1. pyshtools is known to cause issues in Windows.
2. To run the simulations you need the NEURON software. Installing the NEURON software can be tricky. Please refer to their setup guide: https://www.neuron.yale.edu/neuron/download, in case init_setup_run_once.sh is not able to install neuron using pip
3. The code uses the ray library as a parallelization tool to speed up computation, adjust the NUM_CORES parameter according to the parallelization available. If the NUM_CORES is too big, the code will throw an out-of-memory error. The default is set to either 30 or 50 cores.
4. Note that this code uses modified version of the files provided by Aberra et al., so the code will not run with the default files from Aberra et al. The modification are in the xtra mechanism to simulate TI stimulation and the cellChooser.hoc to add L23 Basket Cells.
5.  If you directly run the results for computational study 2 or 3, then the code will throw an error as the simulation initially requires creation of a save_state for running neuron simulation for both PV and Pyr neurons (which are explicitly created when running pointElec_sim.py). Just run the script twice which would result in creation of the save-states after which the scripts should run fine.

