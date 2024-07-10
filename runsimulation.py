import os
import numpy as np
from StuartLandauModel import simulate_SL
from numba import njit
import concurrent.futures
from pymatreader import read_mat

# Change working directory
os.chdir('') #Path to main directory

# Load Connectivity Matrices
mat = read_mat('SC_90aal_32HCP.mat')
# Structural connectivity
connect = mat['mat']
connect[connect < 10] = 0 # Removes negligible connections with less than 10 fibers
np.fill_diagonal(connect, 0) # Ensures there are no self connections
connect = connect / np.mean(connect)
con_mat = connect
# Tract lengths
dist_mat = mat['mat_D'] 

# Ensure the directory exists
os.makedirs('simulations', exist_ok=True)

# Define a function to run a single simulation
def run_single_simulation(dist_mat, con_mat_SL, speed, gc_val, md_val):
    sm_ts = simulate_SL(dist_mat, con_mat_SL, 70000, 60000, 0.1, 1, -5, speed)
    filename = 'simulations/signal_C{:.2f}_md{:.0f}_dt0.1.npy'.format(gc_val, md_val)
    np.save(filename, sm_ts)

# Structural Connectivity
gc = np.logspace(np.log10(0.1), np.log10(50), 40)
con_mat_SL_list = [con_mat * i for i in gc]

# Conduction Speed
md = np.linspace(0, 20, 21)
speedSL = [np.mean(dist_mat[con_mat > 0]) / j if j > 0 else -1000 for j in md]

# Run all simulations using concurrent.futures for parallel execution
def run_all_simulations(dist_mat, con_mat_SL_list, speedSL, gc, md):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for i, con_mat_SL in enumerate(con_mat_SL_list):
            for j, speed in enumerate(speedSL):
                futures.append(executor.submit(run_single_simulation, dist_mat, con_mat_SL, speed, gc[i], md[j]))
        concurrent.futures.wait(futures)

# Run simulations
run_all_simulations(dist_mat, con_mat_SL_list, speedSL, gc, md)
