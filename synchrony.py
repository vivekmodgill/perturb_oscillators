import os
import numpy as np
import scipy
import scipy.signal as sig
import scipy.stats as stat
from pymatreader import read_mat

# Change working directory
os.chdir('#PATH OF MAIN DIRECTORY')

mat = read_mat('SC_90aal_32HCP.mat')
# Structural connectivity
connect = mat['mat']
connect[connect < 10] = 0 # Removes negligible connections with less than 10 fibers
np.fill_diagonal(connect, 0) # Ensures there are no self connections
connect = connect / np.mean(connect)
con_mat = connect
# Tract lengths
dist_mat = mat['mat_D']

con_mat_SL = []
gc = []
C_vec =  np.logspace(np.log10(0.1), np.log10(50), 40)
for i in C_vec:
    x = con_mat * i
    gc.append(i)
    con_mat_SL.append(x)
    del x

speedSL = []
md = []
md_vec = np.linspace(0, 20, 21)
for j in md_vec:
    md.append(j)
    if j > 0:
        y = np.mean(dist_mat[con_mat>0])/j
    else:
        y = -1000
    speedSL.append(y)
    del y

meta_mat = np.nan * np.zeros((len(gc), len(md)))
sync_mat = np.nan * np.zeros((len(gc), len(md)))

for k, m in enumerate(gc):
    for l, n in enumerate(md):
        file_path = f'simulations/signal_C{m:.2f}_md{n:.0f}_dt0.1.npy'
        signal = np.load(file_path)

        # Bandpass the signal
        #fs = 1e3
        #b, a = sig.butter(4, [8, 13], btype='band', output='ba', fs=fs)
        #signal = sig.filtfilt(b, a, signal, axis=-1)

        # Compute Hilbert transform
        hilbert_sigs = sig.hilbert(signal[0:90, :].real, axis=-1)
        hilbert_sigs = hilbert_sigs / np.abs(hilbert_sigs)

        # Compute Kuramoto Order Parameter
        KOP = np.abs(np.nanmean(hilbert_sigs, axis=0))

        # Check if KOP is not empty (contains NaN values)
        if not np.isnan(KOP).all():
            # Update sync and meta matrices
            sync_mat[k, l] = np.nanmean(KOP)
            meta_mat[k, l] = np.nanstd(KOP)

np.save('global_metastability_SL.npy', meta_mat)
np.save('global_synchrony_SL.npy', sync_mat)
