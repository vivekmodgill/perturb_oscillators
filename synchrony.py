import os
import numpy as np
import scipy.signal as sig
from pymatreader import read_mat

def load_structural_data(filepath):
    """
    Load structural connectivity and tract length data from a .mat file.

    Parameters:
    filepath (str): Path to the .mat file.

    Returns:
    tuple: Structural connectivity matrix and tract length matrix.
    """
    mat      = read_mat(filepath)
    connect  = mat['mat']
    dist_mat = mat['mat_D']
    return connect, dist_mat

def preprocess_connectivity(connect):
    """
    Preprocess the structural connectivity matrix.

    Parameters:
    connect (np.ndarray): Structural connectivity matrix.

    Returns:
    np.ndarray: Preprocessed structural connectivity matrix.
    """
    connect[connect < 10] = 0  # Remove negligible connections
    np.fill_diagonal(connect, 0)  # Ensure no self-connections
    connect = connect / np.mean(connect)
    return connect

def generate_connectivity_variants(connect, C_vec):
    """
    Generate scaled connectivity matrices.

    Parameters:
    connect (np.ndarray): Structural connectivity matrix.
    C_vec (np.ndarray): Vector of scaling factors.

    Returns:
    list: List of scaled connectivity matrices.
    """
    return [connect * C for C in C_vec]

def calculate_speeds(dist_mat, connect, md_vec):
    """
    Calculate propagation speeds based on tract lengths.

    Parameters:
    dist_mat (np.ndarray): Tract length matrix.
    connect (np.ndarray): Structural connectivity matrix.
    md_vec (np.ndarray): Vector of mean delays.

    Returns:
    list: List of propagation speeds.
    """
    return [np.mean(dist_mat[connect > 0]) / md if md > 0 else -1000 for md in md_vec]

def process_signals(signal, fs=1000):
    """
    Process signals by bandpass filtering and computing the Hilbert transform.

    Parameters:
    signal (np.ndarray): Signal array.
    fs (int): Sampling frequency.

    Returns:
    np.ndarray: Normalized Hilbert transformed signals.
    """
    # Bandpass the signal (uncomment if needed)
    # b, a = sig.butter(4, [8, 13], btype='band', fs=fs)
    # signal = sig.filtfilt(b, a, signal, axis=-1)
    
    hilbert_sigs = sig.hilbert(signal[0:90, :].real, axis=-1)
    return hilbert_sigs / np.abs(hilbert_sigs)

def compute_kuramoto_order(hilbert_sigs):
    """
    Compute the Kuramoto Order Parameter.

    Parameters:
    hilbert_sigs (np.ndarray): Hilbert transformed signals.

    Returns:
    np.ndarray: Kuramoto Order Parameter.
    """
    return np.abs(np.nanmean(hilbert_sigs, axis=0))

def main():
    # Directory settings
    main_directory = '#PATH OF MAIN DIRECTORY'
    os.chdir(main_directory)
    input_filepath = 'SC_90aal_32HCP.mat'
    simulation_dir = os.path.join(main_directory, 'simulations')
    
    # Load structural data
    connect, dist_mat = load_structural_data(input_filepath)
    connect           = preprocess_connectivity(connect)
    
    # Generate connectivity variants and calculate speeds
    C_vec      = np.logspace(np.log10(0.1), np.log10(50), 40)
    con_mat_SL = generate_connectivity_variants(connect, C_vec)
    
    md_vec  = np.linspace(0, 20, 21)
    speedSL = calculate_speeds(dist_mat, connect, md_vec)
    
    # Initialize matrices for results
    meta_mat = np.full((len(C_vec), len(md_vec)), np.nan)
    sync_mat = np.full((len(C_vec), len(md_vec)), np.nan)
    
    # Iterate over connectivity and mean delay values
    for k, C_val in enumerate(C_vec):
        for l, md_val in enumerate(md_vec):
            file_path = f'simulations/signal_C{C_val:.2f}_md{md_val:.0f}_dt0.1.npy'
            signal    = np.load(file_path)
            
            # Process the signal
            hilbert_sigs = process_signals(signal)
            KOP          = compute_kuramoto_order(hilbert_sigs)
            
            if not np.isnan(KOP).all():
                sync_mat[k, l] = np.nanmean(KOP)
                meta_mat[k, l] = np.nanstd(KOP)
    
    # Save results
    np.save('global_metastability_SL.npy', meta_mat)
    np.save('global_synchrony_SL.npy', sync_mat)
    print("Results saved successfully.")

if __name__ == "__main__":
    main()
