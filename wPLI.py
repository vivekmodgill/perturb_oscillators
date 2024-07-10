import os
import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs

def process_simulation_file(input_filepath, output_directory, n_channels=90, sampling_freq=1000, method='wpli', fmin=0.1, fmax=40, mode='multitaper'):
    """
    Process a single simulation file to compute the wPLI connectivity matrix.

    Parameters:
    input_filepath (str): Path to the input .npy file containing the time series data.
    output_directory (str): Directory to save the computed connectivity matrix.
    n_channels (int): Number of channels.
    sampling_freq (int): Sampling frequency in Hz.
    method (str): Connectivity estimation method.
    fmin (float): Minimum frequency of interest.
    fmax (float): Maximum frequency of interest.
    mode (str): Spectrum estimation mode.

    Returns:
    str: Path to the saved connectivity matrix.
    """
    # Load the source time series data
    src = np.load(input_filepath)
    
    # Ensure the data is real
    if np.iscomplexobj(src):
        src = src.real

    # Create MNE info object and RawArray
    info          = mne.create_info(n_channels, sfreq=sampling_freq)
    simulated_raw = mne.io.RawArray(src, info)

    # Create epochs
    events = mne.make_fixed_length_events(simulated_raw, duration=5.)
    epochs = mne.Epochs(simulated_raw, events=events, tmin=0, tmax=5.0, baseline=None, preload=True)

    # Compute the spectral connectivity
    conn      = spectral_connectivity_epochs(epochs, method=method, sfreq=sampling_freq, mode=mode, fmin=fmin, fmax=fmax)
    conn_data = conn.get_data(output="dense")[:, :, 0]

    # Generate output filepath and save the connectivity matrix
    output_filename = os.path.splitext(os.path.basename(input_filepath))[0] + '_wPLI.npy'
    output_path     = os.path.join(output_directory, output_filename)
    np.save(output_path, conn_data)

    return output_path

def main():
    # Directories for input simulations and output connectivity matrices
    input_directory  = ''
    output_directory = ''

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.npy'):
            input_filepath = os.path.join(input_directory, filename)

            # Process the file and compute the connectivity matrix
            output_path = process_simulation_file(input_filepath, output_directory)
            print(f"Connectivity matrix saved to {output_path}")

if __name__ == "__main__":
    main()
