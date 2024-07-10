import os
import numpy as np
from scipy import signal
import mne
from mne_connectivity import envelope_correlation

def bandpass_filter(data, freq_range, sampling_freq, order=2):
    """
    Apply a bandpass filter to the data.

    Parameters:
    data (np.ndarray): The data to filter.
    freq_range (list): The frequency range for the bandpass filter.
    sampling_freq (float): The sampling frequency of the data.
    order (int): The order of the filter.

    Returns:
    np.ndarray: The filtered data.
    """
    b, a = signal.butter(order, freq_range, btype='band', output='ba', fs=sampling_freq)
    return signal.filtfilt(b, a, data, axis=-1)

def process_file(file_path, output_directory, n_channels, sampling_freq, freq_range):
    """
    Process a single file to compute the spectral connectivity and save the result.

    Parameters:
    file_path (str): The path to the input file.
    output_directory (str): The directory where the output file will be saved.
    n_channels (int): The number of channels in the data.
    sampling_freq (float): The sampling frequency of the data.
    freq_range (list): The frequency range for the bandpass filter.
    """
    # Load the source time series data
    src = np.load(file_path)

    # Apply bandpass filter
    src_ts = bandpass_filter(src, freq_range, sampling_freq)

    # Create MNE info object and RawArray
    info          = mne.create_info(n_channels, sfreq=sampling_freq)
    simulated_raw = mne.io.RawArray(src_ts, info)

    # Create epochs
    events = mne.make_fixed_length_events(simulated_raw, duration=5.0)
    epochs = mne.Epochs(simulated_raw, events=events, tmin=0, tmax=5.0, baseline=None, preload=True)

    # Compute the spectral connectivity
    conn      = envelope_correlation(epochs, orthogonalize='pairwise')
    corr      = conn.combine()
    conn_data = corr.get_data(output="dense")[:, :, 0]

    # Save the connectivity matrix
    output_filename = os.path.splitext(os.path.basename(file_path))[0] + '_aec.npy'
    output_path     = os.path.join(output_directory, output_filename)
    np.save(output_path, conn_data)

    print(f"Connectivity matrix saved to {output_path}")

def main(input_directory, output_directory, n_channels, sampling_freq, freq_range):
    """
    Main function to process all .npy files in the specified directory.

    Parameters:
    input_directory (str): The directory containing the input .npy files.
    output_directory (str): The directory where the output files will be saved.
    n_channels (int): The number of channels in the data.
    sampling_freq (float): The sampling frequency of the data.
    freq_range (list): The frequency range for the bandpass filter.
    """
    for filename in os.listdir(input_directory):
        if filename.endswith('.npy'):
            file_path = os.path.join(input_directory, filename)
            process_file(file_path, output_directory, n_channels, sampling_freq, freq_range)

# Parameters
input_directory  = '' #Path to Input Directory
output_directory = '' #Path to Output Directory 
n_channels       = 90
sampling_freq    = 1000  # in Hertz
freq_range       = [8, 13]

if __name__ == "__main__":
    main(input_directory, output_directory, n_channels, sampling_freq, freq_range)
