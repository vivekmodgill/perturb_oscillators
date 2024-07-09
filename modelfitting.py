import os
import numpy as np

# Change working directory
os.chdir('/vol/specs04/vivek/perturb_oscillator/')

# Load the empirical wPLI matrix
fc_matrix = np.load('emperical/average_AEC_HCP.npy')

# Directory containing the simulated wPLI matrices
simulations_dir = 'aec_simulations/'

# Get a list of all wPLI files in the directory
simulations = [f for f in os.listdir(simulations_dir) if f.endswith('_aec.npy')]

# Initialize an array to store correlation results
correlation_results = []

# Function to filter out NaN and Inf values from a matrix
def filter_valid_values(matrix1, matrix2):
    valid_indices = ~np.isnan(matrix1) & ~np.isinf(matrix1) & ~np.isnan(matrix2) & ~np.isinf(matrix2)
    return matrix1[valid_indices], matrix2[valid_indices]

# Iterate over all wPLI files and compute correlation with the empirical wPLI matrix
for simulation in simulations:
    file_path = os.path.join(simulations_dir, simulation)

    # Load the simulated wPLI matrix
    simulated = np.load(file_path)

    # Check if matrices have the same shape
    if fc_matrix.shape != simulated.shape:
        print(f"Shape mismatch for {simulation}: {fc_matrix.shape} vs {simulated.shape}")
        continue

    # Filter out NaN and Inf values from both matrices
    fc_matrix_filtered, simulated_filtered = filter_valid_values(fc_matrix, simulated)

    # Check for empty filtered matrices
    if fc_matrix_filtered.size == 0 or simulated_filtered.size == 0:
        print(f"Filtered matrices are empty for {simulation}")
        continue

    # Compute the correlation between simulated wPLI and empirical wPLI
    correlation_value = np.corrcoef(fc_matrix_filtered, simulated_filtered)[0, 1]
    correlation_results.append((simulation, correlation_value))
    print(f"Processed {simulation}: Correlation = {correlation_value}")

# Check if any files were processed
if not correlation_results:
    print("No files were processed. Please check the file paths and naming conventions.")
else:
    # Convert correlation results to a structured array for easy processing
    dtype = [('filename', 'U100'), ('correlation', float)]
    correlation_results = np.array(correlation_results, dtype=dtype)

    # Find the file with the maximum correlation
    max_corr_index = np.argmax(correlation_results['correlation'])
    max_corr_file = correlation_results[max_corr_index]['filename']
    max_corr_value = correlation_results[max_corr_index]['correlation']

    # Print the file and correlation value with the highest correlation
    print(f"Maximum correlation ({max_corr_value}) found in file {max_corr_file}.")
