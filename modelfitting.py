import os
import numpy as np

# Change working directory
os.chdir('/vol/specs04/vivek/perturb_oscillator/')

# Load the empirical wPLI matrix
fc_matrix = np.load('emperical/average_AEC_HCP.npy')

# Directory containing the simulated wPLI matrices
simulations_dir = 'aec_simulations/'

# Get a list of all wPLI files in the directory
wpli_files = [f for f in os.listdir(simulations_dir) if f.endswith('_aec.npy')]

# Initialize an array to store correlation results
correlation_results = []

# Function to filter out NaN and Inf values from a matrix
def filter_valid_values(matrix):
    valid_indices = ~np.isnan(matrix) & ~np.isinf(matrix)
    return matrix[valid_indices]

# Iterate over all wPLI files and compute correlation with the empirical wPLI matrix
for wpli_file in wpli_files:
    file_path = os.path.join(simulations_dir, wpli_file)

    # Load the simulated wPLI matrix
    simulated_wpli = np.load(file_path)

    # Filter out NaN and Inf values from both matrices
    fc_matrix_filtered = filter_valid_values(fc_matrix)
    simulated_wpli_filtered = filter_valid_values(simulated_wpli)

    # Check for empty filtered matrices
    if fc_matrix_filtered.size == 0 or simulated_wpli_filtered.size == 0:
        print(f"Filtered matrices are empty for {wpli_file}")
        continue

    # Check if filtered matrices have the same shape
    if fc_matrix_filtered.shape != simulated_wpli_filtered.shape:
        print(f"Shape mismatch for {wpli_file}: {fc_matrix_filtered.shape} vs {simulated_wpli_filtered.shape}")
        continue

    # Compute the correlation between simulated wPLI and empirical wPLI
    correlation_value = np.corrcoef(fc_matrix_filtered, simulated_wpli_filtered)[0, 1]
    correlation_results.append((wpli_file, correlation_value))
    print(f"Processed {wpli_file}: Correlation = {correlation_value}")

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
