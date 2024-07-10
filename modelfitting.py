import os
import numpy as np

def change_working_directory(directory):
    """
    Change the current working directory.

    Parameters:
    directory (str): The path of the new working directory.
    """
    os.chdir(directory)

def load_empirical_matrix(filepath):
    """
    Load the empirical wPLI matrix from a .npy file.

    Parameters:
    filepath (str): The path to the .npy file.

    Returns:
    np.ndarray: The empirical wPLI matrix.
    """
    return np.load(filepath)

def filter_valid_values(matrix1, matrix2):
    """
    Filter out NaN and Inf values from two matrices.

    Parameters:
    matrix1 (np.ndarray): The first matrix.
    matrix2 (np.ndarray): The second matrix.

    Returns:
    tuple: Filtered values of matrix1 and matrix2.
    """
    valid_indices = ~np.isnan(matrix1) & ~np.isinf(matrix1) & ~np.isnan(matrix2) & ~np.isinf(matrix2)
    return matrix1[valid_indices], matrix2[valid_indices]

def compute_correlation(empirical_matrix, simulations_dir):
    """
    Compute correlation between the empirical wPLI matrix and simulated wPLI matrices.

    Parameters:
    empirical_matrix (np.ndarray): The empirical wPLI matrix.
    simulations_dir (str): Directory containing the simulated wPLI matrices.

    Returns:
    list: List of tuples containing simulation filenames and their correlation values.
    """
    correlation_results = []
    simulations         = [f for f in os.listdir(simulations_dir) if f.endswith('_aec.npy')]

    for simulation in simulations:
        file_path = os.path.join(simulations_dir, simulation)

        simulated_matrix = np.load(file_path)

        if empirical_matrix.shape != simulated_matrix.shape:
            print(f"Shape mismatch for {simulation}: {empirical_matrix.shape} vs {simulated_matrix.shape}")
            continue

        empirical_filtered, simulated_filtered = filter_valid_values(empirical_matrix, simulated_matrix)

        if empirical_filtered.size == 0 or simulated_filtered.size == 0:
            print(f"Filtered matrices are empty for {simulation}")
            continue

        correlation_value = np.corrcoef(empirical_filtered, simulated_filtered)[0, 1]
        correlation_results.append((simulation, correlation_value))
        print(f"Processed {simulation}: Correlation = {correlation_value}")

    return correlation_results

def main():
    # Change working directory
    change_working_directory('/vol/specs04/vivek/perturb_oscillator/')

    # Load the empirical wPLI matrix
    empirical_matrix = load_empirical_matrix('emperical/average_AEC_HCP.npy')

    # Directory containing the simulated wPLI matrices
    simulations_dir = 'aec_simulations/'

    # Compute correlation results
    correlation_results = compute_correlation(empirical_matrix, simulations_dir)

    if not correlation_results:
        print("No files were processed. Please check the file paths and naming conventions.")
        return

    # Convert correlation results to a structured array for easy processing
    dtype               = [('filename', 'U100'), ('correlation', float)]
    correlation_results = np.array(correlation_results, dtype=dtype)

    # Find the file with the maximum correlation
    max_corr_index = np.argmax(correlation_results['correlation'])
    max_corr_file  = correlation_results[max_corr_index]['filename']
    max_corr_value = correlation_results[max_corr_index]['correlation']

    # Print the file and correlation value with the highest correlation
    print(f"Maximum correlation ({max_corr_value}) found in file {max_corr_file}.")

if __name__ == "__main__":
    main()
