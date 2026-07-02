import numpy as np
import matplotlib.pyplot as plt
import os
from plot import vorticity_plots


def print_config(obj, indent=0):
    """Recursively prints all attributes of a class or object."""
    if not hasattr(obj, "__dict__") and not isinstance(obj, type):  # If it's a simple value, print it
        print(" " * indent + str(obj))
        return

    for attr_name in dir(obj):
        if attr_name.startswith("__"):  # Skip special attributes
            continue

        attr_value = getattr(obj, attr_name)

        if isinstance(attr_value, type):  # If it's a class, recurse into it
            print(" " * indent + f"{attr_name}:")
            print_config(attr_value, indent + 4)
        elif not callable(attr_value):  # Print regular attributes
            print(" " * indent + f"{attr_name} = {attr_value}")

def save_file(solution_field, run_number, time_params):
    base_dir = '/gdata/projects/ml_scope/Turbulence/QG_V0001/Results'
    save_dir = os.path.join(base_dir, f'Run{run_number:05d}')
    os.makedirs(save_dir, exist_ok=True)
    
    file_name = f'fields_Run{run_number:05d}.npy'
    file_path = os.path.join(save_dir, file_name)
    
    # Save np file
    np.save(file_path, solution_field.cpu().numpy()) 
    
    
    ### Move all code files
    # Move config file to results folder
    source_directory = f"/gdata/projects/ml_scope/Turbulence/QG_V0001/Src"
    # Define the destination directory where you want to move config.json
    destination_directory = f"/gdata/projects/ml_scope/Turbulence/QG_V0001/Results/Run{run_number:05d}/Code"
    os.makedirs(destination_directory, exist_ok=True)
    # List of folders to ignore
    folders_to_ignore = [
        f"/gdata/projects/ml_scope/Turbulence/QG_V0001/Src/Config"]
    # Move the configuration file to the destination directory
    for root, dirs, files in os.walk(source_directory):
        # Determine the relative path and the destination path
        relative_path = os.path.relpath(root, source_directory)
        destination_path = os.path.join(destination_directory, relative_path)
        # Check if the current directory should be ignored
        if any(ignored_dir in root for ignored_dir in folders_to_ignore):
            continue
        # Create the directory structure in the destination
        os.makedirs(destination_path, exist_ok=True)
        # Copy the files
        for file in files:
            source_file = os.path.join(root, file)
            destination_file = os.path.join(destination_path, file)
            shutil.copy(source_file, destination_file)
    
    
    base_dir = '/gdata/projects/ml_scope/Turbulence/QG_V0001/Results/'
    save_dir = os.path.join(base_dir, f'Run{run_number:05d}', 'Plots')
    os.makedirs(save_dir, exist_ok=True)
    
    for timestep in range(solution_field.shape[2]):
        fig, ax = vorticity_plots(solution_field, timestep, time_params)

        # Save the plot as an image 
        plot_file_name = f'vorticity_Run{run_number:05d}_t_{timestep*time_params.save_int:06d}.png'
        plot_file_path = os.path.join(save_dir, plot_file_name)
        fig.savefig(plot_file_path,bbox_inches='tight',dpi=300)  # Save the plot as an image
        plt.close(fig)  # Close the figure to free memory
    
    
    
