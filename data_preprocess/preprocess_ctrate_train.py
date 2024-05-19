import os
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from multiprocessing import Pool
from tqdm import tqdm


def read_nii_files(directory):
    """
    Retrieve the list of NIfTI files in the specified directory.
    Args:
    directory (str): Path to the directory containing NIfTI files.
    Returns:
    list: List of tuples containing the folder name and file name for each NIfTI file.
    """
    folders = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]
    nii_files = []
    for folder in folders:
        folder_path = os.path.join(directory, folder)
        for file in os.listdir(folder_path):
            if file.endswith('.nii.gz'):
                nii_files.append((folder, file))
    return nii_files


def read_nii_data(file_path):
    """
    Read NIfTI file data.
    Args:
    file_path (str): Path to the NIfTI file.
    Returns:
    np.ndarray: NIfTI file data.
    """
    try:
        nii_img = nib.load(file_path)
        nii_data = nii_img.get_fdata()
        return nii_data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.

    Args:
    array (torch.Tensor): Input array to be resized.
    current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
    target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
    np.ndarray: Resized array.
    """
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    # Resize the array
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array


def process_file(file_info):
    """
    Process a single NIfTI file.

    Args:
    file_info (tuple): Tuple containing folder name and file name.

    Returns:
    None
    """
    folder, file_name = file_info
    file_path = os.path.join(split_to_preprocess, folder, file_name)

    img_data = read_nii_data(file_path)
    if img_data is None:
        print(f"Read {file_path} unsuccessful. Passing")
        return

    # TODO: Read the corresponding CSV file and extract the necessary information
    slope = 1.0
    intercept = 0.0
    xy_spacing = 1.0
    z_spacing = 1.0
    # Define the target spacing values
    target_x_spacing = 0.75
    target_y_spacing = 0.75
    target_z_spacing = 1.5

    current = (z_spacing, xy_spacing, xy_spacing)
    target = (target_z_spacing, target_x_spacing, target_y_spacing)

    img_data = slope * img_data + intercept
    hu_min, hu_max = -1000, 1000
    img_data = np.clip(img_data, hu_min, hu_max)
    img_data = ((img_data / 1000)).astype(np.float32)

    img_data = img_data.transpose(2, 0, 1)
    tensor = torch.tensor(img_data)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    resized_array = resize_array(tensor, current, target)
    resized_array = resized_array[0][0]

    # Create the corresponding npz folder structure
    save_folder = '/home/dxw/Desktop/common_datasets/CTRG-Chest-548K-3D-npz/'
    folder_path_new = os.path.join(save_folder, folder)
    os.makedirs(folder_path_new, exist_ok=True)

    npz_file_name = file_name.replace('.nii.gz', '.npz')
    save_path = os.path.join(folder_path_new, npz_file_name)
    np.savez(save_path, resized_array)


# Example usage:
if __name__ == "__main__":
    split_to_preprocess = '/home/dxw/Desktop/common_datasets/CTRG-Chest-548K-3D/'
    # Get the list of NIfTI files
    nii_files = read_nii_files(split_to_preprocess)
    num_workers = 20  # Number of worker processes
    # Process files using multiprocessing with tqdm progress bar
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap(process_file, nii_files), total=len(nii_files)))