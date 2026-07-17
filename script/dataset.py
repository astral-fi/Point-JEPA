import h5py 
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

torch.manual_seed(42)  # For reproducibility
np.random.seed(42)  # For reproducibility

DATASET_PATH = '../modelnet40_ply_hdf5_2048'  # Replace with your HDF5 file path


def inspect_h5_file(file_path):
    """
    Inspects the contents of an HDF5 file and prints its structure.

    Parameters:
    file_path (str): The path to the HDF5 file to inspect.
    """
    with h5py.File(file_path, 'r') as h5_file:
        print(f"Inspecting HDF5 file: {file_path}")
        print("Contents:")
        def print_structure(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")
        h5_file.visititems(print_structure)


def concatenate_h5_files(dataset_path, num_files=5, dataset_type='train'):
    data = []
    label = []
    for i in range(num_files):
        file_path = f"{dataset_path}/ply_data_{dataset_type}{i}.h5"
        with h5py.File(file_path, 'r') as h5_file:
            data.append(h5_file['data'][:])
            label.append(h5_file['label'][:])
    data = np.concatenate(data, axis=0)
    label = np.concatenate(label, axis=0)
    return data, label


class ModelNet40Dataset(Dataset):
    def __init__(self, data, labels, target_points=2048):
        self.data = data
        self.labels = labels
        self.target_points = target_points  # Number of points to sample from each point cloud

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        random_indices = np.random.choice(sample.shape[0], self.target_points, replace=False)
        sample = sample[random_indices]
        sample = sample - sample.mean(axis=0)  # Center the point cloud
        sample = sample / np.linalg.norm(sample, axis=1).max()  # Normalize to unit sphere
        sample = sample.astype(np.float32)
        sample = torch.from_numpy(sample)
        label = self.labels[idx][0].astype(np.int64)
        label = torch.tensor(label)

        return sample, label


def visualize_point_cloud(point_cloud, title="Point Cloud"):
    """
    Visualizes a 3D point cloud using matplotlib.

    Parameters:
    point_cloud (numpy.ndarray): The point cloud data to visualize, shape (N, 3).
    title (str): The title of the plot.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
