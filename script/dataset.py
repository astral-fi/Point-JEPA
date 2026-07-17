import h5py 
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

torch.manual_seed(42)  # For reproducibility

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

def rotate_point_cloud(points, axis='y'):
    theta = np.random.uniform(0, 2 * np.pi)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    if axis == 'y':
        rotation_matrix = np.array([
            [cos_t,  0, sin_t],
            [0,      1, 0    ],
            [-sin_t, 0, cos_t],
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [cos_t, -sin_t, 0],
            [sin_t,  cos_t, 0],
            [0,      0,     1],
        ])

    return points @ rotation_matrix.T

def jitter_point_cloud(points, sigma=0.01, clip=0.02):
    noise = np.clip(np.random.normal(0, sigma, points.shape), -clip, clip)
    return points + noise

def scale_point_cloud(points, scale_low=0.8, scale_high=1.2):
    scale = np.random.uniform(scale_low, scale_high)
    return points * scale




class ModelNet40Dataset(Dataset):
    def __init__(self, data, labels, target_points=2048, augment =True):
        self.data = data
        self.labels = labels
        self.target_points = target_points  # Number of points to sample from each point cloud
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.augment:
            sample = rotate_point_cloud(sample)
            sample = jitter_point_cloud(sample)
            sample = scale_point_cloud(sample)
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
