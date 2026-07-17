import h5py 
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
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

        sample = sample - sample.mean(axis=0)  # Center the point cloud
        sample = sample / np.linalg.norm(sample, axis=1).max()  # Normalize to unit sphere

        random_indices = np.random.choice(sample.shape[0], self.target_points, replace=False)
        sample = sample[random_indices]

        if self.augment:
            sample = jitter_point_cloud(sample)
            sample = rotate_point_cloud(sample)
            sample = scale_point_cloud(sample)
            sample = sample - sample.mean(axis=0)  # Center the point cloud
            
        sample = sample.astype(np.float32)
        sample = torch.from_numpy(sample)
        label = self.labels[idx][0].astype(np.int64)
        label = torch.tensor(label)

        return sample, label


import plotly.graph_objects as go

def visualize_point_cloud_plotly(point_cloud, title="ModelNet40 Object"):
    # 1. Automatically handle PyTorch tensors and batch dimensions
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.squeeze().cpu().numpy()
    
    # Ensure shape is (N, 3)
    if len(point_cloud.shape) != 2 or point_cloud.shape[1] != 3:
        raise ValueError(f"Expected shape (N, 3), but got {point_cloud.shape}")

    # Extract coordinates
    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]

    # 2. Build the plot with slightly larger, soft-edged markers
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=3.5,                  # Slightly larger to fill gaps
            color=z,                  # Color-coded by depth (Z-axis)
            colorscale='Viridis',
            opacity=0.85              # Soft edges blend together better
        )
    )])

    # 3. Force 1:1:1 scale ratios and clean up background clutter
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode='data',        # CRITICAL: Prevents stretching/distortion
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        template="plotly_dark"        # Dark mode makes the neon point cloud pop beautifully!
    )
    fig.show()

if __name__ == "__main__":
    data_train, label_train = concatenate_h5_files(DATASET_PATH, 5, 'train')
    data_test, label_test = concatenate_h5_files(DATASET_PATH, 2, 'test')

    dataset_train = ModelNet40Dataset(data_train, label_train, 1024)
    dataset_train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
    dataset_test = ModelNet40Dataset(data_test, label_test, 1024)
    dataset_test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False)

    for points, labels in dataset_test_loader:

        for i in range(min(5, points.size(0))):  # Visualize up to 5 point clouds
            visualize_point_cloud_plotly(points[i].numpy(), title=f"Label: {labels[i].item()}")
            rotated_points = rotate_point_cloud(points[i].numpy())
            jittered_points = jitter_point_cloud(points[i].numpy())
            scaled_points = scale_point_cloud(points[i].numpy())
            visualize_point_cloud_plotly(scaled_points, title=f"Scaled Point Cloud - Label: {labels[i].item()}")


        break  # Just inspect the first batch