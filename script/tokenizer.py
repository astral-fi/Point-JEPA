import numpy as np
from dataset import ModelNet40Dataset, concatenate_h5_files, DATASET_PATH
import torch
from torch.utils.data import DataLoader
import torch.nn as nn


def visualize_knn_clusters(points_batches, sampled_points_batches, knn_indices):
    """
    Visualize the k-nearest neighbor clusters for each sampled point in a batch of point clouds.

    Parameters:
    points_batches (numpy.ndarray): Input point cloud of shape (B, N, 3).
    sampled_points_batches (numpy.ndarray): Sampled points of shape (B, M, 3).
    knn_indices (numpy.ndarray): Indices of the k-nearest neighbors for each sampled point, shape (B, M, k).
    """

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    B, M, k = knn_indices.shape

    for b in range(B):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_batches[b][:, 0], points_batches[b][:, 1], points_batches[b][:, 2], c='gray', s=1, label='Original Points')
        
        for m in range(M):
            sampled_point = sampled_points_batches[b][m]
            knn_points = points_batches[b][knn_indices[b][m]]
            ax.scatter(knn_points[:, 0], knn_points[:, 1], knn_points[:, 2], c='red', s=10)
            ax.scatter(sampled_point[0], sampled_point[1], sampled_point[2], c='blue', s=50)

        ax.set_title(f'Batch {b+1} - KNN Clusters')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend()
        plt.show()


data_train, label_train = concatenate_h5_files(DATASET_PATH, 5, 'train')
data_test, label_test = concatenate_h5_files(DATASET_PATH, 2, 'test')

dataset_train = ModelNet40Dataset(data_train, label_train, 1024)
dataset_train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
dataset_test = ModelNet40Dataset(data_test, label_test, 1024)
dataset_test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False)

class PointCloudTokenizer(nn.Module):
    
    def __init__(self, num_samples=64, k_neighbors=32, token_dim=128):
        super().__init__()
        self.num_samples = num_samples
        self.k_neighbors = k_neighbors
        self.token_dim = token_dim

        self.tokens_layer_1 = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(), #Gaussian Error Linear Unit activation function
            nn.Linear(64, 128),
            nn.LayerNorm(128),
        )

        self.tokens_layer_2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, self.token_dim),
            nn.LayerNorm(self.token_dim),
        )

        self.positional_layer_1 = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, self.token_dim),
            nn.LayerNorm(self.token_dim),
        )

    
    def farthest_point_sampling(self, points_batches, num_samples):
        """
        Perform farthest point sampling on a set of points.

        Parameters:
        points_batches (torch.tensor): Input point cloud of shape (B, N, 3).
        num_samples (int): Number of points to sample.

        Returns:
        torch.tensor: Sampled points of shape (B, num_samples, 3).
        """

        B, N, _ = points_batches.shape #Batch size, N number of points, 3 coordinates

        sampled_indices = torch.zeros((B, num_samples), dtype=torch.int64).to(points_batches.device) #Batch size, num_samples

        distances = torch.full((B, N), float('inf')).to(points_batches.device) #Initialize distances to infinity for each point in the batch

        # Randomly select the first point
        sampled_indices[:, 0] = torch.randint(0, N, size=(B,)).to(points_batches.device) #[row_section, column_section] Assign a random index for each batch on the first column of sampled_indices
        
        for i in range(1, num_samples):
            # Update distances to the nearest sampled point
            last_sampled_points = points_batches[torch.arange(B)[:, None], sampled_indices[:, i - 1].reshape(-1, 1)] #Get the last sampled point for each batch
            dist_to_last_sampled = torch.cdist(points_batches, last_sampled_points)
            distances = torch.minimum(distances, dist_to_last_sampled.squeeze(-1))  # Update distances to the nearest sampled point
            
            # Select the farthest point 
            sampled_indices[:, i] = torch.topk(distances, k=1, dim=1, largest=True).indices.squeeze(-1)

        return points_batches[torch.arange(B)[:, None], sampled_indices]
    

    def k_nearest_neighbors(self, points_batches, sampled_points_batches, k):
        """
        Find the k-nearest neighbors for each sampled point in a batch of point clouds.

        Parameters:
        points_batches (torch.tensor): Input point cloud of shape (B, N, 3).
        sampled_points_batches (torch.tensor): Sampled points of shape (B, M, 3).
        k (int): Number of nearest neighbors to find.

        Returns:
        torch.tensor: Indices of the k-nearest neighbors for each sampled point, shape (B, M, k).
        """

        B, N, _ = points_batches.shape
        _, M, _ = sampled_points_batches.shape

        # Compute pairwise distances between sampled points and all points
        distances = torch.cdist(sampled_points_batches, points_batches, p=2)  # Shape: (B, M, N)

        # Get the indices of the k-nearest neighbors
        knn_indices = torch.topk(distances, k=k, dim=-1, largest=False).indices  # Shape: (B, M, k)

        return knn_indices

    def tokenize(self, point_clouds):
        """
        Tokenize a batch of point clouds using farthest point sampling and k-nearest neighbors.

        Parameters:
        point_clouds (torch.Tensor): Input point cloud batch of shape (B, N, 3).

        Returns:
        tuple: Sampled points and their corresponding k-nearest neighbor indices.
        """

        sampled_points = self.farthest_point_sampling(point_clouds, self.num_samples)
        knn_indices = self.k_nearest_neighbors(point_clouds, sampled_points, self.k_neighbors)
        grouped_points = point_clouds[torch.arange(point_clouds.shape[0])[:, None, None], knn_indices]  # Shape: (B, M, k, 3)
        grouped_points = grouped_points - sampled_points.unsqueeze(2)  # Center the k-nearest neighbors around the sampled point

        return sampled_points, grouped_points

    def forward(self, point_clouds):
        """
        Forward pass of the tokenizer.

        Parameters:
        point_clouds (torch.Tensor): Input point cloud batch of shape (B, N, 3).

        Returns:
        torch.Tensor: Tokenized representation of the point clouds.
        """
        
        sampled_points, grouped_points = self.tokenize(point_clouds)
    
        # Convert sampled points to torch tensor
        grouped_points_tensor = grouped_points.float().to(point_clouds.device)  # Shape: (B, M, k, 3)
        sampled_points_tensor = sampled_points.float().to(point_clouds.device)  # Shape: (B, M, 3)

        # Pass through the first layer
        features = self.tokens_layer_1(grouped_points_tensor)  # Shape: (B, M, k, 128)
        group_features = features.max(dim=2).values  # Max pooling over the k-nearest neighbors
        group_features = group_features.unsqueeze(2)

        group_features = group_features.repeat(1, 1, self.k_neighbors, 1)  # Repeat for each neighbor

        point_features = torch.cat([group_features, features], dim=-1)  # Concatenate group and point features
        
        # Pass through the second layer
        features = self.tokens_layer_2(point_features)
        tokens = features.max(dim=2).values  # Max pooling over the k-nearest neighbors
        positional_features = self.positional_layer_1(sampled_points_tensor)  # Shape: (B, M, token_dim)

        return tokens, positional_features



class TokenizerWrapper(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.linear_head = nn.Linear(tokenizer.token_dim, 40)  # Linear layer to combine tokens and positional features

    def forward(self, point_clouds):
        tokens, positional_features = self.tokenizer(point_clouds)
        tokens = tokens + positional_features  # Concatenate tokens and positional features
        tokens = torch.mean(tokens, dim=1)  # Compute the mean across the last dimension
        logits = self.linear_head(tokens)  # Pass through the linear head
        return logits



if __name__ == "__main__":
    tokenizer = PointCloudTokenizer(num_samples=64, k_neighbors=32, token_dim=128)
    tokenizer.to('cuda' if torch.cuda.is_available() else 'cpu')  # Move the model to GPU if available
    model = TokenizerWrapper(tokenizer)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Move the model to GPU if available

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    EPOCHS = 20

    for epoch in range(EPOCHS):
        train_loss = 0.0

        for points, labels in dataset_train_loader:

            points = points.to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')

            logits = model(points)

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss/len(dataset_train_loader):.4f}")

        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0

            for points, labels in dataset_test_loader:
                points = points.to('cuda' if torch.cuda.is_available() else 'cpu')
                labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')

                logits = model(points)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f"Test Accuracy: {accuracy:.2f}%")
            


