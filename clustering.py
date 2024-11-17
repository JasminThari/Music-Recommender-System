import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

hex_color_list = ["#000004", "#1b0c41", "#4a0c6b", "#781c6d", "#a52c60", "#cf4446", "#ed6925", "#fb9b06", "#f7d13d", "#fcffa4"]



def run_pca(data, n_components):
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(data)
    pca_transformed = pca.fit_transform(data)
    return pca, pca_transformed


def determine_number_of_clusters_for_KMeans(data, max_clusters=10, random_state=42):
    """
    Evaluates KMeans clustering using WCSS, Silhouette Score, and Davies-Bouldin Index for different number of clusters to determine the optimal number.

    Parameters:
        data (pd.DataFrame or np.ndarray): The input data for clustering.
        max_clusters (int): Maximum number of clusters to evaluate.
        random_state (int): Random state for reproducibility.

    Returns:
        dict: A dictionary containing:
              - 'wcss': Within-cluster sum of squares for each cluster size.
              - 'silhouette_scores': Silhouette scores for each cluster size (2 to max_clusters).
              - 'davies_bouldin_indices': Davies-Bouldin indices for each cluster size (2 to max_clusters).
    """
    wcss = []
    silhouette_scores = []
    davies_bouldin_indices = []

    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=random_state)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

        if i > 1:  # Silhouette and Davies-Bouldin require at least 2 clusters
            cluster_labels = kmeans.labels_
            silhouette_scores.append(silhouette_score(data, cluster_labels))
            davies_bouldin_indices.append(davies_bouldin_score(data, cluster_labels))

    return {
        "wcss": wcss,
        "silhouette_scores": silhouette_scores,
        "davies_bouldin_indices": davies_bouldin_indices
    }


class SongDataset(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data).float()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # Using Sigmoid as input features are scaled between 0 and 1
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, weight_decay):
    """
    Train a PyTorch model with given parameters.

    Parameters:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        num_epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        save_path (str, optional): Path to save the model checkpoints.

    Returns:
        dict: A dictionary containing training and validation losses per epoch.
    """
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Store losses for visualization
    loss_history = {'train': [], 'val': []}

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        # Training phase
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as t:
            for batch in t:
                batch = batch.to(device)

                # Forward pass
                outputs = model(batch)
                loss = criterion(outputs, batch)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch.size(0)

                # Update tqdm progress bar
                t.set_postfix(loss=loss.item())

        train_loss /= len(train_loader.dataset)
        loss_history['train'].append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch)
                loss = criterion(outputs, batch)
                val_loss += loss.item() * batch.size(0)

        val_loss /= len(val_loader.dataset)
        loss_history['val'].append(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the model checkpoint if save_path is provided
    torch.save(model.state_dict(), f"models/model_AE.pth")

    return loss_history, model
