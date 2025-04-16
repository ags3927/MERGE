import os
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp

from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import KMeans
from torch.utils.data import Dataset
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import from_scipy_sparse_matrix

### GRAPH DATASET ###
class GraphDataset(Dataset):
    def __init__(self, adj, labels, patch_embeddings, data, config):
        self.slide_indices = []
        self.edge_indices = []
        self.labels = []
        self.patch_embeddings = []
        
        start = 0
        for idx, _ in enumerate(data['slides']):
            spotnum = data['spotnum'][idx]
            
            slide_edge_indices = from_scipy_sparse_matrix(sp.coo_matrix(adj[idx]))[0].to(config['device'])
            slide_labels = labels[start:start+spotnum]
            slide_embeddings = patch_embeddings[start:start+spotnum]
            
            self.slide_indices.append(idx)
            self.edge_indices.append(slide_edge_indices)
            self.labels.append(slide_labels)
            self.patch_embeddings.append(slide_embeddings)
            start += spotnum
            
    def __len__(self):
        return len(self.edge_indices)
    
    def __getitem__(self, idx):
        return self.slide_indices[idx], self.edge_indices[idx], self.labels[idx], self.patch_embeddings[idx]
     
### GRAPH CONSTRUCTION ###
def update_adj(adj, cluster_labels, patch_embeddings, old_labels=None):
    # Get the unique cluster labels
    unique_cluster_labels = np.unique(cluster_labels)
    
    centroid_spots = []
    
    # For each cluster, find the spots closest to the centroid
    for cluster_label in unique_cluster_labels:
        # Find the spots in the cluster
        cluster_spots = np.where(cluster_labels == cluster_label)[0]
        
        # Find the spot closest to the centroid of the cluster
        if unique_cluster_labels.shape[0] == 1:
            # Pick a random spot as the centroid
            nearest_spot_idx = np.random.randint(0, len(cluster_spots))
            nearest_spot = cluster_spots[nearest_spot_idx]
        
        cluster_centroid = patch_embeddings[cluster_spots].mean(axis=0)
        # Find the nearest spot to the centroid
        nearest_spot_idx = np.argmin(np.linalg.norm(patch_embeddings[cluster_spots] - cluster_centroid, axis=1))
        nearest_spot = cluster_spots[nearest_spot_idx]
        
        # Connect the nearest spot to all other spots in the cluster
        for j in range(len(cluster_spots)):
            if cluster_spots[j] != nearest_spot:
                adj[cluster_spots[j], nearest_spot] = 1
                adj[nearest_spot, cluster_spots[j]] = 1
        
        # Save the nearest spot
        centroid_spots.append(nearest_spot_idx)
        
        # Multiply the cluster label of the nearest spot by -1
        cluster_labels[cluster_spots[nearest_spot_idx]] *= -1
        
        # If the cluster label is zero, set it to -(len+1)
        if cluster_labels[cluster_spots[nearest_spot_idx]] == 0:
            cluster_labels[cluster_spots[nearest_spot_idx]] = -(len(unique_cluster_labels))
    
    # Iterate over the old_labels, take the negative labels and append them to the centroid_spots
    if old_labels is not None:
        for j, old_label in enumerate(old_labels):
            if old_label < 0:
                centroid_spots.append(j)    
    
    # Make the centroid_spots unique
    centroid_spots = list(set(centroid_spots))
    
    # Connect the centroid spots to each other
    for j in range(len(centroid_spots)):
        for k in range(j+1, len(centroid_spots)):
            adj[centroid_spots[j], centroid_spots[k]] = 1
            adj[centroid_spots[k], centroid_spots[j]] = 1
        
    return adj, cluster_labels

def build_one_hop_graph(data, config):   
    # Get the number of non-zero spots for the train, val and test sets
    adj = []
    
    for i in range(len(data['slides'])):
        adj.append(torch.zeros(data['spotnum'][i], data['spotnum'][i]))
        
    labels = []
    patch_embeddings = []
    
    for slide_idx, _ in enumerate(data['slides']):
        # Build a graph of the 8 one-hop neighbors for each spot
        tmp_adj = kneighbors_graph(data['tissue_positions'][slide_idx].reset_index()[['pxl_col_in_fullres', 'pxl_col_in_fullres']].to_numpy(), mode='connectivity', n_neighbors=8).toarray()
        
        # Make the adjacency matrix symmetric and build a boolean matrix of non-zero values
        tmp_adj = (tmp_adj + tmp_adj.T) > 0
        tmp_adj = torch.tensor(tmp_adj)
        
        # Add the labels
        if config['mode'] != 'infer':
            labels.append(torch.tensor(data['counts'][slide_idx]))
        
        # Add the patch embeddings
        patch_embeddings.append(torch.tensor(data['patch_embeddings'][slide_idx]))
        
        # Set the adjacency matrix for the target slide
        adj[slide_idx] = tmp_adj
        
        # Make the diagonal of the adjacency matrix equal to 1 to include the self-loops
        adj[slide_idx].fill_diagonal_(1)
    
    # Concatenate the labels, barcodes and patch embeddings
    if config['mode'] != 'infer':
        labels = torch.cat(labels, dim=0)
    patch_embeddings = torch.cat(patch_embeddings, dim=0)
    
    return adj, labels, patch_embeddings

def build_herarchical_graph(data, config, adj):
    # Create the output directories for the clusters
    feature_cluster_path = os.path.join(config['output_dir'], 'clusters', 'feature')
    spatial_cluster_path = os.path.join(config['output_dir'], 'clusters', 'spatial')
    Path(feature_cluster_path).mkdir(parents=True, exist_ok=True)
    Path(spatial_cluster_path).mkdir(parents=True, exist_ok=True)
    
    for i in tqdm(range(len(data['slides']))):
        # Build the feature vectors
        patch_embeddings_for_slide = data['patch_embeddings'][i]
        x_coords = data['tissue_positions'][i].reset_index()[4].values
        y_coords = data['tissue_positions'][i].reset_index()[5].values
        
        coords = np.stack([x_coords, y_coords], axis=1)
        
        # Convert patch_embeddings to numpy array
        patch_embeddings_for_slide = np.array(patch_embeddings_for_slide)
        
        ### SPATIAL CLUSTERING ###
        # Get the number of clusters from the config file
        n_clusters = config['GNN']['clusters']['spatial']

        # Create a clusterer
        clusterer = KMeans(n_clusters=n_clusters, max_iter=1000)
        
        # Fit the clusterer
        clusterer.fit(coords)
        
        # Predict the cluster labels
        cluster_labels = clusterer.predict(coords)
        
        # Update the adjacency matrices
        adj[i], cluster_labels = update_adj(adj[i], cluster_labels, patch_embeddings_for_slide, None)
        
        # Create a dataframe for the slide
        slide_df = pd.DataFrame({
                'cluster_labels': cluster_labels
        })
        
        # Save the cluster labels
        slide_df.to_csv(f'{spatial_cluster_path}/{data["slides"][i]}.csv', index=False)          
        
        spatial_cluster_labels = cluster_labels.copy()
        
        ### FEATURE CLUSTERING ### 
        # Get the number of clusters from the config file
        n_clusters = config['GNN']['clusters']['feature']
        
        # Create a clusterer
        clusterer = KMeans(n_clusters=n_clusters, max_iter=1000)
        
        # Fit the clusterer
        clusterer.fit(patch_embeddings_for_slide)
        
        # Predict the cluster labels
        cluster_labels = clusterer.predict(patch_embeddings_for_slide)
        
        # Update the adjacency matrices
        adj[i], cluster_labels = update_adj(adj[i], cluster_labels, patch_embeddings_for_slide, spatial_cluster_labels)
        
        # Create a dataframe for the slide
        slide_df = pd.DataFrame({
            'cluster_labels': cluster_labels
        })
        
        # Save the cluster labels
        slide_df.to_csv(f'{feature_cluster_path}/{data["slides"][i]}.csv', index=False)
    return adj
        
def graph_construction(data, config):
    # Build the spatial graph
    print('Building the spatial graph...')
    adj, labels, patch_embeddings = build_one_hop_graph(data, config)
    print('Building the spatial graph done.')
    
    if config['GNN']['hierarchical']:
        print('Building the hierarchical graph...')
        adj = build_herarchical_graph(data, config, adj)
        print('Building the hierarchical graph done.')
    
    # Create graph_dataset
    dataset = GraphDataset(adj, labels, patch_embeddings, data, config)
    
    # Create the dataloader, the batch size is set to 1 because we want to process each slide separately
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=config['Model']['num_workers'])
    
    return dataloader
