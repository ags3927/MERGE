import pandas as pd
import numpy as np
import scipy.sparse as sp
import scprep as scp
import json
import skimage.io
import torch
import os
from torchvision import transforms
from utils import GeneDataset
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.stats import pearsonr
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.utils.data import DataLoader

def preprocess_data(config):
    # Extract the data path, slides path, and train test split path
    data_path = config['Data']['path']
    
    # Initialize the data dictionary
    data = {}
    
    # Initialize the lists for the data dictionary
    key_list = [
        'barcodes',
        'spotnum',
        'counts',
        'nonzero',
        'tissue_positions',
        'patches',
        'counts',
        'patch_embeddings'
    ]
    for key in key_list:
        data[key] = []
        
    # Load the slides file
    data['slides'] = pd.read_csv(config['Data']['slides'], header=None)[0].values
    data['slides'].sort()
    
    # Load the number of genes
    data['num_genes'] = config['Data']['num_genes']
    
    # Do 8 fold cross validation
    kf = KFold(n_splits=config['Data']['folds'], shuffle=True, random_state=3927)
    train_idx, val_idx = list(kf.split(data['slides']))[config['fold']]
            
    # Iterate over the slides and assign them to train or test sets
    train_test_split = {
        'train': data['slides'][train_idx].tolist(),
        'val': data['slides'][val_idx].tolist()
    }
    data['train_slides'] = np.isin(data['slides'], np.array(train_test_split['train']))
    data['val_slides'] = np.isin(data['slides'], np.array(train_test_split['val']))
    
    for slide in tqdm(data['slides']):
        # Load the barcodes file and add to the data dictionary
        barcodes = pd.read_csv(f'{data_path}/barcodes/{slide}.csv', header=None)[0].values
        data['barcodes'].append(barcodes)
        
        # Save the number of spots
        data['spotnum'].append(len(barcodes))
        
        # Load the tissue positions file and extract the x and y coordinates
        tissue_positions = pd.read_csv(f'{data_path}/tissue_positions/{slide}.csv')
        tissue_positions = tissue_positions[tissue_positions['in_tissue'] == 1]
        data['tissue_positions'].append(tissue_positions)
        
        # Load the counts file
        counts = np.load(f'{data_path}/counts_spcs_to_8n/{slide}.npy')
        
        # Save the non-zero umi count indices
        data['nonzero'].append(counts.sum(axis=1) > 0)
        
        # Load the whole slide image and extract the patches
        # Check if the slide is in the data path
        if os.path.exists(f'{data_path}/wsi/{slide}.tif'):
            wsi_name = f'{slide}.tif'
        elif os.path.exists(f'{data_path}/wsi/{slide}.tiff'):
            wsi_name = f'{slide}.tiff'
        elif os.path.exists(f'{data_path}/wsi/{slide}.svs'):
            wsi_name = f'{slide}.svs'
        elif os.path.exists(f'{data_path}/wsi/{slide}.jpg'):
            wsi_name = f'{slide}.jpg'
        
        wsi = skimage.io.imread(f'{data_path}/wsi/{wsi_name}')
        patches = []
        
        
        # Extract the x and y coordinates
        x_coords = tissue_positions['pxl_col_in_fullres'].values
        y_coords = tissue_positions['pxl_row_in_fullres'].values
        
        # Extract the patches
        for x, y in zip(x_coords, y_coords):      
            x, y = round(x), round(y)
            
            # If patch is out of bounds, extract with padding
            if x < 128:
                x = 128
            if y < 128:
                y = 128
            if x > wsi.shape[0] - 128:
                x = wsi.shape[0] - 128
            if y > wsi.shape[1] - 128:
                y = wsi.shape[1] - 128
            
            patch = wsi[y-128:y+128, x-128:x+128, :3]
            
            if patch.shape != (256, 256, 3):
                print(f'Patch shape: {patch.shape}')
                print(f'WSI shape: {wsi.shape}')
                print(f'x: {x}, y: {y}')
            
            patches.append(patch)
            
        # Save the counts
        data['counts'].append(counts)
        
        # Transpose the patches from (N, H, W, C) to (N, C, H, W)
        patches = torch.tensor(np.array(patches).transpose([0,3,1,2]), dtype=torch.float)
        data['patches'].append(patches)
        
    data_transforms = {
        'train': torch.nn.Sequential(
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ),
        'val': torch.nn.Sequential(
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )
    }
    
    # Extract the non-zero indices, patches, relabeled_counts for the training and testing sets
    train_nonzero, val_nonzero = [], []
    train_patches, val_patches = [], []
    train_relabeled_counts, val_relabeled_counts = [], []
        
    for i in range(len(data['train_slides'])):
        if data['train_slides'][i]:
            train_nonzero.append(data['nonzero'][i])
            train_patches.append(data['patches'][i])
            train_relabeled_counts.append(data['counts'][i])
        if data['val_slides'][i]:
            val_nonzero.append(data['nonzero'][i])
            val_patches.append(data['patches'][i])
            val_relabeled_counts.append(data['counts'][i])
            
    # Concatenate the non-zero indices, patches, relabeled_counts for the train, and val sets
    # Non-zero indices
    train_nonzero = np.concatenate(train_nonzero)
    val_nonzero = np.concatenate(val_nonzero)
    
    # Patches
    train_patches = torch.cat(train_patches)[train_nonzero]
    val_patches = torch.cat(val_patches)[val_nonzero]
    
    # Relabeled counts
    train_relabeled_counts = np.concatenate(train_relabeled_counts)[train_nonzero]
    val_relabeled_counts = np.concatenate(val_relabeled_counts)[val_nonzero]
    
    # Create the training, validation and testing datasets
    train_dataset = GeneDataset(train_patches, train_relabeled_counts, data_transforms['train'])
    val_dataset = GeneDataset(val_patches, val_relabeled_counts, data_transforms['val'])
    
    # Create the image datasets dictionary
    image_datasets = {
        'train': train_dataset,
        'val': val_dataset
    }
    
    # Extract dataset sizes
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset)
    }
    
    # Initialize the dataloaders dictionary
    dataloaders = {i: torch.utils.data.DataLoader(image_datasets[i], batch_size=config['cnn_batch_size'], shuffle=False, num_workers=8) for i in ['train', 'val']}
    
    return data, image_datasets, dataloaders, dataset_sizes