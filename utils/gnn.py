import os
import json
import torch

import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from einops import rearrange
from scipy.stats import pearsonr
from torch.utils.data import Dataset
from torch.optim import lr_scheduler
from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup

from model import GATNet

def gnn_train(gnn, input, edge_index, train_lbls, optimizer, alpha=0):
    # Set the model to training mode
    gnn.train()
    
    # Track history if only in train
    with torch.set_grad_enabled(True):
        output = gnn(input, edge_index)

    output = output.type(train_lbls.dtype)
    output = output.view_as(train_lbls)

    # Calculate the MSE loss
    mse = F.mse_loss(output, train_lbls)
    
    # Calculate the correlation
    output = output.T
    train_lbls = train_lbls.T
    corr = []
    for g in range(train_lbls.shape[0]):
        corr.append(pearsonr(output[g].cpu().detach(), train_lbls[g].cpu().detach())[0])
    corr = torch.tensor(corr)
    
    # Average the correlation using torch
    corr = torch.mean(corr)
    
    # Add correlation as a loss, to be maximized. Normally we do not use correlation loss, so alpha=0
    loss = mse + alpha * (1-corr)
    
    # Backpropagation    
    loss.backward()

    # Update the weights
    optimizer.step()
    optimizer.zero_grad()
    
    return mse, corr

def gnn_test_save(plot_path, gnn, input, edge_index, test_lbls, data, config):
    # Set the model to evaluation mode
    gnn.eval()
    
    # Get the number of genes
    num_genes = data['num_genes']
    
    with torch.set_grad_enabled(False):
        output = gnn(input, edge_index)
        
    # Remove the last / and the last word from config['output_dir'] to get the base directory
    base_dir = '/'.join(config['output_dir'].split('/')[:-1])       

    # Create the directory to save the predictions
    preds_save_path = f'{base_dir}/preds/'
    Path(preds_save_path).mkdir(parents=True, exist_ok=True)
    
    
    # Separate the output counts into the val slice dimensions save the results
    start = 0
    # Iterate over the slides and save the predictions
    for idx, slide in enumerate(data['slides']):
        # Skip the slides that are in the train set
        if data['train_slides'][idx]:
            continue
        
        # Create a numpy array of zeros with the shape of (num_spots, num_genes)
        predicted_counts = np.zeros(shape=(data['spotnum'][idx], num_genes))
        non_zero = data['nonzero'][idx]
        non_zero_count = np.sum(non_zero)
        
        # Assign the predicted counts to the non-zero indices. We have to do this because when we train the model, we only use the non-zero spots. So the predicted counts are only for the non-zero spots. While saving we assign those predicted counts to the non-zero spots in appropriate indices.
        predicted_counts[non_zero] = output.cpu().detach().numpy()[start:start+non_zero_count]
        start += non_zero_count
        
        # Save the predicted counts
        np.save(f'{preds_save_path}/{slide}.npy', predicted_counts)
    
    output = output.type(test_lbls.dtype)
    output = output.view_as(test_lbls)
    
    # Calculate the MSE and MAE loss
    mse_loss = F.mse_loss(output, test_lbls)
    mae_loss = F.l1_loss(output, test_lbls)
    
    # Calculate the correlation
    output = output.T
    test_lbls = test_lbls.T
    corr = []
    for g in range(num_genes):
        corr.append(pearsonr(output[g].cpu().detach(), test_lbls[g].cpu().detach())[0])
    corr = torch.tensor(corr)
    
    # Save the results to a json file
    results = {
        'mse': mse_loss.item(),
        'mae': mae_loss.item(),
        'corr': np.average(corr.cpu().numpy())
    }
    results_file_path = f'{plot_path}/results.json'
    with open(results_file_path, 'w') as f:
        json.dump(results, f)
           
    return

def gnn_test(gnn, input, edge_index, test_lbls, num_genes):
    # Set the model to evaluation mode
    gnn.eval()
    
    # Track history if only in train
    with torch.set_grad_enabled(False):
        output = gnn(input, edge_index)

    output = output.type(test_lbls.dtype)
    output = output.view_as(test_lbls)
    
    # Calculate the MSE and MAE loss
    mse = F.mse_loss(output, test_lbls)
    mae = F.l1_loss(output, test_lbls)
    
    # Calculate the correlation
    output = output.T
    test_lbls = test_lbls.T
    corr = []
    for g in range(num_genes):
        corr.append(pearsonr(output[g].cpu().detach(), test_lbls[g].cpu().detach())[0])
    corr = torch.tensor(corr)
    
    # Average the correlation
    corr = np.average(corr.cpu().numpy())
    
    return mse, mae, 

def gnn_block(data, image_datasets, config, graph_datasets):
    x = {}
    
    # Get the number of non-zero spots for the train and val sets
    train_nonzero, val_nonzero = [], []
    train_features, val_features = [], []
    
    for i in range(len(data['slides'])):
        if data['train_slides'][i]:
            train_nonzero.append(data['nonzero'][i])
            train_features.append(data['patch_embeddings'][i])
        elif data['val_slides'][i]:
            val_nonzero.append(data['nonzero'][i])
            val_features.append(data['patch_embeddings'][i])
    
    # Non-zero indices
    train_nonzero = np.concatenate(train_nonzero)
    val_nonzero = np.concatenate(val_nonzero)
    
    # Patch embeddings concatenated
    x["train"] = torch.tensor(np.concatenate(train_features)[train_nonzero]).to(config['device'])
    x["val"] = torch.tensor(np.concatenate(val_features)[val_nonzero]).to(config['device'])
    
    if config['mode'] in ['gnn_test']:
        # Create the GNN model
        if config['GNN']['type'] == 'GAT':
            gnn = GATNet(num_genes=data['num_genes'], num_heads=config['GNN']['attn_heads'], drop_edge=config['GNN']['drop_edge']).to(config['device'])
        
        # Load the model weights
        model_path = config['gnn_path']
        gnn.load_state_dict(torch.load(model_path, map_location=config['device']))
        
        # Test the model
        # Get the directory of the model path
        plot_path = os.path.dirname(model_path)
        gnn_test_save(plot_path, gnn, x['val'], graph_datasets['val'][0], torch.tensor(image_datasets['val'].y).type(torch.double).to(config['device']), data, config)
        
        del gnn
    elif config['mode'] == 'gnn_train':
        # Train the model 5 times to account for the randomness in the training process
        for run_idx in range(5):
            # Create the GNN model
            if config['GNN']['type'] == 'GAT':
                gnn = GATNet(num_genes=data['num_genes'], num_heads=config['GNN']['attn_heads'], drop_edge=config['GNN']['drop_edge']).to(config['device'])
            
            if config['GNN']['optimizer']['type'] == "adam":
                optimizer = torch.optim.Adam(gnn.parameters(), lr=config['GNN']['optimizer']['lr'], weight_decay=config['GNN']['optimizer']['weight_decay'])
                
            if config['GNN']['scheduler']['type'] == "warmup":            
                scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps=config['GNN']['scheduler']['warmup_steps'], num_training_steps=config['GNN']['epochs'])

            # Train the model
            for i in range(config['gnn_epochs'] + 1):
                train_mse, train_corr = gnn_train(gnn, x['train'], graph_datasets['train'][0], torch.tensor(image_datasets['train'].y).to(config['device']), optimizer)
                
                # Run validation every 40 epochs and step the scheduler
                if(i % 40 == 0):
                    scheduler.step()
                    
                    test_mse, test_mae, test_corr = gnn_test(gnn, x['val'], graph_datasets['val'][0], torch.tensor(image_datasets['val'].y).type(torch.double).to(config['device']), data['num_genes'])
                    
                    # Print the training and validation losses and correlations
                    print(f'Epoch: {i}, Train MSE: {train_mse}, Train Corr: {train_corr}')
                    print(f'Epoch: {i}, Test MSE: {test_mse}, Test MAE: {test_mae}, Test Corr: {test_corr}')
            
            # Save the model once the training is done
            plot_path = config['output_dir'] + "/" + str(run_idx) + "/gnn/"
            Path(plot_path).mkdir(parents=True, exist_ok=True)
            
            gnn_test_save(plot_path, gnn, x['val'], graph_datasets['val'][0], torch.tensor(image_datasets['val'].y).type(torch.double).to(config['device']), data, config)
            
            model_path = plot_path + f"model_state_dict.pt"
            torch.save(gnn.cpu().state_dict(), model_path)
            
            del gnn
    return
