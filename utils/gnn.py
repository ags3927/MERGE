import os
import json
import torch

import numpy as np
import torch.nn.functional as F

from pathlib import Path
from scipy.stats import pearsonr
from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup

from utils.model import GATNet

def gnn_train(gnn, dataloader, optimizer, alpha=0):
    # Set the model to training mode
    gnn.train()
    
    # Create variables to store the training losses and correlations
    train_mse, train_corr = [], []
    
    for batch in dataloader:
        # Get the batch data
        slide_index, edge_indices, labels, patch_embeddings = batch
        
        # Squeeze to remove the extra dimension
        labels = labels.squeeze()
        edge_indices = edge_indices.squeeze()
        patch_embeddings = patch_embeddings.squeeze()
    
        # Track history if only in train
        with torch.set_grad_enabled(True):
            output = gnn(patch_embeddings, edge_indices)

        output = output.type(labels.dtype)
        output = output.view_as(labels)

        # Calculate the MSE loss
        mse = F.mse_loss(output, labels)
    
        # Calculate the correlation
        output = output.T
        labels = labels.T
        corr = []
        for g in range(labels.shape[0]):
            corr.append(pearsonr(output[g].cpu().detach(), labels[g].cpu().detach())[0])
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
        
        # Add the loss and correlation to the training losses and correlations
        train_mse.append(mse.item())
        train_corr.append(corr.item())
    
    # Average the training losses and correlations
    train_mse = np.mean(train_mse)
    train_corr = np.mean(train_corr)
    
    return train_mse, train_corr

def gnn_test_save(plot_path, gnn, dataloader, data, config):
    # Set the model to evaluation mode
    gnn.eval()
    
    # Get the number of genes
    num_genes = data['num_genes']
    
    # Create variables to store the test losses and correlations
    test_mse, test_mae, test_corr = [], [], []
    
    for batch in dataloader:
        # Get the batch data
        slide_index, edge_indices, labels, patch_embeddings = batch
        
        # Squeeze to remove the extra dimension
        labels = labels.squeeze()
        edge_indices = edge_indices.squeeze()
        patch_embeddings = patch_embeddings.squeeze()
    
        with torch.set_grad_enabled(False):
            output = gnn(patch_embeddings, edge_indices)
            
        # Remove the last / and the last word from config['output_dir'] to get the base directory
        base_dir = '/'.join(config['output_dir'].split('/')[:-1])       

        # Create the directory to save the predictions
        preds_save_path = f'{base_dir}/preds/'
        Path(preds_save_path).mkdir(parents=True, exist_ok=True)
        
        # Separate the output counts into the val slice dimensions save the results
        # Iterate over the slides and save the predictions
        for idx, slide in enumerate(data['slides']):
            # Skip the slides that are in the train set
            if data['train_slides'][idx]:
                continue
            
            # Create a numpy array of zeros with the shape of (num_spots, num_genes)
            predicted_counts = np.zeros(shape=(data['spotnum'][idx], num_genes))
            
            # Assign the predicted counts to the non-zero indices. We have to do this because when we train the model, we only use the non-zero spots. So the predicted counts are only for the non-zero spots. While saving we assign those predicted counts to the non-zero spots in appropriate indices.
            predicted_counts[data['nonzero'][idx]] = output.cpu().detach().numpy()
            
            # Save the predicted counts
            np.save(f'{preds_save_path}/{slide}.npy', predicted_counts)
        
        output = output.type(labels.dtype)
        output = output.view_as(labels)
        
        # Calculate the MSE and MAE loss
        mse_loss = F.mse_loss(output, labels)
        mae_loss = F.l1_loss(output, labels)
        
        # Calculate the correlation
        output = output.T
        labels = labels.T
        corr = []
        for g in range(num_genes):
            corr.append(pearsonr(output[g].cpu().detach(), labels[g].cpu().detach())[0])
        corr = torch.tensor(corr)
        
        # Add the loss and correlation to the testing losses and correlations
        test_mse.append(mse_loss.item())
        test_mae.append(mae_loss.item())
        test_corr.append(corr)
    
    # Average the testing losses and correlations
    test_mse = np.mean(test_mse)
    test_mae = np.mean(test_mae)
    test_corr = np.mean(test_corr)
    
    # Save the results to a json file
    results = {
        'mse': test_mse,
        'mae': test_mae,
        'corr': test_corr
    }
    results_file_path = f'{plot_path}/results.json'
    with open(results_file_path, 'w') as f:
        json.dump(results, f)
           
    return

def gnn_test(gnn, dataloader, num_genes):
    # Set the model to evaluation mode
    gnn.eval()
    
    # Create variables to store the test losses and correlations
    test_mse, test_mae, test_corr = [], [], []
    
    for batch in dataloader:
        # Get the batch data
        slide_index, edge_indices, labels, patch_embeddings = batch
        
        # Squeeze to remove the extra dimension
        labels = labels.squeeze()
        edge_indices = edge_indices.squeeze()
        patch_embeddings = patch_embeddings.squeeze()
    
        # Track history if only in train
        with torch.set_grad_enabled(False):
            output = gnn(patch_embeddings, edge_indices)

        output = output.type(labels.dtype)
        output = output.view_as(labels)
        
        # Calculate the MSE and MAE loss
        mse = F.mse_loss(output, labels)
        mae = F.l1_loss(output, labels)
    
        # Calculate the correlation
        output = output.T
        labels = labels.T
        corr = []
        for g in range(num_genes):
            corr.append(pearsonr(output[g].cpu().detach(), labels[g].cpu().detach())[0])
        corr = torch.tensor(corr)
    
        # Average the correlation
        corr = np.average(corr.cpu().numpy())
        
        # Add the loss and correlation to the testing losses and correlations
        test_mse.append(mse.item())
        test_mae.append(mae.item())
        test_corr.append(corr)

    # Average the testing losses and correlations
    test_mse = np.mean(test_mse)
    test_mae = np.mean(test_mae)
    test_corr = np.mean(test_corr)
        
    return test_mse, test_mae, test_corr

def gnn_block(data, dataloaders, config):    
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
        gnn_test_save(plot_path, gnn, dataloaders['val'], data, config)
        
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
                scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps=config['GNN']['scheduler']['warmup_steps'], t_total=config['GNN']['epochs'])

            # Train the model
            for i in range(config['GNN']['epochs'] + 1):
                train_mse, train_corr = gnn_train(gnn, dataloaders['train'], optimizer)
                
                # Run validation every 40 epochs and step the scheduler
                if(i % 40 == 0):
                    scheduler.step()
                    
                    test_mse, test_mae, test_corr = gnn_test(gnn, dataloaders['val'], data['num_genes'])
                    
                    # Print the training and validation losses and correlations
                    print(f'Epoch: {i}, Train MSE: {train_mse}, Train Corr: {train_corr}')
                    print(f'Epoch: {i}, Test MSE: {test_mse}, Test MAE: {test_mae}, Test Corr: {test_corr}')
            
            # Save the model once the training is done
            plot_path = config['output_dir'] + "/" + str(run_idx) + "/gnn/"
            Path(plot_path).mkdir(parents=True, exist_ok=True)
            
            gnn_test_save(plot_path, gnn, dataloaders['val'], data, config)
            
            model_path = plot_path + f"model_state_dict.pt"
            torch.save(gnn.cpu().state_dict(), model_path)
            
            del gnn
    return
