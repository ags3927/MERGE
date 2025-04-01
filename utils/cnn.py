import os
import time
import json
import torch

import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from model import CNN_Predictor
from scipy.stats import pearsonr
from torch.optim import lr_scheduler

from misc import recursively_serialize

def train_model(dataloaders, model, criterion, optimizer, scheduler, dataset_sizes, num_epochs, device):
    # Get start time
    since = time.time()

    # Training phase
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        model.train()  # Set model to training mode
        
        # Reset the running loss
        running_mse_loss = 0.0
        running_mae_loss = 0.0
        
        # Initialize lists to store predictions and ground truths
        preds = []
        ground_truths = []
        
        # Iterate over data
        for i in dataloaders['train']:
            inputs = i['image'].to(device)
            labels = i['label'].to(device)
            squeezed_labels = labels.squeeze()
            
            # If the batch size is 1, reshape the labels
            squeezed_labels = squeezed_labels.unsqueeze(0) if len(squeezed_labels.shape) == 1 else squeezed_labels
            
            # Append the squeezed labels to the ground truths
            ground_truths.append(squeezed_labels)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                outputs = outputs.type(labels.dtype)
                outputs = outputs.view_as(labels)
                
                squeezed_outputs = outputs.squeeze()
                
                # If the batch size is 1, reshape the outputs
                squeezed_outputs = squeezed_outputs.unsqueeze(0) if len(squeezed_outputs.shape) == 1 else squeezed_outputs
                
                # Append the squeezed outputs to the predictions
                preds.append(squeezed_outputs)
                
                # Calculate the losses
                mse_loss = criterion(outputs, labels)
                mae_loss = F.l1_loss(outputs, labels)
            
                mse_loss.backward()
                optimizer.step()

            # Add the losses to the running loss (multiplied by the batch size)
            running_mse_loss += mse_loss.item() * inputs.size(0)
            running_mae_loss += mae_loss.item() * inputs.size(0)
        
        # Concatenate the predictions and ground truths
        preds = torch.cat(preds, dim=0)
        ground_truths = torch.cat(ground_truths, dim=0)
        
        # Calculate pearson correlation
        r = []
        for g in range(ground_truths.shape[1]):
            r.append(pearsonr(preds[:,g].cpu().detach(), ground_truths[:,g].cpu().detach())[0])
        corr = torch.tensor(r)
        
        # Step the scheduler
        scheduler.step()
        
        # Calculate the average training losses for the epoch and print them
        epoch_mse_loss = running_mse_loss / dataset_sizes['train']
        epoch_mae_loss = running_mae_loss / dataset_sizes['train']
        epoch_corr = np.average(corr.cpu().numpy())

        print(f'Training MSE: {epoch_mse_loss:.4f}, MAE: {epoch_mae_loss:.4f}, Corr: {epoch_corr:.4f}')
    
    # Validation phase
    model.eval()   # Set model to evaluate mode
    
    # Reset the running loss
    running_mse_loss = 0.0
    running_mae_loss = 0.0
    
    # Initialize lists to store predictions and ground truths
    preds = []
    ground_truths = []
    
    # Iterate over data
    for i in dataloaders['val']:
        inputs = i['image'].to(device)
        labels = i['label'].to(device)
        squeezed_labels = labels.squeeze()
        
        # If the batch size is 1, reshape the labels
        squeezed_labels = squeezed_labels.unsqueeze(0) if len(squeezed_labels.shape) == 1 else squeezed_labels
        
        # Append the squeezed labels to the ground truths
        ground_truths.append(squeezed_labels)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            outputs = outputs.type(labels.dtype)
            outputs = outputs.view_as(labels)
            squeezed_outputs = outputs.squeeze()
            
            # If the batch size is 1, reshape the outputs
            squeezed_outputs = squeezed_outputs.unsqueeze(0) if len(squeezed_outputs.shape) == 1 else squeezed_outputs
            
            # Append the squeezed outputs to the predictions
            preds.append(squeezed_outputs)
            
            # Calculate the losses
            mse_loss = criterion(outputs, labels)
            mae_loss = F.l1_loss(outputs, labels)
            
        # Add the losses to the running loss (multiplied by the batch size)
        running_mse_loss += mse_loss.item() * inputs.size(0)
        running_mae_loss += mae_loss.item() * inputs.size(0)
    
    # Concatenate the predictions and ground truths
    preds = torch.cat(preds, dim=0)
    ground_truths = torch.cat(ground_truths, dim=0) 
    
    # Calculate pearson correlation
    r = []
    for g in range(ground_truths.shape[1]):
        r.append(pearsonr(preds[:,g].cpu().detach(), ground_truths[:,g].cpu().detach())[0])
    corr = torch.tensor(r)
    
    # Print the training time
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    # Calculate the average validation losses for the epoch and print them
    val_mse = running_mse_loss / dataset_sizes['val']
    val_mae = running_mae_loss / dataset_sizes['val']
    val_corr = np.average(corr.cpu().numpy())
    
    print(f'Val MSE: {val_mse:4f}')
    print(f'Val MAE: {val_mae:.4f}')
    print(f'Val Corr: {val_corr:.4f}')
    
    return model, val_mse, val_mae, val_corr

def test_model(data, dataloader, model, config):
    device = config['device']
    model.eval()   # Set model to evaluate mode
    preds = []
    ground_truths = []
    for i in dataloader:
        inputs = i['image'].to(device)
        labels = i['label'].to(device)
        squeezed_labels = labels.squeeze()
            
        # If the batch size is 1, reshape the labels
        squeezed_labels = squeezed_labels.unsqueeze(0) if len(squeezed_labels.shape) == 1 else squeezed_labels
        
        ground_truths.append(squeezed_labels)
        
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            outputs = outputs.type(labels.dtype)
            outputs = outputs.view_as(labels)
            
            squeezed_outputs = outputs.squeeze()
            
            # If the batch size is 1, reshape the outputs
            squeezed_outputs = squeezed_outputs.unsqueeze(0) if len(squeezed_outputs.shape) == 1 else squeezed_outputs
            preds.append(squeezed_outputs)
    
    # Preds save path
    save_path = os.path.join('/'.join(config['output_dir'].split('/')[:-1]), 'preds')
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    start = 0
    for i in len(data['slides']):
        if data['train_slides'][i]:
            continue
        slide_preds = np.zeros((data['spotnum'][i], data['num_genes']))
        slide_preds[data['nonzero'][i]] = preds[start : start+data['nonzero'][i]].cpu().detach().numpy()
        np.save(f'{save_path}/{data["slides"][i]}.npy', slide_preds)
        
def generate_features(data, model, config):
    model.to(config['device'])
    model.eval()
    data['patch_embeddings'] = []
    
    # Remove the last layer of the model
    model = torch.nn.Sequential(*list(model.children())[:-1])
    
    for i in tqdm(range(len(data['slides']))):
        # Remove the last layer of the model
        features = []
        for patch in data['patches'][i]:
            # Move the patch to the device
            patch = patch.to(config['device'])
            
            # Generate the features and append to the list
            feature = model(patch).detach().cpu().numpy()
            features.append(feature)
            
        # Convert features to a numpy array
        features = np.array(features)
        
        # Append the features to the data dictionary        
        data['patch_embeddings'].append(features)
        
    return data

def cnn_block(data, dataloaders, dataset_sizes, config):
    if config['mode'] != 'cnn_train':
        # If the mode is not cnn_train, load the model and run the test or generate features
        model_path = config['cnn_path']
        model_ft = CNN_Predictor(num_genes=config['Data']['num_genes'], config=config)
        model_ft.load_state_dict(torch.load(model_path, map_location=config['device']))
        
        if config['mode'] == 'cnn_test':
            test_model(data, dataloaders['val'], model_ft, config)
        else:
            data = generate_features(data, model_ft, config)
    else:
        # Train the model 5 times to account for the randomness
        for run_idx in range(5):
            # Set cnn output path and create the directory
            plot_path = config['output_dir'] + "/" + str(run_idx) + "/cnn/"
            model_path = plot_path + "model_state_dict.pt"
            Path(plot_path).mkdir(parents=True, exist_ok=True)

            # Initialize the model
            model_ft = CNN_Predictor(num_genes=data['num_genes'], config=config)
            
            # Set MSE loss as the criterion
            criterion = F.mse_loss

            # Load the optimizer
            if config['CNN']['optimizer']['type'] == 'adam':
                optimizer = optim.Adam(model_ft.parameters(), lr=config['CNN']['optimizer']['lr'], weight_decay=config['CNN']['optimizer']['weight_decay'])
            
            # Load the learning rate scheduler
            if config['CNN']['scheduler']['type'] == 'step':
                lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config['CNN']['scheduler']['step_size'], gamma=config['CNN']['scheduler']['gamma'])
            
            # Train the model
            model_ft, val_mse, val_mae, val_corr = train_model(dataloaders, model_ft, criterion, optimizer, lr_scheduler, dataset_sizes, config['CNN']['epochs'], device=config['device'])
            
            # Save the model    
            torch.save(model_ft.cpu().state_dict(), model_path)
        
            # Create the metrics dictionary
            metrics = {
                'val_mse': val_mse,
                'val_mae': val_mae,
                'val_corr': val_corr
            }
            
            # Save the metrics to a json file
            metrics_file_path = plot_path + "metrics.json"
            metrics = recursively_serialize(metrics)
            with open(metrics_file_path, 'w') as f:
                json.dump(metrics, f)
        
    return data