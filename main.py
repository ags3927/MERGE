import yaml
import torch

from pathlib import Path
from argparse import ArgumentParser
from utils.data import preprocess_data
from utils.graph import graph_construction
from utils.cnn import cnn_block
from utils.gnn import gnn_block

def main(args):
    # Extract the gene and config file
    config_file = args.config
    output_dir = args.output
    device = args.device
    cnn_path = args.cnn_path
    gnn_path = args.gnn_path
    mode = args.mode
    
    # Load the config file as yaml
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Set additional config parameters
    config['output_dir'] = output_dir
    config['Data']['fold'] = args.fold
    config['cnn_path'] = cnn_path
    config['gnn_path'] = gnn_path
    config['mode'] = mode
    
    # Create the output directory
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    # Config file path
    config_file_path = f'{config["output_dir"]}/config.yaml'
    
    config['device'] = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        
    # Save the config file as yaml
    with open(config_file_path, 'w') as f:
        yaml.dump(config, f)
    
    # Initialize the device
    config['device'] = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    
    # Preprocess the data
    data, dataloaders, dataset_sizes = preprocess_data(config)
    
    # CNN Block
    data = cnn_block(data=data, dataloaders=dataloaders, dataset_sizes=dataset_sizes, config=config)
    
    if mode in ['gnn_train', 'gnn_test', 'gnn_test_ext']:
        # Construct graph datasets
        graph_dataloaders = graph_construction(data, config)
        
        # GNN Block
        gnn_block(data=data, dataloaders=graph_dataloaders, config=config)
   
    return
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-d', '--device', type=int, required=True)
    parser.add_argument('-f', '--fold', type=int, default=0)
    parser.add_argument('--cnn_path', type=str, default='Path to CNN model')
    parser.add_argument('--gnn_path', type=str, default='Path to GNN model')
    parser.add_argument('--mode', type=str, default='cnn_train', help='cnn_train, cnn_test, gnn_train, gnn_test')
    
    args = parser.parse_args()
        
    main(args)