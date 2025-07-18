import os
import pdb
import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm
import scprep as scp

def main(args):
    # Extract command line arguments
    input_dir = args.input
    output_dir = args.output
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Glob all files in input directory
    umi_counts_paths = glob(os.path.join(input_dir, 'umi_counts', '*.npy'))
    tissue_positions_paths = glob(os.path.join(input_dir, 'tissue_positions', '*.csv'))
    
    # Sort the paths
    umi_counts_paths.sort()
    tissue_positions_paths.sort()
    
    # Check if the file names are the same
    for umi_counts_path, tissue_positions_path in zip(umi_counts_paths, tissue_positions_paths):
        assert Path(umi_counts_path).stem == Path(tissue_positions_path).stem, 'File names do not match'
    
    # Iterate over all files
    for umi_counts_path, tissue_positions_path in tqdm(zip(umi_counts_paths, tissue_positions_paths)):
        # Load data
        counts = np.load(umi_counts_path)
        tissue_positions = pd.read_csv(tissue_positions_path, header=None)
        
        # Get the in-tissue spots
        tissue_positions = tissue_positions[tissue_positions['in_tissue'] == 1]
        
        # Get the x and y coordinates
        x_coords = tissue_positions['array_row'].values
        y_coords = tissue_positions['array_col'].values
        
        # Normalize the counts
        counts = scp.transform.log(scp.normalize.library_size_normalize(counts))
        coord_names = {}
        
        # Create a dictionary of the coordinates to the index
        for coord_idx, x, y in zip(range(len(x_coords)), x_coords, y_coords):
            coord_names[f'{round(x)}_{round(y)}'] = coord_idx
        
        # Create a delta array for the 8 neighbors
        delta = np.array([[1,0],
            [0,1],
            [-1,0],
            [0,-1],
            [1,1],
            [-1,-1],
            [1,-1],
            [-1,1],
            [0,0]])
        
        for x, y in zip(x_coords, y_coords):
            ## Smooth the counts using the 8 neighbors ###
            # Find the center spot coordinates
            center = f'{round(x)}_{round(y)}'
            
            # Find the neighbors of the center using the delta
            neighbors = []
            for d in delta:
                neighbor = f'{round(x+d[0])}_{round(y+d[1])}'
                if neighbor in coord_names:
                    neighbors.append(coord_names[neighbor])
            
            # Find the counts of the neighbors
            neighbor_counts = counts[neighbors]
            
            # Smooth the counts
            smooth_counts = np.mean(neighbor_counts, axis=0)
            
            # Replace the counts of the center spot with the smoothed counts
            counts[coord_names[center]] = smooth_counts        
        
        # Save normalized data
        np.save(os.path.join(output_dir, Path(umi_counts_path).stem + '.npy'), counts, allow_pickle=False)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Input directory for dataset containing umi counts and tissue positions')
    parser.add_argument('-o', '--output', type=str, default='output', help='Output directory to store the eight-neighbor smoothed files')
    args = parser.parse_args()
    main(args)