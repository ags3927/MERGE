# MERGE

## Multi-faceted Hierarchical Graph-based GNN for Gene Expression Prediction from Whole Slide Histopathology Images, CVPR 2025. [[arXiv]](https://arxiv.org/abs/2412.02601)


## Architecture
![Architecture](fig/architecture.png)
<b>Figure:</b> The schematic of MERGE shows the overall workflow of our method. (a) Outlines the architecture of our method. The ResNetSimCLR model is fine-tuned on the gene expression prediction task using MSE loss. The last layer is discarded to yield 256 dimensional feature vectors for the patches. The graph construction step produces the multi-faceted hierarchical graph for our GNN,  which is trained on MSE loss. The output of the GNN is a M-dimensional gene expression vector at each node. (b) Shows the graph construction strategy demonstrated through reduced examples. The left column shows feature space clustering and the right column shows spatial clustering. The internal edges of a cluster are shown in white, while the shortcut edge is shown in blue. The two yellow spots represent the centroid spots of the two clusters.

## Usage

### Clone this repository
```bash
git clone git@github.com:ags3927/merge.git
cd merge
```

### Setup conda environment
```bash
conda env create -n merge -f env.yml
conda activate merge
```

### Download data
The preprocessed data for the three datasets - ST-Net, Her2ST and SCC - are available here [MERGE Data](https://drive.google.com/file/d/1Q4fP4ofDessMtCJ0a4GYbN_1XhI_hfkq/view?usp=sharing). Download the `.tar.gz` file and then extract. The ST-Net [repository](https://github.com/bryanhe/ST-Net) has a script for generating TIF files from the JPEGs. We can optionally use it to generate the TIF files from the provided JPEG files.
```bash
tar -xvzf data.tar.gz
```

### Data directory structure
The data directory structure is as follows:
```
data
├── DATASET_NAME
│   ├── barcodes
│   ├── counts_8n
│   ├── counts_spcs
│   ├── counts_spcs_to_8n
│   ├── features
│   ├── tissue_positions
│   ├── umi_counts
│   └── wsi
```
Each directory and associated files are described below:
#### The `barcodes` directory
This directory contains the barcodes of the spots in the dataset. Each file is named as `SAMPLE_NAME.csv` and contains a list of barcodes for the spots, with no header row.

#### The `counts_8n` directory
This directory contains the 8n smoothed counts of the spots in the dataset. Each file is named as `SAMPLE_NAME.npy` and contains a numpy array of the shape `(N, M)`, where `N` is the number of spots and `M` is the number of genes.

#### The `counts_spcs` directory
This directory contains the SPCS smoothed counts of the spots in the dataset. Each file is named as `SAMPLE_NAME.npy` and contains a numpy array of the shape `(N, M)`, where `N` is the number of spots and `M` is the number of genes.

#### The `counts_spcs_to_8n` directory
This directory contains the SPCS smoothed counts min-max scaled to the 8n smoothed counts. Each file is named as `SAMPLE_NAME.npy` and contains a numpy array of the shape `(N, M)`, where `N` is the number of spots and `M` is the number of genes.

#### The `features` directory
This directory contains the names of genes for each sample. Each file is named as `SAMPLE_NAME.csv` and contains a list of gene names, with no header row. <i>It is imperative that a dataset has a constant number of genes and the same order of genes across all samples.</i>

#### The `tissue_positions` directory
This directory contains the tissue positions of the spots in the dataset. Each file is named as `SAMPLE_NAME.csv` and contains a list of tissue positions for the spots, with an index column and a header row. A sample Dataframe is shown below:
```
| index | in_tissue | array_row | array_col | pxl_col_in_fullres | pxl_row_in_fullres |
| 12x24 |   1       |   12      |   24      |       4194.8       |      5340.2        |
```
As seen in the example above, the file is formatted with columns similar to a 10x h5ad Anndata file. So the `pxl_col_in_fullres` and `pxl_row_in_fullres` columns can be used to locate the spots in the whole slide images. The `in_tissue` column indicates whether the spot is in tissue or not, with `1` indicating presence in tissue. The `array_row` and `array_col` columns indicate the row and column indices of the spots in the tissue array.

#### The `umi_counts` directory
This directory contains the raw UMI counts of the spots in the dataset. Each file is named as `SAMPLE_NAME.npy` and contains a numpy array of the shape `(N, M)`, where `N` is the number of spots and `M` is the number of genes.

#### The `wsi` directory
This directory contains the whole slide images (WSIs) of the dataset. Each file is named as `SAMPLE_NAME.jpg` and contains the WSI in JPEG format. <b>For higher quality imaging, it is advisable to procure and download the TIF files of datasets and place them in this directory.</b>

## Config File
The config file `config.yaml` contains the hyperparameters for the model. The config file is structured as follows (containing example values):

```yaml
General:
  seed: 3927 # Random seed for reproducibility

Data:
  dataset: DATASET_NAME # DATASET_NAME can be stnet, her2st or skin
  num_genes: NUM_GENES # Number of genes in the dataset
  folds: 8 # Number of folds for cross-validation
  path: PATH_TO_DATASET # Path to the dataset directory
  slides: PATH_TO_SLIDES # Path to a csv file containing the list of sample names to use, can be a subset of the files present in the data directory. Sample files are provided in the config directory.

CNN:
  pretrained_path: 'pretrained/model-low-v1.pth' # Path to the pretrained ResNet model
  batch_size: 8 # Batch size for CNN training (8 seems to work well for our three datasets)
  epochs: 15 # Number of epochs for CNN training
  dropout: 0.2 # Dropout rate for CNN
  optimizer: # Optimizer settings for CNN. If you change this, make sure you adjust code accordingly. The code currently supports only adam optimizer.
    type: adam
    lr: 0.00005
    weight_decay: 0.0
  scheduler: # Scheduler settings for CNN. If you change this, make sure you adjust code accordingly. The code currently supports only step scheduler.
    type: step
    step_size: 2
    gamma: 0.5

GNN:
  type: GAT # Type of GNN model to use. Currently supports only GAT.
  epochs: 400 # Number of epochs for GNN training
  attn_heads: 8 # Number of attention heads for the GAT model
  drop_edge: 0.2 # Drop edge rate for GNN
  optimizer: # Optimizer settings for CNN. If you change this, make sure you adjust code accordingly. The code currently supports only adam optimizer.
    type: adam
    lr: 0.001
    weight_decay: 0.0
  scheduler: # Scheduler settings for CNN. If you change this, make sure you adjust code accordingly. The code currently supports only warmup scheduler.
    type: warmup
    warmup_steps: 10
```
