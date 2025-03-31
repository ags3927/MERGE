import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch_geometric.utils import dropout_edge
from torch_geometric.nn import GATConv, LayerNorm

class ResnetMLP(nn.Module):
    def __init__(self):
        super(ResnetMLP, self).__init__()
        # Load the ResNet model
        resnet = models.resnet18(weights=None, norm_layer=nn.InstanceNorm2d)
        
        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add a projection MLP
        num_ftrs = resnet.fc.in_features
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        
        # This is hardcoded to 256, because the pretrained model was trained with this size
        self.l2 = nn.Linear(num_ftrs, 256)
        
    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return x

class CNN_Predictor(nn.Module):
    def __init__(self, num_genes, config):
        super(CNN_Predictor, self).__init__()
        
        # Load the ResNetMLP model
        self.module = ResnetMLP().to(config['device'])
        
        # Load the TCGA pretrained ResNetMLP weights
        self.load_state_dict(torch.load(config['CNN']['pretrained_path'], map_location=config['device']))   
        
        # Add a dropout layer        
        self.dropout = nn.Dropout(p=config['CNN']['dropout']).to(config['device'])
        
        # Add a fully connected layer to predict the gene expressions
        num_features = self.module.l2.out_features    
        self.fc = nn.Linear(num_features, num_genes).to(config['device'])

    def forward(self, x):
        x = self.module(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc(x)  
        return x

class GATNet(torch.nn.Module):
    def __init__(self, num_genes, num_heads=8, drop_edge=0.2):
        super(GATNet, self).__init__()
        dim1 = 448
        dim2 = 384
        dim3 = 256
        
        # The number of attention heads is the same for all layers
        # TODO: Experiment with different number of heads for each layer
        headn = num_heads
        
        self.drop_edge = drop_edge
        
        # The first input dimension is 256 because the ResNetMLP model outputs 256 features
        self.nn1 = GATConv(256, dim1, headn)
        self.layer_norm1 = LayerNorm(dim1 * headn)
        
        self.nn2 = GATConv(dim1 * headn, dim2, headn)
        self.layer_norm2 = LayerNorm(dim2 * headn)
        
        self.nn3 = GATConv(dim2 * headn, dim3, headn)
        self.layer_norm3 = LayerNorm(dim3 * headn)
        
        # The output dimension is the number of genes
        self.nn4 = GATConv(dim3 * headn, num_genes)
        
    def forward(self, x, edge_index):
        # pdb.set_trace()
        
        # # Randomly drops out edges with probability p        
        edge_index, _ = dropout_edge(edge_index, p=self.drop_edge, training=self.training)
        
        x = F.relu(self.nn1(x, edge_index))
        # Layer normalization
        x = self.layer_norm1(x)
        
        x = F.relu(self.nn2(x, edge_index))
        # Layer normalization
        x = self.layer_norm2(x)
        
        x = F.relu(self.nn3(x, edge_index))
        # Layer normalization
        x = self.layer_norm3(x)
        
        x = self.nn4(x, edge_index)
        
        return x
    
