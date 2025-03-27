import torch
from cbm.cbm import cbm
from cbm.mkl import mkl
from cbm.cbm4mm import cbm4mm
from cbm.cbm4ad import cbm4ad
from cbm.cbm4dad import cbm4dad
from cbm.mkl4mm import mkl4mm
from cbm.mkl4ad import mkl4ad
from cbm.mkl4dad import mkl4dad
from nn.gcn import CBMGCNInference, MKLGCNInference
from ogb.nodeproppred import PygNodePropPredDataset
#from torch import float32, arange, randint
from torch_geometric.data import Batch
from torch_geometric.datasets import TUDataset, SuiteSparseMatrixCollection, Planetoid

import os
data_dir = os.path.dirname(os.path.abspath(__file__)) + '/../data'

############################################################
########################## LAYERS ##########################
############################################################

def set_adjacency_matrix(layer, edge_index, d_left=None, d_right=None, alpha=16):
    if layer == "cbm-gcn-inference":
        # Class defined in 'cbm/cbm4dad' - used in our experimental evaluation.
        return cbm4dad(edge_index.to(torch.int32), torch.ones(edge_index.size(1), dtype=torch.float), alpha)
    elif layer == "mkl-gcn-inference":
        # Class defined in 'cbm/mkl4dad' - used in our experimental evaluation.  
        return mkl4dad(edge_index)
    elif layer == "cbm-ax":
        # Class defined in 'cbm/cbm4mm' - used in our experimental evaluation.  
        return cbm4mm(edge_index.to(torch.int32), torch.ones(edge_index.size(1), dtype=torch.float), alpha)
    elif layer == "cbm-adx":
        # Class defined in 'cbm/cbm4ad' - used in our experimental evaluation.  
        return cbm4ad(edge_index.to(torch.int32), torch.ones(edge_index.size(1), dtype=torch.float), alpha)
    elif layer == "cbm-dadx":
        # Class defined in 'cbm/cbm4dad' (equivalent to cbm4gcn') - used in our experimental evaluation.
        return cbm4dad(edge_index.to(torch.int32), torch.ones(edge_index.size(1), dtype=torch.float), alpha)
    elif layer == "mkl-ax":
        # Class defined in 'cbm/mkl4mm' - used in our experimental evaluation.  
        return mkl4mm(edge_index)
    elif layer == "mkl-adx":
        # Class defined in 'cbm/mkl4ad' - used in our experimental evaluation.  
        return mkl4ad(edge_index)
    elif layer == "mkl-dadx":
        # Class defined in 'cbm/mkl4dad' (equivalent to mkl4gcn') - used in our experimental evaluation.
        return mkl4dad(edge_index)
    elif layer == "cbm":
        # Call to generalized CBM class (defined in 'cbm/cbm.py') - NOT used in our experimental evaluation.
        return cbm(edge_index.to(torch.int32), torch.ones(edge_index.size(1), dtype=torch.float), d_left, d_right, alpha)
    elif layer == "mkl":
        # Call to generalized MKL class (defined in 'cbm/mkl.py') - NOT used in our experimental evaluation.
        return mkl(edge_index.to(torch.int32), torch.ones(edge_index.size(1), dtype=torch.float), d_left, d_right)
    else:
        raise NotImplementedError(f"Layer {layer} is not valid")

def set_layer(layer, in_features, out_features, bias, a, a_t=None):
    if layer == "cbm-gcn-inference":
        return CBMGCNInference(in_features, out_features, bias, a)
    elif layer == "mkl-gcn-inference":
        return MKLGCNInference(in_features, out_features, bias, a)
    else:
        raise NotImplementedError(f"Layer {layer} is not valid")


############################################################
######################### DATASETS #########################
############################################################

def print_dataset_info(name, dataset):
    print('------------------------------------------------------')
    print(f'Dataset: {name}')
    print(f'Number of Nodes: {dataset.num_nodes}')
    print(f'Number of Edges: {dataset.num_edges}')
    print('------------------------------------------------------')


# Examples:
# load_tudataset('COLLAB')
def load_tudataset(name):     # graph and node prediction
    dataset = Batch.from_data_list(TUDataset(root=data_dir, name=name))
    return dataset


# Examples:
# load_snap('ca-HepPh')
# load_snap('ca-AstroPh')
def load_snap(name):      # node prediction
    dataset = SuiteSparseMatrixCollection(root=data_dir, name=name, group='SNAP')[0]
    return dataset
    
# examples:
# load_ogbn_proteins()
def load_ogbn_proteins():
    dataset = PygNodePropPredDataset(name='ogbn-proteins', root=data_dir)[0]
    return dataset

# Examples:
# load_planetoid('Cora')
# load_planetoid('PubMed')
def load_planetoid(name):       # node prediction
    dataset = Planetoid(root=data_dir, name=name)[0]
    return dataset

# Examples:
# load_dimacs('coPapersDBLP')
# load_dimacs('coPapersCiteseer')
def load_dimacs(name):      # node prediction
    dataset = SuiteSparseMatrixCollection(root=data_dir, name=name, group='DIMACS10')[0]
    return dataset

# The second parameter returned represents CBM's alpha value. In this script all
# all alpha values were set to 16, which was shown to work well in multi-core
# settings. To find the best alpha values for each dataset (and experimental 
# environment) run alpha_searcher.sh and add them to load_dataset.
def load_dataset(name):
    if name == "ca-HepPh":
        return load_snap("ca-HepPh"), 16
    elif name == "ca-AstroPh":
        return load_snap("ca-AstroPh"), 16
    elif name == "ogbn-proteins-raw":
        return load_ogbn_proteins(), 16
    elif name == "PubMed":
        return load_planetoid("PubMed"), 16
    elif name == "Cora":
        return load_planetoid("Cora"), 16
    elif name == "coPapersCiteseer":
        return load_dimacs("coPapersCiteseer"), 16
    elif name == "coPapersDBLP":
        return load_dimacs("coPapersDBLP"), 16
    elif name == "COLLAB":
        return load_tudataset("COLLAB"), 16
    else:
        raise NotImplementedError(f"Dataset {name} is not valid")