"""
This script records the execution time of a standard Graph Convolutional Network
(GCN) inference stage. The graph's laplacian is converted to CBM or CSR formats
with 'cbm/cbm4dad.py' or 'cbm/mkl4dad.py', respectively.

Then, the matrix products involving the graph's laplacian inside the forward 
pass are carried by the matrix multiplication methods of 'cbm/cbm4dad.py' or 
'cbm/mkl4dad.py', depending on the representation chosen previously.

At the end of execution, the script outputs benchmarking statistics including:
    - mean inference time and standard deviation.
    - minimum and maximum inference time recorded.

Example Usage:
    [OMP_ENV_VARS] python benchmark/benchmark_inference --nn cbm-gcn-inference

Note: 
    - See 'parser.add_arguments' for more options.
    - see 'benchmark/utilities.py' for datasets options. 
"""

import argparse
from time import time
from torch import inference_mode, rand, tensor
from torch.nn import Module, ModuleList
from utilities import set_layer, load_dataset, print_dataset_info, set_adjacency_matrix

import warnings
warnings.simplefilter("ignore", UserWarning)

class NodePrediction(Module):
    """
    A PyTorch module implementing a multi-layer GCN for node prediction tasks.
    
    This class creates a GCN architecture with:
    - One input layer
    - Configurable number of hidden layers
    - One output layer
    - ReLU activation between layers
    
    Args:
        layer (str): 
            Layer implementation to use ('cbm-gcn-inference' or 'mkl-gcn-inference').
        
        in_features (int): 
            Number of input features.
        
        hidden_features (int): 
            Number of features in hidden layers.
        
        out_features (int): 
            Number of output features.
        
        num_hidden_layers (int): 
            Number of hidden layers.
        
        bias (bool): 
            Whether to use bias in linear layers.
        
        a (torch.Tensor): 
            Adjacency matrix instance in either CBM or CSR format.
        
        a_t (torch.Tensor): 
            Transpose of adjacency matrix (optional)
    """
    
    def __init__(self, layer, in_features, hidden_features, out_features, num_hidden_layers, bias, a, a_t=None):
        super(NodePrediction, self).__init__()

        # input layer
        self.layers = [set_layer(layer, in_features, hidden_features, bias, a, a_t)]
        # hidden layers
        self.layers += [set_layer(layer, hidden_features, hidden_features, bias, a, a_t) for _ in range(num_hidden_layers)]
        # output layer
        self.layers += [set_layer(layer, hidden_features, out_features, bias, a, a_t)]
        # required for training so that the optimizer can find the parameters
        self.layers = ModuleList(self.layers)

    # standard forward-pass for GCN inference    
    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            # this layer uses:
            #   - the 'cbm4gcn' class, when '--nn cbm-gcn-inference' was passed;
            #   - the 'mkl4gcn' class, when '--nn mkl-gcn-inference' was passed.
            x = layer(x, edge_index)
            x = x.relu()
        out = self.layers[-1](x, edge_index)
        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nn", choices=["cbm-gcn-inference", "mkl-gcn-inference"], required=True)
    parser.add_argument("--dataset", type=str, default="ca-HepPh")
    parser.add_argument("--hidden_features", type=int, default=500, help="Number of hidden features to use in the model. If --fake is set, this will also be the number of input features.")
    parser.add_argument("--num_hidden_layers", type=int, default=0, help="Number of hidden layers to use in the model.")
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to use in training/inference.")
    parser.add_argument("--alpha", type=int, help="Overwrite default CBM's alpha value.")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup epochs.")
    args = parser.parse_args()

    # Loads 'dataset' and prepares a random feature matrix 'dataset.x' for GCN 
    # inference. By default the shape of 'dataset.x' is () 
    dataset, alpha = load_dataset(args.dataset)
    dataset.x = rand((dataset.num_nodes, args.hidden_features))
    dataset.num_classes = args.hidden_features
    print_dataset_info(f"{args.dataset}", dataset)

    if args.alpha is not None:
        alpha = args.alpha

    # Represents adjacency matrix (for GCN inference)
    a = set_adjacency_matrix(args.nn, dataset.edge_index, alpha=alpha)
    
    # Model setup
    model = NodePrediction(args.nn, 
                           dataset.num_features, 
                           args.hidden_features, 
                           dataset.num_classes, 
                           args.num_hidden_layers, 
                           args.bias, a, None)

    
    # run and record GCN inference with --nn choices
    model.eval()
    performance = []
    with inference_mode():
        for epoch in range(1, args.warmup + args.epochs + 1):
            time_start = time()

            # forward pass
            y = model(x=dataset.x, edge_index=None)

            time_end = time()
            performance.append(time_end - time_start)
    performance = tensor(performance[args.warmup:])
    
    alpha_string= f" [alpha: {alpha}] " if "cbm-" in args.nn else " "
    print(f"[{args.nn}] [{args.dataset}]{alpha_string}[hidden_layers: {args.num_hidden_layers}] [hidden_features: {args.hidden_features}] Mean: {performance.mean():.6f} s   |   Std: {performance.std():.6f} s   |   Min: {performance.min():.6f} s   |   Max: {performance.max():.6f} s")