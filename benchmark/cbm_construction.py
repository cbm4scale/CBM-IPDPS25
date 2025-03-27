"""
This script record the compression time and memory reduction of the CBM format.
    
The script converts the dataset specified in '--dataset' to CBM format, with 
'cbm/cbm4mm.py', as many times as specified by '--iterations'. At the end of 
execution, the script outputs benchmarking statistics including:
    - mean compression time and standard deviation
    - minimum and maximum compression time recorded.
    - compression ratio with respect to CSR format.

Example Usage:
    [OMP_ENV_VARS] python benchmark/cbm_construction.py --dataset COLLAB 

Note: 
    - See 'parser.add_arguments' for more options.
    - see 'benchmark/utilities.py' to check datasets options. 
"""

import argparse
from time import time
from torch import inference_mode, ones, zeros, randint, arange, empty, tensor
from utilities import load_dataset, print_dataset_info, set_adjacency_matrix

import warnings
warnings.simplefilter("ignore", UserWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ca-HepPh")
    parser.add_argument("--iterations", type=int, default=5, help="Overwrites default number of matrix multiplications tests.")
    parser.add_argument("--alpha", type=int, help="Overwrites default alpha value for the adjacency matrix.")
    parser.add_argument("--warmup", type=int, default=10, help="Overwrites default number of warmup iterations.")
    args = parser.parse_args()

    # Load dataset
    dataset, alpha = load_dataset(args.dataset)
    print_dataset_info(f"{args.dataset}", dataset)

    
    if args.alpha is not None:
        alpha = args.alpha

    a_cbm = set_adjacency_matrix('cbm-ax', dataset.edge_index, alpha=alpha)
    a_mkl = set_adjacency_matrix('mkl-ax', dataset.edge_index)

    # Calculate total number of elements in CSR representation
    csr_size = (
        a_mkl.a.crow_indices().numel() +
        a_mkl.a.col_indices().numel() + 
        a_mkl.a.values().numel()
    )

    # Calculate total number of elements in CBM representation
    cbm_size = (
        a_cbm.deltas.crow_indices().numel() +
        a_cbm.deltas.col_indices().numel() +
        a_cbm.deltas.values().numel() +
        a_cbm.mca_branches.numel() +
        a_cbm.mca_src_idx.numel() +
        a_cbm.mca_dst_idx.numel()
    )
    
    # Compute compression ratios for A, AD, and DAD
    compression_ratio = csr_size / cbm_size

    elsapsed_time = []
    with inference_mode():  # Avoid computing gradients (probably not needed).  
        for iterations in range(1, args.warmup + args.iterations + 1):
            time_start = time()

            # Convert adjacency matrix
            a = set_adjacency_matrix('cbm-ax', dataset.edge_index, alpha=alpha)

            time_end = time()
            elsapsed_time.append(time_end - time_start)

    elsapsed_time = tensor(elsapsed_time[args.warmup:])
    print(f"[cbm] [{args.dataset}] [alpha: {alpha}] [iterations: {args.iterations}] [warmups: {args.warmup}] Compression Ratio: {compression_ratio:.2f}x   Mean: {elsapsed_time.mean():.6f} s   |   Std: {elsapsed_time.std():.6f} s   |   Min: {elsapsed_time.min():.6f} s   |   Max: {elsapsed_time.max():.6f} s")
