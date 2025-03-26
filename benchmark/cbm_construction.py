import argparse
from time import time
from torch import inference_mode, ones, zeros, randint, arange, empty, tensor
from utilities import load_dataset, print_dataset_info, set_adjacency_matrix

import warnings
warnings.simplefilter("ignore", UserWarning)

#def calculate_compression_ratio(csr_matrix, cbm_matrix):
#    # Calculate total number of elements in a_matrix representation
#    csr_size = (
#        csr_matrix.crow_indices().numel() +
#        csr_matrix.col_indices().numel() + 
#        csr_matrix.values().numel()
#    )
#
#    # Calculate total number of elements in c_matrix representation
#    cbm_size = (
#        cbm_matrix.deltas.crow_indices().numel() +
#        cbm_matrix.deltas.col_indices().numel() +
#        cbm_matrix.deltas.values().numel() +
#        cbm_matrix.mca_branches.numel() +
#        cbm_matrix.mca_row_idx.numel() +
#        cbm_matrix.mca_col_idx.numel()
#    )
#    
#    num_rows = cbm_matrix.crow_indices().numel() - 1
#    # Compute compression ratios for A, AD, and DAD
#    return csr_size / cbm_size, csr_size / (cbm_size + num_rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nn", choices=[
        "cbm-ax", "cbm-adx", "cbm-dadx",
        "mkl-ax", "mkl-adx", "mkl-dadx"
    ], required=True)
    parser.add_argument("--dataset", type=str, default="ca-HepPh")
    parser.add_argument("--iterations", type=int, default=50, help="Number of times format is built.")
    parser.add_argument("--alpha", type=int, help="Overwrite default alpha value for the adjacency matrix.")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations.")
    args = parser.parse_args()

    # dataset
    dataset, alpha = load_dataset(args.dataset)
    
    if args.alpha is not None:
        alpha = args.alpha
        
    elsapsed_time = []
    for iterations in range(1, args.warmup + args.iterations + 1):
        time_start = time()

        # adjacency matrix
        a, _ = set_adjacency_matrix(args.nn, dataset.edge_index, alpha)

        time_end = time()
        elsapsed_time.append(time_end - time_start)    
    elsapsed_time = tensor(elsapsed_time[args.warmup:])

    alpha_string= f" [alpha: {alpha}] " if "cbm-" in args.nn else " "
    print(f"[{args.nn}] [{args.dataset}]{alpha_string}[iterations: {args.iterations}] [warmups: {args.warmup}]   Mean: {elsapsed_time.mean():.6f} s   |   Std: {elsapsed_time.std():.6f} s   |   Min: {elsapsed_time.min():.6f} s   |   Max: {elsapsed_time.max():.6f} s")