"""
This script records the execution time of the matrix multiplication methods in:
    - 'cbm/cbm4mm.py'       - 'cbm/mkl4mm.py'  
    - 'cbm/cbm4ad.py'       - 'cbm/mkl4ad.py'
    - 'cbm/cbm4dad.py'      - 'cbm/mkl4dad.py'
    
It converts the dataset specified by '--dataset' into one of the following:
    - A (processed by 'cbm/cbm4mm.py' or 'cbm/mkl4mm.py')
    - A @ D^(-1/2) (processed by 'cbm/cbm4ad.py' or 'cbm/mkl4ad.py')
    - D^(-1/2) @ A @ D^(-1/2) (processed by 'cbm/cbm4dad.py' or 'cbm/mkl4dad.py')

Where:
    - A is the adjacency matrix of the dataset.
    - D is the diagonal degree matrix of A.

The script then performs '--iterations' matrix multiplications between the 
converted matrices and a randomly generated matrix 'x' with '--columns' columns.
At the end of execution, the script outputs benchmarking statistics including:
    - mean matrix multiplication time and standard deviation.
    - minimum and maximum matrix multiplication time recorded.

Example Usage:
    [OMP_ENV_VARS] python benchmark/benchmark_matmul.py --operation cbm-adx  

Note: 
    - See 'parser.add_arguments' for more options.
    - see 'benchmark/utilities.py' for datasets options. 
"""

import argparse
from time import time
from torch import inference_mode, rand, empty, tensor
from utilities import load_dataset, print_dataset_info, set_adjacency_matrix

import warnings
warnings.simplefilter("ignore", UserWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=[
        "cbm-ax", "cbm-adx", "cbm-dadx",
        "mkl-ax", "mkl-adx", "mkl-dadx"
    ], required=True)
    parser.add_argument("--dataset", type=str, default="ca-HepPh")
    parser.add_argument("--columns", type=int, default=500, help="Overwrites default number of columns in matrix 'x'.")
    parser.add_argument("--iterations", type=int, default=50, help="Overwrites default number of matrix multiplications tests.")
    parser.add_argument("--alpha", type=int, help="Overwrites default alpha value for the adjacency matrix.")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations.")
    args = parser.parse_args()

    # Load dataset
    dataset, alpha = load_dataset(args.dataset)
    print_dataset_info(f"{args.dataset}", dataset)
    
    if args.alpha is not None:
        alpha = args.alpha
    
    # Convert adjacency matrices in the format specified in '--operation'
    a = set_adjacency_matrix(args.operation, dataset.edge_index, alpha=alpha)

    performance = []
    with inference_mode():
        x = rand((dataset.num_nodes, args.columns))
        y = empty((dataset.num_nodes, args.columns))
        for iterations in range(1, args.warmup + args.iterations + 1):
            time_start = time()

            # matrix multiplication
            a.matmul(x, y)

            time_end = time()
            performance.append(time_end - time_start)
    
    performance = tensor(performance[args.warmup:])

    alpha_string= f" [alpha: {alpha}] " if "cbm-" in args.operation else " "
    print(f"[{args.operation}] [{args.dataset}]{alpha_string}[columns: {args.columns}] [iterations: {args.iterations}] [warmups: {args.warmup}]   Mean: {performance.mean():.6f} s   |   Std: {performance.std():.6f} s   |   Min: {performance.min():.6f} s   |   Max: {performance.max():.6f} s")
