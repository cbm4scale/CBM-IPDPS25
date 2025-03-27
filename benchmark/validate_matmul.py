"""
This script validates matrix multiplication using the following methods:
    - `cbm/cbm4mm.py`
    - `cbm/cbm4ad.py`
    - `cbm/cbm4dad.py`

It converts the dataset specified by '--dataset' into one of the following:
    - A (processed by 'cbm/cbm4mm.py' or 'cbm/mkl4mm.py')
    - A @ D^(-1/2) (processed by 'cbm/cbm4ad.py' or 'cbm/mkl4ad.py')
    - D^(-1/2) @ A @ D^(-1/2) (processed by 'cbm/cbm4dad.py' or 'cbm/mkl4dad.py')

Where:
    - A is the adjacency matrix of the dataset.
    - D is the diagonal degree matrix of A.

The resulting matrix is simultaneously converted into CSR and CBM formats. 
The script then performs '--iterations' matrix multiplications between these 
matrices and a randomly generated matrix 'x' with '--columns' columns.  
The resulting matrices ('csr_y' and 'cbm_y') are compared to validate  
matrix multiplication using the CBM format.  

By default, matrix comparison uses an absolute tolerance (--atol) and relative 
tolerace (--rtol) equal to '0' and '1e-5', but these values can be modified. 

Example Usage:
    [OMP_ENV_VARS] python benchmark/validate_generic_matmul --operation adx  

Note: 
    - See 'parser.add_arguments' for more options.
    - see 'benchmark/utilities.py' to see available dataset options. 
"""

import argparse
from torch import inference_mode, empty, rand, testing
from utilities import load_dataset, set_adjacency_matrix, print_dataset_info

import warnings
warnings.simplefilter("ignore", UserWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=["ax", "adx", "dadx"], required=True)
    parser.add_argument("--dataset", type=str, default="ca-HepPh")
    parser.add_argument("--columns", type=int, default=500, help="Overwrites default number of columns in matrix 'x'.")
    parser.add_argument("--iterations", type=int, default=50, help="Overwrites default number of matrix multiplications tests.")
    parser.add_argument("--alpha", type=int, help="Overwrites default alpha value for the adjacency matrix.")
    parser.add_argument("--atol", type=float, default=0, help="Overwrites default absolute tolerance.")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Overwrites default relative tolerance.")
    args = parser.parse_args()

    # Load Dataset
    dataset, alpha = load_dataset(args.dataset)
    print_dataset_info(f"{args.dataset}", dataset)
    
    if args.alpha is not None:
        alpha = args.alpha
    
    # Represent adjacency matrices in CBM and CSR formats.
    cbm_a = set_adjacency_matrix(f"cbm-{args.operation}", dataset.edge_index, alpha=alpha)
    mkl_a = set_adjacency_matrix(f"mkl-{args.operation}", dataset.edge_index)
    pyg_a = mkl_a.a.clone()

    cbm_y = empty((dataset.num_nodes, args.columns))
    mkl_y = empty((dataset.num_nodes, args.columns))

    with inference_mode():  # Avoid computing gradients (probably not needed).  
        print("------------------------------------------------------------")
        for iteration in range(1, args.iterations + 1):
            x = rand((dataset.num_nodes, args.columns))
            
            # matrix multiplication
            cbm_a.matmul(x, cbm_y)
            mkl_a.matmul(x, mkl_y)
            pyg_y = pyg_a @ x

            try:
                testing.assert_close(pyg_y, cbm_y, atol=args.atol, rtol=args.rtol)
                testing.assert_close(mkl_y, cbm_y, atol=args.atol, rtol=args.rtol)
                print(f"[{iteration}/{args.iterations}] PASSED")
            except AssertionError as e:
                print(f"[{iteration}/{args.iterations}] FAILED: {e}")
            print("------------------------------------------------------------")