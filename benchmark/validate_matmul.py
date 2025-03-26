import argparse
from torch import inference_mode, empty, rand, testing
from utilities import load_dataset, set_adjacency_matrix, print_dataset_info

import warnings
warnings.simplefilter("ignore", UserWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=["ax", "adx", "dadx"], required=True)
    parser.add_argument("--dataset", type=str, default="ca-HepPh")
    parser.add_argument("--columns", type=int, default=500, help="Number of columns in matrix 'x'.")
    parser.add_argument("--iterations", type=int, default=50, help="Number of matrix multiplications.")
    parser.add_argument("--alpha", type=int, help="Overwrite default alpha value for the adjacency matrix.")
    parser.add_argument("--atol", type=float, default=0)
    parser.add_argument("--rtol", type=float, default=1e-5)
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

    # Allocate empty tensors for result matrices.
    cbm_y = empty((dataset.num_nodes, args.columns if args.columns else dataset.num_features))
    mkl_y = empty((dataset.num_nodes, args.columns if args.columns else dataset.num_features))

    with inference_mode():  # Avoid computing gradients (probably not needed).  
        print("------------------------------------------------------------")
        for iteration in range(1, args.iterations + 1):
            x = rand((dataset.num_nodes, args.columns))
            
            # matrix multiplication
            cbm_a.matmul(x, cbm_y)
            mkl_a.matmul(x, mkl_y)
            pyg_y = pyg_a @ x

            # compare
            try:
                testing.assert_close(pyg_y, cbm_y, atol=args.atol, rtol=args.rtol)
                testing.assert_close(mkl_y, cbm_y, atol=args.atol, rtol=args.rtol)
                print(f"[{iteration}/{args.iterations}] PASSED")
            except AssertionError as e:
                print(f"[{iteration}/{args.iterations}] FAILED: {e}")
            print("------------------------------------------------------------")