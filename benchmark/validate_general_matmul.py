import torch
import argparse
from utilities import load_dataset, set_adjacency_matrix, print_dataset_info

import warnings
warnings.simplefilter("ignore", UserWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ca-HepPh")
    parser.add_argument("--columns", type=int, default=500, help="Number of columns to use in X. If not set, the original number of columns will be used.")
    parser.add_argument("--iterations", type=int, default=50, help="Number of matrix multiplication tests.")
    parser.add_argument("--alpha", type=int, help="Overwrite default alpha value for the adjacency matrix.")
    parser.add_argument("--atol", type=float, default=0)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--d_left", action="store_true", default=False, help="Generate a random diagonal matrix D1 (A = D1 @ A)")
    parser.add_argument("--d_right", action="store_true", default=False, help="Generate a random diagonal matrix D2 (A = A @ D2)")
    args = parser.parse_args()

    # dataset
    dataset, alpha = load_dataset(args.dataset)
    print_dataset_info(f"{args.dataset}", dataset)
    
    if args.alpha is not None:
        alpha = args.alpha

    d1 = None
    if args.d_left:
        d1 = torch.rand(dataset.num_nodes, dtype=float)

    d2 = None
    if args.d_right:
        d2 = torch.rand(dataset.num_nodes, dtype=float)

    cbm_a = set_adjacency_matrix(f"cbm", dataset.edge_index, d_left=d1, d_right=d2, alpha=alpha)
    mkl_a = set_adjacency_matrix(f"mkl", dataset.edge_index, d_left=d1, d_right=d2)
    pyg_a = mkl_a.a.clone()

    cbm_y = torch.empty((dataset.num_nodes, args.columns))
    mkl_y = torch.empty((dataset.num_nodes, args.columns))
    #pyg_y = torch.empty((dataset.num_nodes, args.columns)) 

    print("------------------------------------------------------------")
    with torch.inference_mode():
        for iteration in range(1, args.iterations + 1):
            x = torch.rand((dataset.num_nodes, args.columns))
            
            # matrix multiplication
            cbm_a.matmul(x, cbm_y)
            mkl_a.matmul(x, mkl_y)
            pyg_y = pyg_a @ x

            # compare
            try:
                torch.testing.assert_close(pyg_y, cbm_y, atol=args.atol, rtol=args.rtol)
                torch.testing.assert_close(mkl_y, cbm_y, atol=args.atol, rtol=args.rtol)
                print(f"[{iteration}/{args.iterations}] PASSED")
            except AssertionError as e:
                print(f"[{iteration}/{args.iterations}] FAILED: {e}")
            print("------------------------------------------------------------")